"""
RealEstateVision — FastAPI Application
Run: uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
Docs: http://localhost:8000/docs
"""

import os
import sqlite3
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.dependencies import DB_PATH, _state
from api.inference import get_device, load_model
from api.routers import models as models_router
from api.routers import predict as predict_router
from api.routers import runs as runs_router
from api.routers import validation as validation_router
from api.schemas import HealthResponse

DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "mobilenet_v3_small")
DEFAULT_VERSION_TAG = os.getenv("DEFAULT_VERSION_TAG", "cleaned_v1")
API_VERSION = "1.0.0"


@asynccontextmanager
async def lifespan(app: FastAPI):
    device = get_device()
    _state["device"] = device
    print(f"[startup] Device: {device}")
    print(f"[startup] Loading {DEFAULT_MODEL} ({DEFAULT_VERSION_TAG}) ...")

    try:
        _state["model"] = load_model(DEFAULT_MODEL, DEFAULT_VERSION_TAG, device)
        _state["model_name"] = DEFAULT_MODEL
        _state["version_tag"] = DEFAULT_VERSION_TAG
        print("[startup] Model ready.")
    except FileNotFoundError as exc:
        print(f"[startup] WARNING: {exc}")

    yield

    print("[shutdown] Cleaning up.")
    _state["model"] = None


app = FastAPI(
    title="RealEstateVision API",
    description="Room type classification API — part of the RealEstateVision ML pipeline.",
    version=API_VERSION,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(predict_router.router)
app.include_router(models_router.router)
app.include_router(validation_router.router)
app.include_router(runs_router.router)


@app.get("/health", response_model=HealthResponse, tags=["System"])
def health():
    db_ok = False
    try:
        conn = sqlite3.connect(str(DB_PATH))
        conn.execute("SELECT 1 FROM images LIMIT 1;")
        conn.close()
        db_ok = True
    except Exception:
        pass

    mlflow_ok = False
    try:
        import mlflow
        client = mlflow.tracking.MlflowClient()
        client.search_experiments()
        mlflow_ok = True
    except Exception:
        pass

    return HealthResponse(
        status="ok" if (_state["model"] is not None and db_ok) else "degraded",
        db_ok=db_ok,
        model_loaded=_state["model"] is not None,
        active_model=_state.get("model_name") or "none",
        device=str(_state.get("device") or "unknown"),
        mlflow_ok=mlflow_ok,
        version=API_VERSION,
    )


@app.get("/", tags=["System"])
def root():
    return {
        "project": "RealEstateVision",
        "docs": "/docs",
        "health": "/health",
    }
