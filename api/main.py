"""
RealEstateVision — FastAPI Application
Run: uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
Docs: http://localhost:8000/docs
"""

from __future__ import annotations

import os
import sqlite3
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.dependencies import DB_PATH, _state
from api.inference import get_device, load_model
from api.routers import dashboard as dashboard_router
from api.routers import models as models_router
from api.routers import predict as predict_router
from api.routers import runs as runs_router
from api.routers import data_validation as data_validation_router
from api.schemas import HealthResponse

FALLBACK_MODEL = os.getenv("DEFAULT_MODEL", "mobilenet_v3_small")
FALLBACK_VERSION_TAG = os.getenv("DEFAULT_VERSION_TAG", "cleaned_v1")
API_VERSION = "1.3.0"


def _resolve_startup_recommendation() -> tuple[str, str, str]:
    """
    Decide which model/version to load at startup.

    Priority:
    1. Dashboard recommendation for the configured/default dataset version.
    2. Environment fallback values.
    """
    requested_version_tag = FALLBACK_VERSION_TAG

    try:
        models = dashboard_router._load_models_from_csv(requested_version_tag)
        if models:
            ranked = dashboard_router._score_models(models)
            if ranked:
                recommended_model = ranked[0].get("model")
                recommended_version = ranked[0].get("version_tag") or requested_version_tag
                if recommended_model:
                    return recommended_model, recommended_version, "dashboard_recommendation"
    except Exception as exc:
        print(f"[startup] Recommendation lookup failed: {exc}")

    return FALLBACK_MODEL, FALLBACK_VERSION_TAG, "env_fallback"


def _try_load_startup_model(device: str) -> tuple[object | None, str | None, str | None]:
    """
    Load the recommended model first. If that artifact is unavailable, fall back once more
    to the env-configured default model/version.
    """
    model_name, version_tag, source = _resolve_startup_recommendation()
    print(f"[startup] Startup selection source: {source}")
    print(f"[startup] Loading {model_name} ({version_tag}) ...")

    try:
        model = load_model(model_name, version_tag, device)
        return model, model_name, version_tag
    except FileNotFoundError as exc:
        print(f"[startup] WARNING: {exc}")

        if (model_name, version_tag) != (FALLBACK_MODEL, FALLBACK_VERSION_TAG):
            print(f"[startup] Falling back to {FALLBACK_MODEL} ({FALLBACK_VERSION_TAG}) ...")
            try:
                model = load_model(FALLBACK_MODEL, FALLBACK_VERSION_TAG, device)
                return model, FALLBACK_MODEL, FALLBACK_VERSION_TAG
            except FileNotFoundError as fallback_exc:
                print(f"[startup] WARNING: {fallback_exc}")

    return None, None, None


@asynccontextmanager
async def lifespan(app: FastAPI):
    device = get_device()
    _state["device"] = device
    print(f"[startup] Device: {device}")

    model, model_name, version_tag = _try_load_startup_model(device)
    _state["model"] = model
    _state["model_name"] = model_name
    _state["version_tag"] = version_tag

    if model is not None:
        print(f"[startup] Model ready: {model_name} ({version_tag}).")
    else:
        print("[startup] WARNING: No startup model could be loaded.")

    yield

    print("[shutdown] Cleaning up.")
    _state["model"] = None
    _state["model_name"] = None
    _state["version_tag"] = None


app = FastAPI(
    title="RealEstateVision API",
    description=(
        "Room type classification API — part of the RealEstateVision ML pipeline. \n"
        """
        RealEstateVision is an end-to-end, production-oriented machine learning framework designed to simulate how modern real-estate tech companies process, validate, benchmark, and serve property image intelligence workflows.\n
        
        This project focuses on building a reliable system around data ingestion, metadata engineering, data quality monitoring using CleanVision package, reproducible experimentation, and evidence-based build-vs-buy decisions under real-world constraints such as quality (or accuracy), latency (or performance), cost, and maintainability.\n
        
        This project is intentionally narrowed to "indoor room-type image classification" (such as bathroom, bedroom, diningroom, kitchen, livingroom) using the MIT Indoor Scenes dataset with their official train/test split for each room type / class. This enabled me to keep the project aligned with property imagery while creating this reproducible workflow.
        """
    ),
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
app.include_router(data_validation_router.router)
app.include_router(dashboard_router.router)
app.include_router(runs_router.router)


@app.get("/health", response_model=HealthResponse, tags=["System"])
def health():
    """
    Check the operational health of the RealEstateVision API
    """
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
        "routes": {
            #"predict": "/predict",
            "models": "/models",
            #"validation": "/validation",
            "dashboard_summary": "/dashboard/summary",
            "dashboard_models_compare": "/dashboard/models/compare",
            "dashboard_recommendation": "/dashboard/recommendation",
            "runs": "/runs",
        },
        "startup_default": {
        "selection_policy": "dashboard recommendation first, env fallback second",
        "active_model": _state.get("model_name") or "none",
        "active_version_tag": _state.get("version_tag") or "none",
        },
    }
