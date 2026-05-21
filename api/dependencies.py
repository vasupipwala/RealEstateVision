"""
Shared dependencies: DB connection, model registry, active model state.
"""

import sqlite3
from pathlib import Path
from typing import Generator

from fastapi import HTTPException

DB_PATH = Path(__file__).parent.parent / "db" / "realestatevision.db"
MODELS_DIR = Path(__file__).parent.parent / "models"

_state: dict = {
    "model": None,
    "model_name": None,
    "version_tag": None,
    "device": None,
}


def get_state() -> dict:
    return _state


def get_db() -> Generator[sqlite3.Connection, None, None]:
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def require_model() -> dict:
    if _state["model"] is None:
        raise HTTPException(
            status_code=503,
            detail="No model loaded. POST /models/load?model_name=...&version_tag=... first.",
        )
    return _state
