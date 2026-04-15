"""Shared utilities for the Crop Recommendation module."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import joblib


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def load_artifact(path: Path) -> Any:
    """Load a joblib-persisted artifact."""
    if not path.exists():
        raise FileNotFoundError(f"Artifact not found: {path}")
    return joblib.load(path)


def save_artifact(obj: Any, path: Path) -> None:
    """Persist an object via joblib."""
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, path)


def save_json(data: dict, path: Path) -> None:
    """Save dict as JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)


def load_json(path: Path) -> dict:
    """Load JSON file."""
    with open(path) as f:
        return json.load(f)
