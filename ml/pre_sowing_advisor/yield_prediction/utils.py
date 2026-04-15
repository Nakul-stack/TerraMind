"""Shared utilities for the Yield Prediction module."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

import joblib


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def normalise_string(s: str) -> str:
    """Lowercase, trim, collapse whitespace."""
    s = str(s).strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


def load_artifact(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"Artifact not found: {path}")
    return joblib.load(path)


def save_artifact(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, path)


def save_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
