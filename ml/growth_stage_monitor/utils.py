import logging
import json
import joblib
from pathlib import Path
from typing import Any

def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        # Force ASCII formatting
        fmt = logging.Formatter("%(asctime)s | %(name)-35s | %(levelname)-8s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
        ch = logging.StreamHandler()
        ch.setFormatter(fmt)
        logger.addHandler(ch)
    return logger

logger = get_logger(__name__)

def save_artifact(obj: Any, path: str | Path) -> None:
    joblib.dump(obj, path)

def load_artifact(path: str | Path) -> Any:
    return joblib.load(path)

def save_json(obj: Any, path: str | Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)

def load_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
