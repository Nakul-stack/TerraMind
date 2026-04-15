"""
TerraMind - Edge Inference Service

Thin wrapper that ensures edge mode uses local artifacts and caches
without hitting the central server.
"""
from __future__ import annotations

from backend.services.inference_pipeline import get_pipeline
from backend.core.logging_config import log


def edge_predict(raw_input: dict) -> dict:
    """Force edge mode and run inference."""
    raw_input["mode"] = "edge"
    pipeline = get_pipeline()
    result = pipeline.predict(raw_input)
    result["execution_mode"] = "edge"
    return result
