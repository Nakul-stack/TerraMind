"""
TerraMind - Prediction API routes.
"""
from __future__ import annotations

import json
from fastapi import APIRouter, HTTPException
from backend.schemas.request_response import PredictRequest, PredictResponse
from backend.services.inference_pipeline import get_pipeline
from backend.core.logging_config import log

router = APIRouter(tags=["prediction"])


@router.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest):
    """
    Run the full pre-sowing prediction pipeline.

    Supports three modes:
      - **central**: Full-power centralized model (gold standard)
      - **edge**: Compressed model + local adaptation layer
      - **local_only**: Local-only benchmark model
    """
    try:
        pipeline = get_pipeline()
        result = pipeline.predict(req.model_dump())
        return result
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=f"Models not trained yet: {exc}")
    except Exception as exc:
        log.error("Prediction error: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/metadata")
async def metadata():
    """Return model versions, training dates, and dataset info."""
    from backend.core.config import CENTRAL_ARTIFACTS
    meta = {}
    for model_name in ["crop_recommender", "yield_predictor", "agri_advisor"]:
        path = CENTRAL_ARTIFACTS / model_name / "metadata.json"
        if path.exists():
            with open(path) as f:
                meta[model_name] = json.load(f)
        else:
            meta[model_name] = {"status": "not trained"}
    return meta
