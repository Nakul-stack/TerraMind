"""
API routes for the Pre-Sowing Advisor.

Endpoints:
    POST /predict       - Run the full advisory pipeline
    POST /train/all     - Train all 3 models
    GET  /metadata      - Model metadata & versions
"""

import json
import logging
import threading
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import JSONResponse

from backend.app.core.rate_limit import limiter
from backend.app.core.runtime_config import (
    ADVISOR_PREDICT_RATE_LIMIT,
    ADVISOR_TRAIN_RATE_LIMIT,
    API_RATE_LIMIT,
)
from backend.app.schemas.advisor import BeforeSowingRequest
from ml.pre_sowing_pipeline import run_standard_pipeline

router = APIRouter()
logger = logging.getLogger(__name__)

_federated_advisor = None
_federated_advisor_lock = threading.Lock()


def get_federated_advisor():
    """Lazy-load the federated advisor."""
    global _federated_advisor
    if _federated_advisor is None:
        with _federated_advisor_lock:
            if _federated_advisor is None:
                try:
                    from federated.inference import FederatedAdvisor
                    _federated_advisor = FederatedAdvisor()
                    if _federated_advisor.is_available():
                        logger.info("[TerraMind] Federated model ready")
                    else:
                        logger.warning("[TerraMind] Federated model not loaded")
                except ImportError:
                    logger.warning("[TerraMind] Federated module not available")
                    _federated_advisor = None
    return _federated_advisor


# ---------------------------------------------------------------------------
# POST /predict
# ---------------------------------------------------------------------------

@router.post(
    "/predict",
    status_code=status.HTTP_200_OK,
    summary="Predict Pre-Sowing Advisory",
    description="Run the unified advisory pipeline: crop recommendation -> yield -> irrigation -> district intelligence.",
)
@limiter.limit(ADVISOR_PREDICT_RATE_LIMIT)
def predict_pre_sowing_advisory(request: Request, body_request: BeforeSowingRequest):
    """Route to the standard or federated pipeline."""
    logger.info("Received prediction request. mode=%s", body_request.model_mode)

    # -- Federated path -------------------------------------------------
    if body_request.model_mode == "federated":
        advisor = get_federated_advisor()
        if advisor is None or not advisor.is_available():
            return JSONResponse(
                status_code=503,
                content={
                    "success": False,
                    "error": "Federated model is not available",
                    "fallback_suggestion": "Use model_mode='standard'.",
                },
            )
        result = advisor.predict(
            N=body_request.N, P=body_request.P, K=body_request.K,
            ph=body_request.ph, temperature=body_request.temperature,
            humidity=body_request.humidity, rainfall=body_request.rainfall,
        )
        return JSONResponse(content=result)

    # -- Standard pipeline -----------------------------------------------
    try:
        response = run_standard_pipeline(body_request)
        logger.info("Returning prediction results.")
        return JSONResponse(content=response)
    except KeyError as e:
        logger.error("Missing input: %s", e)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except ValueError as e:
        logger.error("Invalid input: %s", e)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except RuntimeError as e:
        logger.error("Pipeline error: %s", e)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
    except Exception as e:
        logger.error("Unexpected error: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred processing the pipeline.",
        )


# ---------------------------------------------------------------------------
# POST /train/all
# ---------------------------------------------------------------------------

@router.post(
    "/train/all",
    status_code=status.HTTP_200_OK,
    summary="Train All Models",
    description="Retrain all 3 pre-sowing models and save artifacts.",
)
@limiter.limit(ADVISOR_TRAIN_RATE_LIMIT)
def train_all_models(request: Request):
    """Train all 3 models sequentially."""
    results = {}

    # Model 1: Crop Recommender
    try:
        from ml.pre_sowing_advisor.crop_recommendation.train import main as train_crop
        train_crop()
        results["crop_recommender"] = {"status": "success"}
    except Exception as e:
        results["crop_recommender"] = {"status": "error", "detail": str(e)}

    # Model 2: Yield Predictor
    try:
        from ml.pre_sowing_advisor.yield_prediction.train import main as train_yield
        train_yield()
        results["yield_predictor"] = {"status": "success"}
    except Exception as e:
        results["yield_predictor"] = {"status": "error", "detail": str(e)}

    # Model 3: Pre-Sowing Advisor
    try:
        from ml.pre_sowing_advisor.irrigation_sunlight.train import main as train_irr
        train_irr()
        results["agri_condition_advisor"] = {"status": "success"}
    except Exception as e:
        results["agri_condition_advisor"] = {"status": "error", "detail": str(e)}

    all_ok = all(r["status"] == "success" for r in results.values())
    return JSONResponse(content={"success": all_ok, "results": results})


# ---------------------------------------------------------------------------
# GET /metadata
# ---------------------------------------------------------------------------

@router.get(
    "/metadata",
    status_code=status.HTTP_200_OK,
    summary="Model Metadata",
    description="Show model versions, training dates, metrics, and datasets used.",
)
def get_model_metadata():
    """Returns metadata for all 3 models."""
    metadata = {}
    project_root = Path(__file__).resolve().parents[4]
    model_dirs = {
        "crop_recommender": project_root / "ml" / "pre_sowing_advisor" / "crop_recommendation" / "saved_models",
        "yield_predictor": project_root / "ml" / "pre_sowing_advisor" / "yield_prediction" / "saved_models",
        "agri_condition_advisor": project_root / "ml" / "pre_sowing_advisor" / "irrigation_sunlight" / "saved_models",
    }

    for name, path in model_dirs.items():
        meta_file = path / "metadata.json"
        if meta_file.exists():
            with open(meta_file) as f:
                metadata[name] = json.load(f)
        else:
            metadata[name] = {"status": "not trained", "path": str(meta_file)}

    return JSONResponse(content=metadata)


# ---------------------------------------------------------------------------
# POST /compare  (legacy - kept for backward compat)
# ---------------------------------------------------------------------------

@router.post(
    "/compare",
    status_code=status.HTTP_200_OK,
    summary="Compare Standard and Federated",
)
@limiter.limit(API_RATE_LIMIT)
def compare_models(request: Request, body_request: BeforeSowingRequest):
    """Run both Standard and Federated models side-by-side."""
    results = {
        "input": {
            "N": body_request.N, "P": body_request.P, "K": body_request.K,
            "ph": body_request.ph, "temperature": body_request.temperature,
            "humidity": body_request.humidity, "rainfall": body_request.rainfall,
        },
        "standard_model": None,
        "federated_model": None,
    }

    try:
        std = run_standard_pipeline(body_request)
        results["standard_model"] = {
            "success": True,
            "predicted_crop": std.get("recommended_crop"),
            "model_type": "Standard (Unified Pipeline)",
        }
    except Exception as e:
        results["standard_model"] = {"success": False, "error": str(e)}

    fed = get_federated_advisor()
    if fed is not None and fed.is_available():
        fed_result = fed.predict(
            body_request.N, body_request.P, body_request.K,
            body_request.ph, body_request.temperature,
            body_request.humidity, body_request.rainfall,
        )
        results["federated_model"] = fed_result
    else:
        results["federated_model"] = {"success": False, "error": "Federated model not available"}

    return JSONResponse(content=results)
