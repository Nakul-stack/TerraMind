import logging
from fastapi import APIRouter, HTTPException, status, Request
from backend.app.core.rate_limit import limiter
from backend.app.core.runtime_config import MONITOR_PREDICT_RATE_LIMIT

from backend.app.schemas.monitor import GrowthStageRequest, GrowthStageResponse
from backend.app.services.growth_stage_service import run_growth_stage_pipeline

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post(
    "/predict",
    response_model=GrowthStageResponse,
    status_code=status.HTTP_200_OK,
    summary="Predict During-Growth Advisory",
    description="Run the 5-stage causal ML Pipeline for growth advisory."
)
@limiter.limit(MONITOR_PREDICT_RATE_LIMIT)
def predict_growth_stage_advisory(request: Request, body_request: GrowthStageRequest):
    """
    Endpoint mapping to during-growth monitor ML models:
    - Analyzes numeric + categorical soil parameters.
    - Yields stage dependencies resolving fertilizer, pest level, dosage, application timeline, and expected output.
    """
    logger.info("Received request for growth-stage prediction.")
    try:
        response_model = run_growth_stage_pipeline(body_request)
        return response_model
    except ValueError as e:
        logger.error(f"Invalid input metrics parsed: {e}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except FileNotFoundError as e:
        logger.error(f"Model artifacts missing from saved configs: {e}")
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Model unavailable down the pipeline.")
    except RuntimeError as e:
        logger.error(f"Execution Error within AI Pipeline: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
    except Exception as e:
        logger.error(f"Unhandled Error: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Pipeline critical failure.")
