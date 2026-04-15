"""
Pre-Sowing Advisor service - wraps the ML pipeline for the API layer.
"""

import logging
from typing import Dict, Any
from app.schemas.advisor import BeforeSowingRequest

from ml.pre_sowing_pipeline import run_standard_pipeline

logger = logging.getLogger(__name__)


def run_pre_sowing_pipeline(input_data: BeforeSowingRequest) -> Dict[str, Any]:
    """
    Orchestrates the unified pre-sowing advisory pipeline.
    Returns a dict matching the BeforeSowingResponse schema.
    """
    logger.info("Starting Pre-Sowing Advisory pipeline (unified).")

    # The new pipeline handles everything internally
    result = run_standard_pipeline(input_data)

    logger.info("Pipeline executed successfully -> crop: %s", result.get("recommended_crop"))
    return result
