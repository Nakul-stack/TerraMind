import logging
from app.schemas.monitor import GrowthStageRequest, GrowthStageResponse
from ml.growth_stage_monitor.predict import predict_growth_stage_advisory

logger = logging.getLogger(__name__)

def run_growth_stage_pipeline(input_data: GrowthStageRequest) -> GrowthStageResponse:
    """
    Orchestrates the 5-stage causal ML pipeline by delegating to the ML layer wrapper function.
    """
    logger.info("Executing the Growth Stage Monitor pipeline...")

    # Convert Pydantic request mapped appropriately to dict
    inference_input = input_data.model_dump()
    
    try:
        raw_result = predict_growth_stage_advisory(inference_input)
    except Exception as e:
        logger.error(f"Failed to infer from ML Pipeline: {e}")
        raise RuntimeError(f"Model inference failed: {str(e)}")

    logger.info("ML execution complete, packaging response...")
    
    # Map dictionary returned from the pipeline into Pydantic schema
    response = GrowthStageResponse(
        recommended_fertilizer=raw_result["recommended_fertilizer"],
        pest_level=raw_result["pest_level"],
        dosage=raw_result["dosage"],
        apply_after_days=raw_result["apply_after_days"],
        expected_yield_after_dosage=raw_result["expected_yield_after_dosage"]
    )
    
    return response
