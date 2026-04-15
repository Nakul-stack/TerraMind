import sys
import numpy as np
from pathlib import Path

if __name__ == "__main__":
    _root = Path(__file__).resolve().parents[2]
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

from ml.growth_stage_monitor import config
from ml.growth_stage_monitor.utils import get_logger, load_artifact
from ml.growth_stage_monitor.preprocessing import transform_input
from ml.growth_stage_monitor.model import _augment_features_for_stage2

logger = get_logger(__name__)

_models = None
_preprocessor = None
_pest_encoder = None
_fert_encoder = None

def _load_artifacts():
    global _models, _preprocessor, _pest_encoder, _fert_encoder
    if _models is not None:
        return
        
    logger.info("Loading 2-Stage Model Artifacts...")
    _preprocessor = load_artifact(config.PREPROCESSING_PIPELINE_PATH)
    _pest_encoder = load_artifact(config.PEST_LEVEL_ENCODER_PATH)
    _fert_encoder = load_artifact(config.FERTILIZER_ENCODER_PATH)
    
    _models = {
        "pest_level": load_artifact(config.PEST_LEVEL_MODEL_PATH),
        "recommended_fertilizer": load_artifact(config.FERTILIZER_MODEL_PATH),
        "dosage": load_artifact(config.DOSAGE_MODEL_PATH),
        "apply_after_days": load_artifact(config.APPLY_AFTER_DAYS_MODEL_PATH),
        "expected_yield_after_dosage": load_artifact(config.EXPECTED_YIELD_MODEL_PATH),
    }

def predict_growth_stage_advisory(input_data: dict) -> dict:
    _load_artifacts()
    
    X_trans = transform_input(input_data, _preprocessor)
    
    pest_pred_encoded = _models["pest_level"].predict(X_trans).astype(np.int64)
    fert_pred_encoded = _models["recommended_fertilizer"].predict(X_trans).astype(np.int64)
    
    pest_level_str = _pest_encoder.inverse_transform(pest_pred_encoded)[0]
    fertilizer_str = _fert_encoder.inverse_transform(fert_pred_encoded)[0]
    
    X_trans_aug = _augment_features_for_stage2(X_trans, pest_pred_encoded, fert_pred_encoded)
    
    dosage_pred = max(0.0, float(_models["dosage"].predict(X_trans_aug)[0]))
    apply_days_pred = max(1, int(round(_models["apply_after_days"].predict(X_trans_aug)[0])))
    yield_pred = max(0.0, float(_models["expected_yield_after_dosage"].predict(X_trans_aug)[0]))
    
    return {
        "recommended_fertilizer": fertilizer_str,
        "pest_level": pest_level_str,
        "dosage": round(dosage_pred, 1),
        "apply_after_days": apply_days_pred,
        "expected_yield_after_dosage": round(yield_pred, 1)
    }

def main():
    sample_input = {
        "temperature": 27.3,
        "humidity": 47.0,
        "moisture": 58.0,
        "soil_type": "Loamy",
        "crop_type": "Cotton",
        "N": 81.0,
        "P": 28.0,
        "K": 41.0,
        "ph": 6.0,
        "rainfall": 246.0
    }
    
    logger.info("Executing sample inference...")
    result = predict_growth_stage_advisory(sample_input)
    logger.info(f"Input Features: {sample_input}")
    logger.info(f"Predictions: {result}")

if __name__ == "__main__":
    main()
