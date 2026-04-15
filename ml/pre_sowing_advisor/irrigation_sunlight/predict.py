"""
Inference module for Model 3 — Pre-Sowing Advisor.

Returns sunlight_hours, irrigation_type, irrigation_need,
plus fields for district prior integration.

Usage:
    from ml.pre_sowing_advisor.irrigation_sunlight.predict import predict_irrigation_advisory
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

if __name__ == "__main__":
    _root = Path(__file__).resolve().parents[3]
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

from ml.pre_sowing_advisor.irrigation_sunlight import config
from ml.pre_sowing_advisor.irrigation_sunlight.utils import get_logger, load_artifact, normalise_string

logger = get_logger(__name__)

_preprocessor = None
_model_sunlight = None
_model_irr_type = None
_model_irr_need = None

_INPUT_KEY_MAP: dict[str, str] = {
    "crop": config.RAW_CROP_COL,
    "ph": config.RAW_PH_COL,
    "temperature": config.RAW_TEMP_COL,
    "humidity": config.RAW_HUMIDITY_COL,
    "rainfall": config.RAW_RAINFALL_COL,
    "soil_type": config.RAW_SOIL_COL,
    "season": config.RAW_SEASON_COL,
}
_NUMERIC_KEYS = {"ph", "temperature", "humidity", "rainfall"}
_CATEGORICAL_KEYS = {"crop", "soil_type", "season"}


def _load_artifacts() -> None:
    global _preprocessor, _model_sunlight, _model_irr_type, _model_irr_need
    if _preprocessor is not None:
        return
    logger.info("Loading pre-sowing advisor artifacts …")
    _preprocessor = load_artifact(config.PREPROCESSOR_PATH)
    _model_sunlight = load_artifact(config.MODEL_SUNLIGHT_PATH)
    _model_irr_type = load_artifact(config.MODEL_IRR_TYPE_PATH)
    _model_irr_need = load_artifact(config.MODEL_IRR_NEED_PATH)
    logger.info("Pre-sowing advisor artifacts loaded.")


def predict_irrigation_advisory(input_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Predict sunlight, irrigation type, and irrigation need.

    Args:
        input_dict: Must contain crop, ph, temperature, humidity,
                    rainfall, soil_type, season.

    Returns:
        {
            "sunlight_hours": float,
            "irrigation_type": str,
            "irrigation_need": str,
            "irrigation_type_probabilities": dict (if available),
            "explanation": str,
        }
    """
    _load_artifacts()

    missing = [k for k in _INPUT_KEY_MAP if k not in input_dict]
    if missing:
        raise KeyError(f"Missing required input keys: {missing}")

    # Build single-row DataFrame
    row: Dict[str, Any] = {}
    for api_key, col_name in _INPUT_KEY_MAP.items():
        val = input_dict[api_key]
        if api_key in _CATEGORICAL_KEYS:
            row[col_name] = normalise_string(str(val))
        else:
            row[col_name] = float(val)

    # Add engineered features (must match preprocessing.engineer_features)
    import numpy as _np
    row["temp_rainfall_ratio"] = row[config.RAW_TEMP_COL] / (row[config.RAW_RAINFALL_COL] + 1)
    row["humidity_rainfall"] = row[config.RAW_HUMIDITY_COL] * row[config.RAW_RAINFALL_COL] / 100.0
    row["rainfall_log"] = float(_np.log1p(row[config.RAW_RAINFALL_COL]))
    row["temp_humidity_interaction"] = row[config.RAW_TEMP_COL] * row[config.RAW_HUMIDITY_COL] / 100.0

    df = pd.DataFrame([row])
    X = _preprocessor.transform(df)

    # Predict
    sun_pred = float(max(0.0, _model_sunlight.predict(X)[0]))
    type_pred = str(_model_irr_type.predict(X)[0])
    need_pred = str(_model_irr_need.predict(X)[0])

    # Try to get irrigation type probabilities
    type_probs = {}
    try:
        probs = _model_irr_type.predict_proba(X)[0]
        classes = _model_irr_type.classes_
        type_probs = {str(c): round(float(p), 4) for c, p in zip(classes, probs)}
    except Exception:
        pass

    explanation = (
        f"For {row[config.RAW_CROP_COL]} in {row[config.RAW_SEASON_COL]} season "
        f"on {row[config.RAW_SOIL_COL]} soil: recommended {type_pred} irrigation "
        f"with {need_pred} intensity. Expected sunlight: {sun_pred:.1f} hours/day."
    )

    result = {
        "sunlight_hours": round(sun_pred, 1),
        "irrigation_type": type_pred,
        "irrigation_need": need_pred,
        "irrigation_type_probabilities": type_probs,
        "explanation": explanation,
    }

    logger.info("Prediction: %s", result)
    return result


if __name__ == "__main__":
    sample = {
        "crop": "rice", "ph": 6.5, "temperature": 30.0,
        "humidity": 80.0, "rainfall": 250.0,
        "soil_type": "clay", "season": "kharif",
    }
    print(predict_irrigation_advisory(sample))
