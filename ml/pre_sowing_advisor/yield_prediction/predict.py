"""
Inference module for Model 2 — Yield Predictor.

Returns expected yield + confidence band (lower, upper).

Usage:
    from ml.pre_sowing_advisor.yield_prediction.predict import predict_yield
    result = predict_yield({
        "crop": "rice", "state": "karnataka",
        "district": "mysore", "season": "kharif",
    })
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

from ml.pre_sowing_advisor.yield_prediction import config
from ml.pre_sowing_advisor.yield_prediction.utils import get_logger, load_artifact, normalise_string
from ml.pre_sowing_advisor.normalizers import normalize_state_name, normalize_district_name, normalize_crop_name, normalize_season

logger = get_logger(__name__)

# Lazy-loaded singletons
_model = None
_preprocessor = None
_residual_std = None

# API key → internal column name mapping
_INPUT_KEY_MAP: dict[str, str] = {
    "crop": "crop",
    "state": "state",
    "district": "district",
    "season": "season",
}
# Also accept old keys
_KEY_ALIASES: dict[str, str] = {
    "state_name": "state",
    "district_name": "district",
}


def _load_artifacts() -> None:
    global _model, _preprocessor, _residual_std
    if _model is not None:
        return
    logger.info("Loading yield predictor artifacts …")
    _model = load_artifact(config.MODEL_PATH)
    _preprocessor = load_artifact(config.PREPROCESSOR_PATH)
    try:
        _residual_std = load_artifact(config.RESIDUAL_STD_PATH)
    except FileNotFoundError:
        _residual_std = 0.5  # sensible default
        logger.warning("Residual std not found, using default=0.5")
    logger.info("Yield predictor artifacts loaded.")


def predict_yield(input_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Predict yield and confidence band.

    Args:
        input_dict: Must contain crop, state (or state_name),
                    district (or district_name), season.

    Returns:
        {
            "expected_yield": float,
            "unit": "t/ha",
            "confidence_band": {"lower": float, "upper": float},
            "explanation": str,
        }
    """
    _load_artifacts()

    # Normalise key names
    resolved: Dict[str, str] = {}
    for key, val in input_dict.items():
        mapped = _KEY_ALIASES.get(key, key)
        if mapped in _INPUT_KEY_MAP.values():
            resolved[mapped] = str(val)

    missing = [k for k in _INPUT_KEY_MAP.values() if k not in resolved]
    if missing:
        raise KeyError(f"Missing required input keys: {missing}")

    # Normalise values
    row = {
        "crop": normalize_crop_name(resolved["crop"]),
        "state": normalize_state_name(resolved["state"]),
        "district": normalize_district_name(resolved["district"]),
        "season": normalize_season(resolved["season"]),
    }
    df = pd.DataFrame([row])

    # Transform & predict
    X = _preprocessor.transform(df)
    prediction = float(max(0.0, _model.predict(X)[0]))

    # Confidence band
    multiplier = config.CONFIDENCE_MULTIPLIER
    lower = max(0.0, prediction - _residual_std * multiplier)
    upper = prediction + _residual_std * multiplier

    explanation = (
        f"Predicted yield of {prediction:.2f} t/ha for {row['crop']} "
        f"in {row['district']}, {row['state']} ({row['season']}). "
        f"Confidence band [{lower:.2f}, {upper:.2f}] based on "
        f"training residual distribution (±{_residual_std * multiplier:.2f})."
    )

    logger.info("Yield prediction: %.4f [%.4f, %.4f]", prediction, lower, upper)

    return {
        "expected_yield": round(prediction, 4),
        "unit": "t/ha",
        "confidence_band": {
            "lower": round(lower, 4),
            "upper": round(upper, 4),
        },
        "explanation": explanation,
    }


if __name__ == "__main__":
    sample = {"crop": "rice", "state": "Karnataka", "district": "MYSORE", "season": "Kharif"}
    print(predict_yield(sample))
