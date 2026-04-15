"""
Inference module for Model 1 — Crop Recommender.

Returns top-3 crops with confidence scores, plus the selected (top-1) crop.

Usage:
    from ml.pre_sowing_advisor.crop_recommendation.predict import predict_crop
    result = predict_crop({
        "N": 90, "P": 40, "K": 40,
        "temperature": 25, "humidity": 80,
        "rainfall": 200, "ph": 6.5,
    })
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

if __name__ == "__main__":
    _root = Path(__file__).resolve().parents[3]
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

from ml.pre_sowing_advisor.crop_recommendation import config
from ml.pre_sowing_advisor.crop_recommendation.utils import get_logger, load_artifact

logger = get_logger(__name__)

# Lazy-loaded singletons
_model = None
_scaler = None
_label_encoder = None


def _load_artifacts() -> None:
    """Load model, scaler, label-encoder from disk (once)."""
    global _model, _scaler, _label_encoder
    if _model is not None:
        return
    logger.info("Loading crop recommender artifacts …")
    _model = load_artifact(config.MODEL_PATH)
    _scaler = load_artifact(config.SCALER_PATH)
    _label_encoder = load_artifact(config.LABEL_ENCODER_PATH)
    logger.info("Crop recommender artifacts loaded.")


def predict_crop(input_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Predict top-3 crops with confidence scores.

    Args:
        input_dict: Must contain N, P, K, temperature, humidity, ph, rainfall.

    Returns:
        {
            "top_3": [{"crop": str, "confidence": float}, ...],
            "selected_crop": str,
            "selected_confidence": float,
        }
    """
    _load_artifacts()

    missing = [f for f in config.FEATURE_COLUMNS if f not in input_dict]
    if missing:
        raise KeyError(f"Missing required features: {missing}")

    # Build raw feature values
    raw = {col: float(input_dict[col]) for col in config.FEATURE_COLUMNS}

    # Add engineered features
    raw["NP_ratio"] = raw["N"] / (raw["P"] + 1)
    raw["KP_ratio"] = raw["K"] / (raw["P"] + 1)
    raw["rainfall_humidity"] = raw["rainfall"] * raw["humidity"] / 100.0

    X = np.array(
        [[raw[col] for col in config.ALL_FEATURES]],
        dtype=np.float64,
    )
    X_scaled = _scaler.transform(X)

    # Get probabilities
    probs = _model.predict_proba(X_scaled)[0]
    class_indices = np.argsort(probs)[::-1]  # descending

    # Build top-3 list
    top_3: List[Dict[str, Any]] = []
    for idx in class_indices[:3]:
        crop_encoded = _model.classes_[idx]
        crop_name = str(_label_encoder.inverse_transform([crop_encoded])[0])
        confidence = float(probs[idx])
        top_3.append({"crop": crop_name, "confidence": round(confidence, 4)})

    selected_crop = top_3[0]["crop"]
    selected_confidence = top_3[0]["confidence"]

    logger.info(
        "Prediction: top_1=%s (%.4f), top_3=%s",
        selected_crop, selected_confidence,
        [(c["crop"], c["confidence"]) for c in top_3],
    )

    return {
        "top_3": top_3,
        "selected_crop": selected_crop,
        "selected_confidence": selected_confidence,
    }


if __name__ == "__main__":
    sample = {"N": 90, "P": 40, "K": 40, "temperature": 25.0,
              "humidity": 80.0, "rainfall": 200.0, "ph": 6.5}
    print(predict_crop(sample))
