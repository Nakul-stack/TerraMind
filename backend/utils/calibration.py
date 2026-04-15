"""
TerraMind - Probability calibration utilities.
"""
from __future__ import annotations

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.base import BaseEstimator

from backend.core.logging_config import log


def calibrate_classifier(
    model: BaseEstimator,
    X_val,
    y_val,
    method: str = "isotonic",
    cv: int = 3,
) -> CalibratedClassifierCV:
    """
    Return a calibrated version of the classifier using held-out data.
    Falls back to the original model if calibration fails.
    """
    try:
        calibrated = CalibratedClassifierCV(model, method=method, cv="prefit")
        calibrated.fit(X_val, y_val)
        log.info("Probability calibration applied (method=%s)", method)
        return calibrated
    except Exception as exc:
        log.warning("Calibration failed: %s - returning original model.", exc)
        return model


def softmax_temperature(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Apply temperature scaling to logits/probabilities."""
    scaled = logits / max(temperature, 1e-8)
    exp_scaled = np.exp(scaled - np.max(scaled))
    return exp_scaled / exp_scaled.sum()
