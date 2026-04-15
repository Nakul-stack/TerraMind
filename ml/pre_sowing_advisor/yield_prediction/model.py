"""
Model definition & evaluation for Model 2 — Yield Predictor.

Tries XGBoost first, falls back to RandomForestRegressor.
Computes residual-based confidence bands.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from . import config
from .utils import get_logger, save_artifact

logger = get_logger(__name__)


def build_model() -> Any:
    """Try XGBoost, fallback to RF."""
    try:
        from xgboost import XGBRegressor
        model = XGBRegressor(
            n_estimators=config.XGB_N_ESTIMATORS,
            max_depth=config.XGB_MAX_DEPTH,
            learning_rate=config.XGB_LEARNING_RATE,
            min_child_weight=config.XGB_MIN_CHILD_WEIGHT,
            random_state=config.RANDOM_STATE,
            n_jobs=-1,
            tree_method="hist",
        )
        logger.info("Built XGBRegressor (n_estimators=%d)", config.XGB_N_ESTIMATORS)
        return model, "XGBoost"
    except ImportError:
        logger.warning("XGBoost not available, falling back to RandomForestRegressor")
        model = RandomForestRegressor(
            n_estimators=config.RF_N_ESTIMATORS,
            max_depth=config.RF_MAX_DEPTH,
            min_samples_split=config.RF_MIN_SAMPLES_SPLIT,
            min_samples_leaf=config.RF_MIN_SAMPLES_LEAF,
            random_state=config.RANDOM_STATE,
            n_jobs=-1,
        )
        logger.info("Built RandomForestRegressor (n_estimators=%d)", config.RF_N_ESTIMATORS)
        return model, "RandomForest"


def train_model(model: Any, X_train: np.ndarray, y_train: np.ndarray) -> Any:
    """Fit the model and persist."""
    logger.info("Training started …")
    model.fit(X_train, y_train)
    save_artifact(model, config.MODEL_PATH)
    logger.info("Model saved → %s", config.MODEL_PATH)
    return model


def compute_residual_std(model: Any, X_train: np.ndarray, y_train: np.ndarray) -> float:
    """Compute residual standard deviation for confidence bands."""
    preds = model.predict(X_train)
    residuals = y_train - preds
    std = float(np.std(residuals))
    save_artifact(std, config.RESIDUAL_STD_PATH)
    logger.info("Residual std: %.4f (saved)", std)
    return std


def evaluate_model(
    model: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    n_samples: int = 10,
) -> Dict[str, Any]:
    """Evaluate with MAE, RMSE, R²."""
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    metrics = {
        "train_r2": round(r2_score(y_train, y_train_pred), 4),
        "train_mae": round(float(mean_absolute_error(y_train, y_train_pred)), 4),
        "train_rmse": round(float(np.sqrt(mean_squared_error(y_train, y_train_pred))), 4),
        "test_r2": round(r2_score(y_test, y_test_pred), 4),
        "test_mae": round(float(mean_absolute_error(y_test, y_test_pred)), 4),
        "test_rmse": round(float(np.sqrt(mean_squared_error(y_test, y_test_pred))), 4),
    }

    logger.info("--- Training Metrics ---")
    logger.info("  R²=%.4f  MAE=%.4f  RMSE=%.4f", metrics["train_r2"], metrics["train_mae"], metrics["train_rmse"])
    logger.info("--- Test Metrics ---")
    logger.info("  R²=%.4f  MAE=%.4f  RMSE=%.4f", metrics["test_r2"], metrics["test_mae"], metrics["test_rmse"])

    # Sample predictions
    rng = np.random.RandomState(config.RANDOM_STATE)
    indices = rng.choice(len(y_test), size=min(n_samples, len(y_test)), replace=False)
    logger.info("--- Sample Predictions ---")
    for i in indices:
        logger.info("  actual=%.4f  predicted=%.4f", y_test[i], y_test_pred[i])

    return metrics
