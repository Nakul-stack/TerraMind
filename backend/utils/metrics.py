"""
TerraMind - Evaluation metrics utilities.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score, classification_report, f1_score,
    mean_absolute_error, mean_squared_error, r2_score,
    precision_score, recall_score, top_k_accuracy_score,
)

from backend.core.logging_config import log


def classification_metrics(y_true, y_pred, y_proba=None, labels=None) -> dict:
    """Compute classification metrics including top-3 accuracy if probabilities given."""
    metrics = {
        "accuracy":        float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall":    float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_f1":        float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }
    if y_proba is not None and labels is not None:
        try:
            metrics["top3_accuracy"] = float(
                top_k_accuracy_score(y_true, y_proba, k=min(3, len(labels)), labels=labels)
            )
        except Exception:
            metrics["top3_accuracy"] = None

    report = classification_report(y_true, y_pred, zero_division=0, output_dict=True)
    metrics["per_class"] = {
        k: v for k, v in report.items()
        if k not in ("accuracy", "macro avg", "weighted avg")
    }
    return metrics


def regression_metrics(y_true, y_pred) -> dict:
    """Compute MAE, RMSE, R² for regression targets."""
    return {
        "mae":  float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2":   float(r2_score(y_true, y_pred)),
    }


def log_metrics(name: str, m: dict):
    """Pretty-print metrics to logger."""
    lines = [f"  {k}: {v}" for k, v in m.items() if k != "per_class"]
    log.info("Metrics for %s:\n%s", name, "\n".join(lines))
