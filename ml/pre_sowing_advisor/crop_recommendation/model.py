"""
Model definition & evaluation for the Crop Recommender.

Trains BOTH RandomForest and GradientBoosting, compares on validation set,
and keeps the best performer.  Evaluates accuracy + top-3 accuracy.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

from . import config
from .utils import get_logger, save_artifact

logger = get_logger(__name__)


def build_rf_model() -> RandomForestClassifier:
    """Build RandomForestClassifier."""
    return RandomForestClassifier(
        n_estimators=config.RF_N_ESTIMATORS,
        max_depth=config.RF_MAX_DEPTH,
        min_samples_split=config.RF_MIN_SAMPLES_SPLIT,
        min_samples_leaf=config.RF_MIN_SAMPLES_LEAF,
        random_state=config.RANDOM_STATE,
        n_jobs=-1,
    )


def build_gb_model() -> GradientBoostingClassifier:
    """Build GradientBoostingClassifier."""
    return GradientBoostingClassifier(
        n_estimators=config.GB_N_ESTIMATORS,
        max_depth=config.GB_MAX_DEPTH,
        learning_rate=config.GB_LEARNING_RATE,
        min_samples_leaf=config.GB_MIN_SAMPLES_LEAF,
        subsample=config.GB_SUBSAMPLE,
        random_state=config.RANDOM_STATE,
    )


def _top_k_accuracy(model: Any, X: np.ndarray, y: np.ndarray, k: int = 3) -> float:
    """Compute top-k accuracy from predict_proba."""
    probs = model.predict_proba(X)
    top_k_preds = np.argsort(probs, axis=1)[:, -k:]
    correct = sum(1 for i, yi in enumerate(y) if yi in top_k_preds[i])
    return correct / len(y)


def train_and_select_best(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> tuple[Any, str, Dict[str, Any]]:
    """Train both RF and GB, select best on validation accuracy.

    Returns:
        (best_model, model_name, metrics_dict)
    """
    candidates = {
        "RandomForest": build_rf_model(),
        "GradientBoosting": build_gb_model(),
    }

    best_model = None
    best_name = ""
    best_val_acc = -1.0
    all_metrics: Dict[str, Any] = {}

    for name, model in candidates.items():
        logger.info("Training %s ...", name)
        model.fit(X_train, y_train)

        train_acc = accuracy_score(y_train, model.predict(X_train))
        val_acc = accuracy_score(y_val, model.predict(X_val))
        val_top3 = _top_k_accuracy(model, X_val, y_val, k=3)

        all_metrics[name] = {
            "train_accuracy": round(train_acc, 4),
            "val_accuracy": round(val_acc, 4),
            "val_top3_accuracy": round(val_top3, 4),
        }
        logger.info(
            "  %s → train_acc=%.4f, val_acc=%.4f, val_top3=%.4f",
            name, train_acc, val_acc, val_top3,
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model = model
            best_name = name

    logger.info("✓ Best model: %s (val_acc=%.4f)", best_name, best_val_acc)
    return best_model, best_name, all_metrics


def evaluate_model(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    label_names: list[str] | None = None,
) -> Dict[str, Any]:
    """Full evaluation on test set with per-class recall tracking."""
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    top3 = _top_k_accuracy(model, X_test, y_test, k=3)
    report = classification_report(y_test, y_pred, target_names=label_names, output_dict=True)
    report_str = classification_report(y_test, y_pred, target_names=label_names)

    logger.info("Test Accuracy: %.4f", acc)
    logger.info("Test Top-3 Accuracy: %.4f", top3)
    logger.info("\n%s", report_str)

    # Log per-class recall
    if label_names:
        min_recall = 1.0
        min_recall_class = ""
        for cls in label_names:
            if cls in report and isinstance(report[cls], dict):
                recall = report[cls].get('recall', 0)
                if recall < min_recall:
                    min_recall = recall
                    min_recall_class = cls
        logger.info("Min recall: %.4f (class: %s)", min_recall, min_recall_class)

    importances = model.feature_importances_
    feat_names = config.ALL_FEATURES
    feat_imp = {
        feat: round(float(imp), 4)
        for feat, imp in sorted(
            zip(feat_names, importances),
            key=lambda x: x[1], reverse=True,
        )
    }
    logger.info("Feature importances: %s", feat_imp)

    return {
        "test_accuracy": round(acc, 4),
        "test_top3_accuracy": round(top3, 4),
        "classification_report": report,
        "feature_importances": feat_imp,
    }
