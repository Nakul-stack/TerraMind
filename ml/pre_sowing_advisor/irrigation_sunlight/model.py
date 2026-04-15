"""
Model definition & evaluation for Model 3 — Pre-Sowing Advisor.

Architecture:
    1. sunlight_hours → GradientBoostingRegressor (with subsample)
    2. irrigation_type → Best of {GradientBoosting, ExtraTrees(balanced)}
    3. irrigation_need → Best of {GradientBoosting, ExtraTrees(balanced)}

Accuracy & recall improvements:
    - GBClassifier with lower learning rate + subsample for generalization
    - ExtraTreesClassifier with class_weight='balanced' for recall
    - Model comparison on validation set — pick best per target
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
)
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

from . import config
from .utils import get_logger, save_artifact

logger = get_logger(__name__)

_MODEL_PATHS = {
    "sunlight_hours": config.MODEL_SUNLIGHT_PATH,
    "irrigation_type": config.MODEL_IRR_TYPE_PATH,
    "irrigation_need": config.MODEL_IRR_NEED_PATH,
}


def _build_sunlight_model() -> GradientBoostingRegressor:
    return GradientBoostingRegressor(
        n_estimators=config.GBR_SUN_N_ESTIMATORS,
        max_depth=config.GBR_SUN_MAX_DEPTH,
        learning_rate=config.GBR_SUN_LEARNING_RATE,
        min_samples_leaf=config.GBR_SUN_MIN_SAMPLES_LEAF,
        subsample=config.GBR_SUN_SUBSAMPLE,
        random_state=config.RANDOM_STATE,
    )


def _build_irr_type_candidates() -> Dict[str, Any]:
    """Build multiple diverse candidates for irrigation type — pick best.

    This target has very low feature-target correlation, so we throw a wider
    net of diverse classifiers to find whatever signal exists.
    """
    from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier

    return {
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=config.GB_CLF_TYPE_N_ESTIMATORS,
            max_depth=config.GB_CLF_TYPE_MAX_DEPTH,
            learning_rate=config.GB_CLF_TYPE_LEARNING_RATE,
            min_samples_leaf=config.GB_CLF_TYPE_MIN_SAMPLES_LEAF,
            subsample=config.GB_CLF_TYPE_SUBSAMPLE,
            random_state=config.RANDOM_STATE,
        ),
        "ExtraTrees_balanced": ExtraTreesClassifier(
            n_estimators=config.ET_CLF_N_ESTIMATORS,
            max_depth=config.ET_CLF_MAX_DEPTH,
            class_weight="balanced",
            random_state=config.RANDOM_STATE,
            n_jobs=-1,
        ),
        "RandomForest_balanced": RandomForestClassifier(
            n_estimators=800,
            max_depth=None,
            class_weight="balanced_subsample",
            min_samples_leaf=2,
            random_state=config.RANDOM_STATE,
            n_jobs=-1,
        ),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            max_iter=500,
            max_depth=8,
            learning_rate=0.05,
            min_samples_leaf=10,
            l2_regularization=0.1,
            random_state=config.RANDOM_STATE,
        ),
    }


def _build_irr_need_candidates() -> Dict[str, Any]:
    """Build multiple candidates for irrigation need — pick best."""
    from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier

    return {
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=config.GB_CLF_NEED_N_ESTIMATORS,
            max_depth=config.GB_CLF_NEED_MAX_DEPTH,
            learning_rate=config.GB_CLF_NEED_LEARNING_RATE,
            min_samples_leaf=config.GB_CLF_NEED_MIN_SAMPLES_LEAF,
            subsample=config.GB_CLF_NEED_SUBSAMPLE,
            random_state=config.RANDOM_STATE,
        ),
        "ExtraTrees_balanced": ExtraTreesClassifier(
            n_estimators=config.ET_CLF_N_ESTIMATORS,
            max_depth=config.ET_CLF_MAX_DEPTH,
            class_weight="balanced",
            random_state=config.RANDOM_STATE,
            n_jobs=-1,
        ),
        "RandomForest_balanced": RandomForestClassifier(
            n_estimators=800,
            max_depth=None,
            class_weight="balanced_subsample",
            min_samples_leaf=2,
            random_state=config.RANDOM_STATE,
            n_jobs=-1,
        ),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            max_iter=500,
            max_depth=8,
            learning_rate=0.05,
            min_samples_leaf=10,
            l2_regularization=0.1,
            random_state=config.RANDOM_STATE,
        ),
    }


def _select_best_classifier(
    candidates: Dict[str, Any],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    target_name: str,
) -> tuple[Any, str]:
    """Train all candidates on split data, select best by macro F1."""
    best_model = None
    best_name = ""
    best_f1 = -1.0

    for name, model in candidates.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        acc = accuracy_score(y_val, y_pred)
        f1 = float(f1_score(y_val, y_pred, average="macro", zero_division=0))
        recall = float(recall_score(y_val, y_pred, average="macro", zero_division=0))
        logger.info("  %s → %s: acc=%.4f, macro_f1=%.4f, macro_recall=%.4f",
                    target_name, name, acc, f1, recall)
        if f1 > best_f1:
            best_f1 = f1
            best_model = model
            best_name = name

    logger.info("  ✓ Best for %s: %s (macro_f1=%.4f)", target_name, best_name, best_f1)
    return best_model, best_name


def train_models(
    X_train: np.ndarray,
    y_train: pd.DataFrame,
) -> Dict[str, Any]:
    """Train all 3 sub-models with internal validation for classifier selection."""

    # Internal train/val split for classifier comparison
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.15, random_state=config.RANDOM_STATE,
    )

    models = {}
    model_types = {}

    # 1. Sunlight (Regression — no comparison needed)
    sun_model = _build_sunlight_model()
    logger.info("Training sunlight_hours (GBR)...")
    sun_model.fit(X_train, y_train["sunlight_hours"].values.astype(float))
    save_artifact(sun_model, config.MODEL_SUNLIGHT_PATH)
    models["sunlight_hours"] = sun_model
    model_types["sunlight_hours"] = "GradientBoostingRegressor"

    # 2. Irrigation Type — candidate comparison
    logger.info("Training irrigation_type candidates...")
    type_candidates = _build_irr_type_candidates()
    type_model, type_name = _select_best_classifier(
        type_candidates,
        X_tr, y_tr[config.RAW_IRR_TYPE_COL].values,
        X_val, y_val[config.RAW_IRR_TYPE_COL].values,
        "irrigation_type",
    )
    # Retrain winner on full train data
    type_model.fit(X_train, y_train[config.RAW_IRR_TYPE_COL].values)
    save_artifact(type_model, config.MODEL_IRR_TYPE_PATH)
    models[config.RAW_IRR_TYPE_COL] = type_model
    model_types["irrigation_type"] = type_name

    # 3. Irrigation Need — candidate comparison
    logger.info("Training irrigation_need candidates...")
    need_candidates = _build_irr_need_candidates()
    need_model, need_name = _select_best_classifier(
        need_candidates,
        X_tr, y_tr[config.RAW_IRR_NEED_COL].values,
        X_val, y_val[config.RAW_IRR_NEED_COL].values,
        "irrigation_need",
    )
    # Retrain winner on full train data
    need_model.fit(X_train, y_train[config.RAW_IRR_NEED_COL].values)
    save_artifact(need_model, config.MODEL_IRR_NEED_PATH)
    models[config.RAW_IRR_NEED_COL] = need_model
    model_types["irrigation_need"] = need_name

    logger.info("All 3 models trained and saved. Types: %s", model_types)
    return models, model_types


def evaluate_models(
    models: Dict[str, Any],
    X_train: np.ndarray,
    y_train: pd.DataFrame,
    X_test: np.ndarray,
    y_test: pd.DataFrame,
) -> Dict[str, Any]:
    """Evaluate all 3 models with full metrics including per-class recall."""
    metrics: Dict[str, Any] = {}

    # 1. Sunlight (Regression)
    sun_model = models["sunlight_hours"]
    y_te_sun = y_test["sunlight_hours"].values.astype(float)
    y_te_sun_pred = sun_model.predict(X_test)

    metrics["sunlight_test_r2"] = round(r2_score(y_te_sun, y_te_sun_pred), 4)
    metrics["sunlight_test_mae"] = round(float(mean_absolute_error(y_te_sun, y_te_sun_pred)), 4)
    metrics["sunlight_test_rmse"] = round(float(np.sqrt(mean_squared_error(y_te_sun, y_te_sun_pred))), 4)

    logger.info("Sunlight: R²=%.4f MAE=%.4f RMSE=%.4f",
                metrics["sunlight_test_r2"], metrics["sunlight_test_mae"], metrics["sunlight_test_rmse"])

    # 2. Irrigation Type (Classification)
    type_model = models[config.RAW_IRR_TYPE_COL]
    y_te_type = y_test[config.RAW_IRR_TYPE_COL].values
    y_te_type_pred = type_model.predict(X_test)

    type_acc = accuracy_score(y_te_type, y_te_type_pred)
    type_f1 = float(f1_score(y_te_type, y_te_type_pred, average="weighted", zero_division=0))
    type_recall = float(recall_score(y_te_type, y_te_type_pred, average="macro", zero_division=0))

    metrics["irr_type_test_acc"] = round(type_acc, 4)
    metrics["irr_type_test_f1"] = round(type_f1, 4)
    metrics["irr_type_test_macro_recall"] = round(type_recall, 4)

    logger.info("Irrigation Type: Acc=%.4f F1=%.4f Macro_Recall=%.4f",
                type_acc, type_f1, type_recall)
    report_type = classification_report(y_te_type, y_te_type_pred, zero_division=0)
    logger.info("\n%s", report_type)

    # 3. Irrigation Need (Classification)
    need_model = models[config.RAW_IRR_NEED_COL]
    y_te_need = y_test[config.RAW_IRR_NEED_COL].values
    y_te_need_pred = need_model.predict(X_test)

    need_acc = accuracy_score(y_te_need, y_te_need_pred)
    need_f1 = float(f1_score(y_te_need, y_te_need_pred, average="weighted", zero_division=0))
    need_recall = float(recall_score(y_te_need, y_te_need_pred, average="macro", zero_division=0))

    metrics["irr_need_test_acc"] = round(need_acc, 4)
    metrics["irr_need_test_f1"] = round(need_f1, 4)
    metrics["irr_need_test_macro_recall"] = round(need_recall, 4)

    logger.info("Irrigation Need: Acc=%.4f F1=%.4f Macro_Recall=%.4f",
                need_acc, need_f1, need_recall)
    report_need = classification_report(y_te_need, y_te_need_pred, zero_division=0)
    logger.info("\n%s", report_need)

    return metrics
