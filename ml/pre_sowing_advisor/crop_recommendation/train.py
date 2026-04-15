"""
Training entry-point for Model 1 — Crop Recommender.

Usage:
    python -m ml.pre_sowing_advisor.crop_recommendation.train

Pipeline:
    1. Load CSV dataset.
    2. Validate & clean.
    3. Encode, split (train/val/test), scale.
    4. Train RF + GB, keep best on validation.
    5. Evaluate on test set.
    6. Persist model + scaler + encoder + metadata.
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

if __name__ == "__main__":
    _root = Path(__file__).resolve().parents[3]
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

from ml.pre_sowing_advisor.crop_recommendation import config
from ml.pre_sowing_advisor.crop_recommendation.utils import get_logger, save_artifact, save_json
from ml.pre_sowing_advisor.crop_recommendation.preprocessing import (
    load_dataset,
    validate,
    engineer_features,
    prepare_data,
)
from ml.pre_sowing_advisor.crop_recommendation.model import (
    train_and_select_best,
    evaluate_model,
)

logger = get_logger(__name__)


def main() -> None:
    """Execute the full training pipeline."""
    logger.info("=" * 60)
    logger.info("  Model 1 — Crop Recommender Training")
    logger.info("=" * 60)

    # 1. Load
    df = load_dataset()

    # 2. Validate + merge labels
    df = validate(df)

    # 3. Engineer interaction features
    df = engineer_features(df)

    # 4. Preprocess
    X_train, X_val, X_test, y_train, y_val, y_test, scaler, label_encoder = prepare_data(
        df, fit=True,
    )

    # 4. Train & select best
    best_model, model_name, comparison_metrics = train_and_select_best(
        X_train, y_train, X_val, y_val,
    )

    # 5. Save best model
    save_artifact(best_model, config.MODEL_PATH)
    logger.info("Saved best model (%s) → %s", model_name, config.MODEL_PATH)

    # 6. Evaluate on test
    test_metrics = evaluate_model(
        best_model, X_test, y_test,
        label_names=list(label_encoder.classes_),
    )

    # 7. Save metadata
    metadata = {
        "model_name": "crop_recommender",
        "model_type": model_name,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "dataset": config.DATASET_FILE.name,
        "features": config.FEATURE_COLUMNS,
        "target": config.TARGET_COLUMN,
        "classes": list(label_encoder.classes_),
        "n_classes": len(label_encoder.classes_),
        "comparison_metrics": comparison_metrics,
        "test_metrics": test_metrics,
    }
    save_json(metadata, config.METADATA_PATH)

    logger.info("=" * 60)
    logger.info("  Training complete — Test Accuracy: %.4f | Top-3: %.4f",
                test_metrics["test_accuracy"], test_metrics["test_top3_accuracy"])
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
