"""
Training entry-point for Model 3 — Pre-Sowing Advisor.

Usage:
    python -m ml.pre_sowing_advisor.irrigation_sunlight.train
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

if __name__ == "__main__":
    _root = Path(__file__).resolve().parents[3]
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

from ml.pre_sowing_advisor.irrigation_sunlight import config
from ml.pre_sowing_advisor.irrigation_sunlight.utils import get_logger, save_json
from ml.pre_sowing_advisor.irrigation_sunlight.preprocessing import (
    load_dataset,
    validate_and_clean,
    engineer_features,
    prepare_data,
)
from ml.pre_sowing_advisor.irrigation_sunlight.model import (
    train_models,
    evaluate_models,
)

logger = get_logger(__name__)


def main() -> None:
    logger.info("=" * 60)
    logger.info("  Model 3 — Pre-Sowing Advisor Training")
    logger.info("=" * 60)

    df = load_dataset()
    df = validate_and_clean(df)
    df = engineer_features(df)
    X_train, X_test, y_train, y_test, preprocessor = prepare_data(df, fit=True)

    models, model_types = train_models(X_train, y_train)
    metrics = evaluate_models(models, X_train, y_train, X_test, y_test)

    metadata = {
        "model_name": "pre_sowing_advisor",
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "dataset": config.DATASET_FILE.name,
        "features": config.FEATURE_COLUMNS,
        "engineered_features": config.ENGINEERED_NUMERIC_FEATURES,
        "targets": config.TARGET_COLUMNS,
        "model_types": model_types,
        "metrics": metrics,
    }
    save_json(metadata, config.METADATA_PATH)

    logger.info("=" * 60)
    logger.info("  Training complete.")
    logger.info("  Irr Type Acc: %.4f | Irr Need Acc: %.4f",
                metrics.get("irr_type_test_acc", 0),
                metrics.get("irr_need_test_acc", 0))
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
