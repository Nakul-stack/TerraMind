"""
Training entry-point for Model 2 — Yield Predictor.

Usage:
    python -m ml.pre_sowing_advisor.yield_prediction.train
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

if __name__ == "__main__":
    _root = Path(__file__).resolve().parents[3]
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

from ml.pre_sowing_advisor.yield_prediction import config
from ml.pre_sowing_advisor.yield_prediction.utils import get_logger, save_json
from ml.pre_sowing_advisor.yield_prediction.preprocessing import (
    load_and_merge_datasets,
    validate_and_clean,
    engineer_target,
    prepare_data,
)
from ml.pre_sowing_advisor.yield_prediction.model import (
    build_model,
    train_model,
    compute_residual_std,
    evaluate_model,
)

logger = get_logger(__name__)


def main() -> None:
    logger.info("=" * 60)
    logger.info("  Model 2 — Yield Predictor Training")
    logger.info("=" * 60)

    # 1. Load & merge
    df = load_and_merge_datasets()

    # 2. Clean
    df = validate_and_clean(df)

    # 3. Engineer target
    df = engineer_target(df)

    # 4. Preprocess & split
    X_train, X_test, y_train, y_test, preprocessor = prepare_data(df, fit=True)

    # 5. Build & train
    model, model_type = build_model()
    model = train_model(model, X_train, y_train)

    # 6. Compute residual std for confidence bands
    residual_std = compute_residual_std(model, X_train, y_train)

    # 7. Evaluate
    metrics = evaluate_model(model, X_train, y_train, X_test, y_test)

    # 8. Save metadata
    metadata = {
        "model_name": "yield_predictor",
        "model_type": model_type,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "datasets": [config.PRIMARY_DATASET.name, config.SECONDARY_DATASET.name],
        "features": config.FEATURE_COLUMNS,
        "target": config.TARGET_COL,
        "residual_std": residual_std,
        "confidence_multiplier": config.CONFIDENCE_MULTIPLIER,
        "metrics": metrics,
    }
    save_json(metadata, config.METADATA_PATH)

    logger.info("=" * 60)
    logger.info("  Training complete — Test R²: %.4f", metrics["test_r2"])
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
