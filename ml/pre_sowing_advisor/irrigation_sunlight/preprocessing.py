"""
Data loading, validation, feature engineering, and preprocessing
for Model 3 — Pre-Sowing Advisor.

Improvements:
    - 4 engineered numeric features for better class separation
    - Proper handling of numeric + categorical feature pipeline
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

from . import config
from .utils import get_logger, normalise_string, save_artifact

logger = get_logger(__name__)


def load_dataset(path: Path | None = None) -> pd.DataFrame:
    path = path or config.DATASET_FILE
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    df = pd.read_csv(path)
    logger.info("Loaded: %d rows × %d cols from %s", *df.shape, path.name)
    return df


def validate_and_clean(df: pd.DataFrame) -> pd.DataFrame:
    """Validate schema, normalise categoricals, standardise target column names."""
    # Find the sunlight column (case-insensitive)
    sun_col = None
    for col in df.columns:
        if col.lower().replace("_", "").replace(" ", "") == "sunlighthours":
            sun_col = col
            break
    if sun_col and sun_col != "sunlight_hours":
        df = df.rename(columns={sun_col: "sunlight_hours"})

    required = set(config.FEATURE_COLUMNS + config.TARGET_COLUMNS)
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    initial = len(df)
    df = df.dropna(subset=config.FEATURE_COLUMNS + config.TARGET_COLUMNS)
    df = df.drop_duplicates()

    for col in config.CATEGORICAL_FEATURES:
        df[col] = df[col].astype(str).apply(normalise_string)
    for col in [config.RAW_IRR_TYPE_COL, config.RAW_IRR_NEED_COL]:
        df[col] = df[col].astype(str).apply(normalise_string)

    logger.info("Cleaning: %d → %d rows", initial, len(df))
    return df.reset_index(drop=True)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived numeric features for better classifier performance.

    New features:
        temp_rainfall_ratio:       temperature / (rainfall + 1)
        humidity_rainfall:         humidity * rainfall / 100
        rainfall_log:              log1p(rainfall) — tames large rainfall skew
        temp_humidity_interaction:  temperature * humidity / 100
    """
    df = df.copy()
    df["temp_rainfall_ratio"] = df["temperature"] / (df["rainfall"] + 1)
    df["humidity_rainfall"] = df["humidity"] * df["rainfall"] / 100.0
    df["rainfall_log"] = np.log1p(df["rainfall"])
    df["temp_humidity_interaction"] = df["temperature"] * df["humidity"] / 100.0
    logger.info("Engineered %d features: %s",
                len(config.ENGINEERED_NUMERIC_FEATURES),
                config.ENGINEERED_NUMERIC_FEATURES)
    return df


def build_preprocessor() -> ColumnTransformer:
    """Build preprocessor: StandardScaler for ALL numerics (raw + engineered),
    OrdinalEncoder for categoricals."""
    all_numeric = config.NUMERIC_FEATURES + config.ENGINEERED_NUMERIC_FEATURES
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), all_numeric),
            ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
             config.CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )


def prepare_data(
    df: pd.DataFrame,
    *,
    fit: bool = True,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame, ColumnTransformer]:
    """Split and transform. Returns (X_train, X_test, y_train_df, y_test_df, preprocessor)."""
    all_feature_cols = (config.NUMERIC_FEATURES +
                        config.ENGINEERED_NUMERIC_FEATURES +
                        config.CATEGORICAL_FEATURES)
    X = df[all_feature_cols]
    y = df[config.TARGET_COLUMNS]

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE,
    )
    logger.info("Split → train=%d, test=%d", len(X_train_raw), len(X_test_raw))

    if fit:
        preprocessor = build_preprocessor()
        X_train = preprocessor.fit_transform(X_train_raw)
        X_test = preprocessor.transform(X_test_raw)
        save_artifact(preprocessor, config.PREPROCESSOR_PATH)
        logger.info("Preprocessor fitted and saved.")
    else:
        from .utils import load_artifact
        preprocessor = load_artifact(config.PREPROCESSOR_PATH)
        X_train = preprocessor.transform(X_train_raw)
        X_test = preprocessor.transform(X_test_raw)

    return X_train, X_test, y_train, y_test, preprocessor
