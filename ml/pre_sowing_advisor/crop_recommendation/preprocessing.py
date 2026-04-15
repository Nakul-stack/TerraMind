"""
Data loading, validation, and preprocessing for the Crop Recommender.

Pipeline:
    1. Load CSV.
    2. Validate columns, drop nulls/duplicates.
    3. Merge confusing labels (paddy → rice).
    4. Engineer interaction features (NP_ratio, KP_ratio, rainfall_humidity).
    5. Label-encode target.
    6. Scale features with StandardScaler.
    7. Train / validation / test split.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from . import config
from .utils import get_logger, save_artifact

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# 1. Loading
# ---------------------------------------------------------------------------

def load_dataset(path: Path | None = None) -> pd.DataFrame:
    """Read the crop recommendation CSV into a DataFrame."""
    path = path or config.DATASET_FILE
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    df = pd.read_csv(path)
    logger.info("Loaded dataset: %d rows × %d cols from %s", *df.shape, path.name)
    return df


# ---------------------------------------------------------------------------
# 2. Validation
# ---------------------------------------------------------------------------

def validate(df: pd.DataFrame) -> pd.DataFrame:
    """Validate schema, drop nulls/duplicates, lowercase labels, merge synonyms."""
    required = set(config.FEATURE_COLUMNS + [config.TARGET_COLUMN])
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    initial = len(df)
    df = df.dropna(subset=config.FEATURE_COLUMNS + [config.TARGET_COLUMN])
    df = df.drop_duplicates()
    df[config.TARGET_COLUMN] = df[config.TARGET_COLUMN].str.strip().str.lower()

    # Merge confusing labels (paddy → rice, etc.)
    if config.LABEL_MERGE_MAP:
        before_classes = df[config.TARGET_COLUMN].nunique()
        df[config.TARGET_COLUMN] = df[config.TARGET_COLUMN].replace(config.LABEL_MERGE_MAP)
        after_classes = df[config.TARGET_COLUMN].nunique()
        if before_classes != after_classes:
            logger.info("Label merge: %d → %d classes (merged: %s)",
                        before_classes, after_classes, config.LABEL_MERGE_MAP)

    logger.info("Validation: %d → %d rows, %d classes",
                initial, len(df), df[config.TARGET_COLUMN].nunique())
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# 3. Feature Engineering
# ---------------------------------------------------------------------------

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add interaction features that help separate overlapping crop clusters.

    New features:
        NP_ratio:           N / (P + 1)  — Nitrogen-Phosphorus balance
        KP_ratio:           K / (P + 1)  — Potassium-Phosphorus balance
        rainfall_humidity:  rainfall * humidity / 100  — moisture interaction
    """
    df = df.copy()
    df["NP_ratio"] = df["N"] / (df["P"] + 1)
    df["KP_ratio"] = df["K"] / (df["P"] + 1)
    df["rainfall_humidity"] = df["rainfall"] * df["humidity"] / 100.0
    logger.info("Engineered features added: %s", config.ENGINEERED_FEATURES)
    return df


# ---------------------------------------------------------------------------
# 4. Prepare
# ---------------------------------------------------------------------------

def prepare_data(
    df: pd.DataFrame,
    *,
    fit: bool = True,
) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray,  # X_train, X_val, X_test
    np.ndarray, np.ndarray, np.ndarray,  # y_train, y_val, y_test
    StandardScaler, LabelEncoder,
]:
    """Split, encode, and scale the data.

    When fit=True, creates & saves scaler + encoder.

    Returns:
        (X_train, X_val, X_test, y_train, y_val, y_test, scaler, label_encoder)
    """
    # Use ALL_FEATURES (raw + engineered)
    X = df[config.ALL_FEATURES].values.astype(np.float64)
    y_raw = df[config.TARGET_COLUMN].values

    if fit:
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y_raw)
        save_artifact(label_encoder, config.LABEL_ENCODER_PATH)
        logger.info("Classes (%d): %s", len(label_encoder.classes_), list(label_encoder.classes_))
    else:
        from .utils import load_artifact as _load
        label_encoder = _load(config.LABEL_ENCODER_PATH)
        y = label_encoder.transform(y_raw)

    # Split: train+val / test  → then train / val
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE, stratify=y,
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval,
        test_size=config.VAL_SIZE / (1 - config.TEST_SIZE),
        random_state=config.RANDOM_STATE,
        stratify=y_trainval,
    )

    if fit:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)
        save_artifact(scaler, config.SCALER_PATH)
    else:
        from .utils import load_artifact as _load
        scaler = _load(config.SCALER_PATH)
        X_train = scaler.transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)

    logger.info("Split → train=%d, val=%d, test=%d", len(X_train), len(X_val), len(X_test))
    return X_train, X_val, X_test, y_train, y_val, y_test, scaler, label_encoder
