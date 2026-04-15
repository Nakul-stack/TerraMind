"""
Data loading, validation, target engineering, and preprocessing for
Model 2 — Yield Predictor.

Merges primary (crop_production.csv.xlsx) and secondary (India Agriculture
Crop Production.csv) datasets, engineers yield = Production / Area,
removes outliers, and builds an OrdinalEncoder pipeline.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder

from . import config
from .utils import get_logger, normalise_string, save_artifact

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# 1. Loading & Merging
# ---------------------------------------------------------------------------

def load_and_merge_datasets() -> pd.DataFrame:
    """Load primary + secondary datasets and merge into unified schema."""

    frames = []

    # --- Primary dataset ---
    if config.PRIMARY_DATASET.exists():
        logger.info("Loading primary dataset: %s", config.PRIMARY_DATASET.name)
        try:
            df1 = pd.read_excel(config.PRIMARY_DATASET)
        except Exception:
            df1 = pd.read_csv(config.PRIMARY_DATASET)

        df1 = df1.rename(columns={
            config.RAW_STATE_COL: "state",
            config.RAW_DISTRICT_COL: "district",
            config.RAW_CROP_COL: "crop",
            config.RAW_SEASON_COL: "season",
            config.RAW_AREA_COL: "area",
            config.RAW_PRODUCTION_COL: "production",
            config.RAW_YEAR_COL: "year",
        })
        frames.append(df1)
        logger.info("  Primary: %d rows", len(df1))
    else:
        logger.warning("Primary dataset not found: %s", config.PRIMARY_DATASET)

    # --- Secondary dataset ---
    if config.SECONDARY_DATASET.exists():
        logger.info("Loading secondary dataset: %s", config.SECONDARY_DATASET.name)
        df2 = pd.read_csv(config.SECONDARY_DATASET)
        rename_map = {}
        for orig, target in [
            (config.SEC_STATE_COL, "state"),
            (config.SEC_DISTRICT_COL, "district"),
            (config.SEC_CROP_COL, "crop"),
            (config.SEC_SEASON_COL, "season"),
            (config.SEC_AREA_COL, "area"),
            (config.SEC_PRODUCTION_COL, "production"),
            (config.SEC_YEAR_COL, "year"),
            (config.SEC_YIELD_COL, "yield_raw"),
        ]:
            if orig in df2.columns:
                rename_map[orig] = target
        df2 = df2.rename(columns=rename_map)
        frames.append(df2)
        logger.info("  Secondary: %d rows", len(df2))
    else:
        logger.warning("Secondary dataset not found: %s", config.SECONDARY_DATASET)

    if not frames:
        raise FileNotFoundError("No yield datasets found!")

    # Unify
    common_cols = ["state", "district", "crop", "season", "area", "production"]
    for df in frames:
        for col in common_cols:
            if col not in df.columns:
                df[col] = np.nan

    df = pd.concat([f[common_cols] for f in frames], ignore_index=True)
    logger.info("Merged dataset: %d rows", len(df))
    return df


# ---------------------------------------------------------------------------
# 2. Validation & Cleaning
# ---------------------------------------------------------------------------

def validate_and_clean(df: pd.DataFrame) -> pd.DataFrame:
    """Validate, normalise strings, drop invalid rows."""
    initial = len(df)

    # Normalise text columns
    for col in ["state", "district", "crop", "season"]:
        df[col] = df[col].astype(str).apply(normalise_string)

    # Convert numerics
    df["area"] = pd.to_numeric(df["area"], errors="coerce")
    df["production"] = pd.to_numeric(df["production"], errors="coerce")

    # Drop rows with invalid area/production
    df = df.dropna(subset=["area", "production"])
    df = df[df["area"] > 0]
    df = df.drop_duplicates()

    logger.info("Cleaning: %d → %d rows (%d removed)", initial, len(df), initial - len(df))
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# 3. Target Engineering
# ---------------------------------------------------------------------------

def engineer_target(df: pd.DataFrame) -> pd.DataFrame:
    """Create yield = production / area, remove outliers via IQR."""
    df = df.copy()
    df[config.TARGET_COL] = df["production"] / df["area"]

    # Remove infinite / NaN
    before = len(df)
    df = df[np.isfinite(df[config.TARGET_COL])]
    df = df[df[config.TARGET_COL] >= 0]
    if len(df) < before:
        logger.warning("Removed %d non-finite yield rows", before - len(df))

    # IQR outlier removal
    q1 = df[config.TARGET_COL].quantile(0.25)
    q3 = df[config.TARGET_COL].quantile(0.75)
    iqr = q3 - q1
    upper = q3 + 3.0 * iqr
    before_iqr = len(df)
    df = df[df[config.TARGET_COL] <= upper]
    logger.info(
        "Outlier removal: Q1=%.2f Q3=%.2f upper=%.2f removed=%d",
        q1, q3, upper, before_iqr - len(df),
    )
    logger.info(
        "Yield stats: min=%.4f max=%.4f median=%.4f",
        df[config.TARGET_COL].min(),
        df[config.TARGET_COL].max(),
        df[config.TARGET_COL].median(),
    )
    return df


# ---------------------------------------------------------------------------
# 4. Preprocessing & Split
# ---------------------------------------------------------------------------

def build_preprocessor() -> ColumnTransformer:
    """OrdinalEncoder for categorical features."""
    return ColumnTransformer(
        transformers=[(
            "cat",
            OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
            config.FEATURE_COLUMNS,
        )],
        remainder="drop",
    )


def prepare_data(
    df: pd.DataFrame,
    *,
    fit: bool = True,
    preprocessor: ColumnTransformer | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, ColumnTransformer]:
    """Split and transform. Returns (X_train, X_test, y_train, y_test, preprocessor)."""

    X = df[config.FEATURE_COLUMNS]
    y = df[config.TARGET_COL].values

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
        if preprocessor is None:
            raise ValueError("preprocessor required when fit=False")
        X_train = preprocessor.transform(X_train_raw)
        X_test = preprocessor.transform(X_test_raw)

    return X_train, X_test, y_train, y_test, preprocessor
