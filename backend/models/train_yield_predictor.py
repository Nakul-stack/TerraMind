"""
TerraMind - Model 2: Yield Predictor Training

Trains a GradientBoostingRegressor on combined production data
with historical features. Provides confidence-band estimation
via residual-based intervals.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.core.config import YIELD_GB_PARAMS, CENTRAL_ARTIFACTS, TEST_SIZE, RANDOM_SEED
from backend.core.logging_config import log
from backend.utils.data_loader import load_combined_yield_data
from backend.utils.feature_engineering import engineer_yield_features
from backend.utils.metrics import regression_metrics, log_metrics


YIELD_FEATURES = [
    "crop_enc", "state_enc", "district_enc", "season_enc",
    "area_log", "yield_lag1", "yield_lag2", "yield_lag3",
    "yield_rolling3_mean", "yield_rolling5_mean",
    "area_lag1", "yield_trend_slope",
]


def train_yield_predictor():
    log.info("============================================")
    log.info("  Training Model 2 - Yield Predictor")
    log.info("============================================")

    # ── 1. Load & engineer features ────────────────────────────────────
    df = load_combined_yield_data()
    df = engineer_yield_features(df)

    # Encode categoricals
    le_crop    = LabelEncoder(); df["crop_enc"]     = le_crop.fit_transform(df["Crop"].astype(str))
    le_state   = LabelEncoder(); df["state_enc"]    = le_state.fit_transform(df["State"].astype(str))
    le_district = LabelEncoder(); df["district_enc"] = le_district.fit_transform(df["District"].astype(str))
    le_season  = LabelEncoder(); df["season_enc"]   = le_season.fit_transform(df["Season"].astype(str))

    # Log-transform area
    df["area_log"] = np.log1p(df["Area"].fillna(0).clip(lower=0))

    # Drop rows where yield_lag1 is NaN (first years)
    lag_cols = ["yield_lag1", "yield_lag2", "yield_lag3",
                "yield_rolling3_mean", "yield_rolling5_mean",
                "area_lag1", "yield_trend_slope"]
    df = df.dropna(subset=["yield_lag1"]).copy()

    # Fill remaining NaNs in lag cols with column median
    for c in lag_cols:
        df[c] = df[c].fillna(df[c].median())

    # Final safety: fill any remaining NaN with 0
    df = df.fillna(0)

    log.info("Training data after feature engineering: %d rows", len(df))

    # -- 2. Prepare X, y ---------------------------------------------------
    feature_cols = [
        "crop_enc", "state_enc", "district_enc", "season_enc",
        "area_log", "yield_lag1", "yield_lag2", "yield_lag3",
        "yield_rolling3_mean", "yield_rolling5_mean",
        "area_lag1", "yield_trend_slope",
    ]
    # De-duplicate column names (area_lag1 appears twice in YIELD_FEATURES)
    feature_cols = list(dict.fromkeys(feature_cols))

    X = df[feature_cols].values.astype(float)
    y = df["Yield"].values.astype(float)

    # ── 3. Split ───────────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED
    )
    log.info("Train=%d  Test=%d", len(X_train), len(X_test))

    # ── 4. Scale ───────────────────────────────────────────────────────
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    # ── 5. Train ───────────────────────────────────────────────────────
    t0 = time.time()
    model = GradientBoostingRegressor(**YIELD_GB_PARAMS)
    model.fit(X_train_s, y_train)
    train_time = time.time() - t0
    log.info("Training took %.1fs", train_time)

    # ── 6. Evaluate ────────────────────────────────────────────────────
    y_pred = model.predict(X_test_s)
    metrics = regression_metrics(y_test, y_pred)
    metrics["train_time_s"] = round(train_time, 2)
    log_metrics("YieldPredictor", metrics)

    # ── 7. Residual-based confidence intervals ─────────────────────────
    residuals = y_test - y_pred
    residual_std = float(np.std(residuals))
    residual_q10 = float(np.percentile(residuals, 10))
    residual_q90 = float(np.percentile(residuals, 90))
    log.info("Residual std=%.4f  Q10=%.4f  Q90=%.4f", residual_std, residual_q10, residual_q90)

    # ── 8. Save artifacts ──────────────────────────────────────────────
    out_dir = CENTRAL_ARTIFACTS / "yield_predictor"
    out_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(model,  out_dir / "model.joblib")
    joblib.dump(scaler, out_dir / "scaler.joblib")
    joblib.dump(le_crop,     out_dir / "le_crop.joblib")
    joblib.dump(le_state,    out_dir / "le_state.joblib")
    joblib.dump(le_district, out_dir / "le_district.joblib")
    joblib.dump(le_season,   out_dir / "le_season.joblib")

    metadata = {
        "features": feature_cols,
        "metrics": metrics,
        "residual_std": residual_std,
        "residual_q10": residual_q10,
        "residual_q90": residual_q90,
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "trained_at": pd.Timestamp.now().isoformat(),
        "version": "1.0.0",
    }
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    log.info("Artifacts saved to %s", out_dir)
    return model, metadata


if __name__ == "__main__":
    train_yield_predictor()
