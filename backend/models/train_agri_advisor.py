"""
TerraMind - Model 3: Agri-Condition Advisor Training

Trains three separate models on irrigation_prediction.csv:
  1. sunlight_hours     -> Regression   (RandomForestRegressor)
  2. irrigation_type    -> Classification (GradientBoostingClassifier)
  3. irrigation_need    -> Classification (GradientBoostingClassifier)
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingClassifier,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, OrdinalEncoder

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.core.config import (
    AGRI_RF_PARAMS, CENTRAL_ARTIFACTS, TEST_SIZE, VAL_SIZE, RANDOM_SEED,
    AGRI_INPUT_FEATURES, AGRI_REG_TARGET, AGRI_CLF_TARGETS,
)
from backend.core.logging_config import log
from backend.utils.data_loader import load_irrigation_dataset
from backend.utils.metrics import classification_metrics, regression_metrics, log_metrics


def train_agri_advisor():
    log.info("============================================")
    log.info("  Training Model 3 - Agri-Condition Advisor")
    log.info("============================================")

    # ── 1. Load data ────────────────────────────────────────────────────
    df = load_irrigation_dataset()
    log.info("Dataset shape: %s", df.shape)

    # ── 2. Encode categorical inputs ───────────────────────────────────
    cat_cols = ["crop", "soil_type", "season"]
    encoders: dict[str, LabelEncoder] = {}
    for col in cat_cols:
        le = LabelEncoder()
        df[col + "_enc"] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    num_cols = ["ph", "temperature", "humidity", "rainfall"]
    enc_cols = [c + "_enc" for c in cat_cols]
    feature_cols = num_cols + enc_cols

    X = df[feature_cols].values

    # ── 3. Split ────────────────────────────────────────────────────────
    idx_train, idx_test = train_test_split(
        np.arange(len(df)), test_size=TEST_SIZE, random_state=RANDOM_SEED
    )
    idx_train, idx_val = train_test_split(
        idx_train, test_size=VAL_SIZE / (1 - TEST_SIZE), random_state=RANDOM_SEED
    )

    X_train, X_val, X_test = X[idx_train], X[idx_val], X[idx_test]

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s   = scaler.transform(X_val)
    X_test_s  = scaler.transform(X_test)

    log.info("Train=%d  Val=%d  Test=%d", len(X_train), len(X_val), len(X_test))

    out_dir = CENTRAL_ARTIFACTS / "agri_advisor"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_metrics = {}

    # ───────────────────────────────────────────────────────────────────
    # SUB-MODEL A: sunlight_hours (regression)
    # ───────────────────────────────────────────────────────────────────
    y_sun = df[AGRI_REG_TARGET].values
    y_sun_train, y_sun_val, y_sun_test = y_sun[idx_train], y_sun[idx_val], y_sun[idx_test]

    t0 = time.time()
    sun_model = RandomForestRegressor(**AGRI_RF_PARAMS)
    sun_model.fit(X_train_s, y_sun_train)
    log.info("sunlight_hours training: %.1fs", time.time() - t0)

    sun_pred = sun_model.predict(X_test_s)
    sun_metrics = regression_metrics(y_sun_test, sun_pred)
    log_metrics("sunlight_hours", sun_metrics)
    all_metrics["sunlight_hours"] = sun_metrics

    joblib.dump(sun_model, out_dir / "sunlight_model.joblib")

    # ───────────────────────────────────────────────────────────────────
    # SUB-MODEL B: irrigation_type (classification)
    # ───────────────────────────────────────────────────────────────────
    le_irr_type = LabelEncoder()
    y_type = le_irr_type.fit_transform(df["irrigation_type"].astype(str))
    y_type_train, y_type_test = y_type[idx_train], y_type[idx_test]

    t0 = time.time()
    type_model = GradientBoostingClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1, random_state=RANDOM_SEED
    )
    type_model.fit(X_train_s, y_type_train)
    log.info("irrigation_type training: %.1fs", time.time() - t0)

    type_pred = type_model.predict(X_test_s)
    type_metrics = classification_metrics(y_type_test, type_pred)
    log_metrics("irrigation_type", type_metrics)
    all_metrics["irrigation_type"] = type_metrics

    joblib.dump(type_model, out_dir / "irrigation_type_model.joblib")
    joblib.dump(le_irr_type, out_dir / "le_irrigation_type.joblib")

    # ───────────────────────────────────────────────────────────────────
    # SUB-MODEL C: irrigation_need (classification)
    # ───────────────────────────────────────────────────────────────────
    le_irr_need = LabelEncoder()
    y_need = le_irr_need.fit_transform(df["irrigation_need"].astype(str))
    y_need_train, y_need_test = y_need[idx_train], y_need[idx_test]

    t0 = time.time()
    need_model = GradientBoostingClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1, random_state=RANDOM_SEED
    )
    need_model.fit(X_train_s, y_need_train)
    log.info("irrigation_need training: %.1fs", time.time() - t0)

    need_pred = need_model.predict(X_test_s)
    need_metrics = classification_metrics(y_need_test, need_pred)
    log_metrics("irrigation_need", need_metrics)
    all_metrics["irrigation_need"] = need_metrics

    joblib.dump(need_model, out_dir / "irrigation_need_model.joblib")
    joblib.dump(le_irr_need, out_dir / "le_irrigation_need.joblib")

    # ── Save shared artifacts ──────────────────────────────────────────
    joblib.dump(scaler, out_dir / "scaler.joblib")
    for col, enc in encoders.items():
        joblib.dump(enc, out_dir / f"le_{col}.joblib")

    metadata = {
        "feature_cols": feature_cols,
        "cat_cols": cat_cols,
        "num_cols": num_cols,
        "metrics": all_metrics,
        "irrigation_type_classes": list(le_irr_type.classes_),
        "irrigation_need_classes": list(le_irr_need.classes_),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "trained_at": pd.Timestamp.now().isoformat(),
        "version": "1.0.0",
    }
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    log.info("Artifacts saved to %s", out_dir)
    return all_metrics


if __name__ == "__main__":
    train_agri_advisor()
