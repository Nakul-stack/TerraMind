"""
TerraMind - Model 1: Crop Recommender Training

Trains RandomForest and GradientBoosting classifiers on crop_dataset_rebuilt.csv,
evaluates both, keeps the best, saves artifacts to backend/artifacts/central/.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.core.config import (
    CROP_REC_FEATURES, CROP_REC_RF_PARAMS, CROP_REC_GB_PARAMS,
    CENTRAL_ARTIFACTS, TEST_SIZE, VAL_SIZE, RANDOM_SEED,
)
from backend.core.logging_config import log
from backend.utils.data_loader import load_crop_dataset
from backend.utils.metrics import classification_metrics, log_metrics
from backend.utils.calibration import calibrate_classifier


def train_crop_recommender():
    log.info("============================================")
    log.info("  Training Model 1 - Crop Recommender")
    log.info("============================================")

    # ── 1. Load data ────────────────────────────────────────────────────
    df = load_crop_dataset()
    log.info("Dataset shape: %s", df.shape)

    X = df[CROP_REC_FEATURES].values
    le = LabelEncoder()
    y = le.fit_transform(df["label"].values)
    labels = le.classes_
    log.info("Classes (%d): %s", len(labels), list(labels))

    # ── 2. Split ────────────────────────────────────────────────────────
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=VAL_SIZE / (1 - TEST_SIZE),
        random_state=RANDOM_SEED, stratify=y_temp
    )
    log.info("Train=%d  Val=%d  Test=%d", len(X_train), len(X_val), len(X_test))

    # ── 3. Scale features ──────────────────────────────────────────────
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s   = scaler.transform(X_val)
    X_test_s  = scaler.transform(X_test)

    # ── 4. Train both models ───────────────────────────────────────────
    results = {}

    # Random Forest
    t0 = time.time()
    rf = RandomForestClassifier(**CROP_REC_RF_PARAMS)
    rf.fit(X_train_s, y_train)
    rf_time = time.time() - t0
    rf_pred = rf.predict(X_test_s)
    rf_proba = rf.predict_proba(X_test_s)
    rf_metrics = classification_metrics(y_test, rf_pred, rf_proba, labels=list(range(len(labels))))
    rf_metrics["train_time_s"] = round(rf_time, 2)
    log_metrics("RandomForest", rf_metrics)
    results["rf"] = (rf, rf_metrics)

    # Gradient Boosting
    t0 = time.time()
    gb = GradientBoostingClassifier(**CROP_REC_GB_PARAMS)
    gb.fit(X_train_s, y_train)
    gb_time = time.time() - t0
    gb_pred = gb.predict(X_test_s)
    gb_proba = gb.predict_proba(X_test_s)
    gb_metrics = classification_metrics(y_test, gb_pred, gb_proba, labels=list(range(len(labels))))
    gb_metrics["train_time_s"] = round(gb_time, 2)
    log_metrics("GradientBoosting", gb_metrics)
    results["gb"] = (gb, gb_metrics)

    # ── 5. Select best ─────────────────────────────────────────────────
    best_name = max(results, key=lambda k: results[k][1]["macro_f1"])
    best_model, best_metrics = results[best_name]
    log.info("Best model: %s (F1=%.4f)", best_name, best_metrics["macro_f1"])

    # ── 6. Calibrate ───────────────────────────────────────────────────
    calibrated = calibrate_classifier(best_model, X_val_s, y_val)

    # ── 7. Save artifacts ──────────────────────────────────────────────
    out_dir = CENTRAL_ARTIFACTS / "crop_recommender"
    out_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(calibrated, out_dir / "model.joblib")
    joblib.dump(scaler,     out_dir / "scaler.joblib")
    joblib.dump(le,         out_dir / "label_encoder.joblib")

    metadata = {
        "model_type": best_name,
        "features": CROP_REC_FEATURES,
        "classes": list(labels),
        "n_classes": len(labels),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "metrics": {k: v for k, v in best_metrics.items() if k != "per_class"},
        "per_class_metrics": best_metrics.get("per_class", {}),
        "trained_at": pd.Timestamp.now().isoformat(),
        "version": "1.0.0",
    }
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    log.info("Artifacts saved to %s", out_dir)
    return calibrated, metadata


if __name__ == "__main__":
    train_crop_recommender()
