"""
TerraMind - Edge Model Compression

Creates lightweight versions of centralized models:
  - Reduced-tree RandomForest for crop recommender
  - Smaller GBR for yield predictor
  - Copies agri-advisor models (already compact)
Also triggers cache building for edge priors.
"""
from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.core.config import (
    CENTRAL_ARTIFACTS, EDGE_ARTIFACTS,
    EDGE_RF_N_ESTIMATORS, EDGE_MAX_DEPTH,
)
from backend.core.logging_config import log
from backend.utils.cache_builder import build_all_caches


def _compress_rf(central_path: Path, edge_path: Path):
    """Reduce a RandomForest by keeping only a subset of estimators."""
    model = joblib.load(central_path / "model.joblib")
    # Check if it's a calibrated model wrapping an RF
    inner = model
    if hasattr(model, "estimator"):
        inner = model.estimator
    elif hasattr(model, "estimators_"):
        inner = model

    if hasattr(inner, "estimators_"):
        n_keep = min(EDGE_RF_N_ESTIMATORS, len(inner.estimators_))
        inner.estimators_ = inner.estimators_[:n_keep]
        inner.n_estimators = n_keep
        log.info("Compressed RF: kept %d/%d trees", n_keep, len(inner.estimators_) + n_keep)

    edge_path.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, edge_path / "model.joblib")
    # Copy scaler and encoders
    for f in central_path.glob("*.joblib"):
        if f.name != "model.joblib":
            shutil.copy2(f, edge_path / f.name)
    # Copy metadata
    meta_path = central_path / "metadata.json"
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        meta["edge_compressed"] = True
        meta["edge_n_estimators"] = EDGE_RF_N_ESTIMATORS
        with open(edge_path / "metadata.json", "w") as f:
            json.dump(meta, f, indent=2)


def _compress_gbr(central_path: Path, edge_path: Path):
    """Copy GBR model and associated artifacts to edge (already compact)."""
    edge_path.mkdir(parents=True, exist_ok=True)
    for f in central_path.iterdir():
        if f.is_file():
            shutil.copy2(f, edge_path / f.name)
    # Update metadata
    meta_path = edge_path / "metadata.json"
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        meta["edge_compressed"] = True
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
    log.info("Copied GBR model to edge: %s", edge_path)


def compress_all_edge_models():
    log.info("============================================")
    log.info("  Building Edge-Deployable Assets")
    log.info("============================================")

    t0 = time.time()

    # 1. Compress Crop Recommender
    cr_central = CENTRAL_ARTIFACTS / "crop_recommender"
    cr_edge    = EDGE_ARTIFACTS / "crop_recommender"
    if cr_central.exists():
        _compress_rf(cr_central, cr_edge)
        log.info("[OK] Crop Recommender edge model ready")
    else:
        log.warning("Central crop_recommender not found - train first!")

    # 2. Copy Yield Predictor
    yp_central = CENTRAL_ARTIFACTS / "yield_predictor"
    yp_edge    = EDGE_ARTIFACTS / "yield_predictor"
    if yp_central.exists():
        _compress_gbr(yp_central, yp_edge)
        log.info("[OK] Yield Predictor edge model ready")
    else:
        log.warning("Central yield_predictor not found - train first!")

    # 3. Copy Agri Advisor
    aa_central = CENTRAL_ARTIFACTS / "agri_advisor"
    aa_edge    = EDGE_ARTIFACTS / "agri_advisor"
    if aa_central.exists():
        _compress_gbr(aa_central, aa_edge)
        log.info("[OK] Agri Advisor edge model ready")
    else:
        log.warning("Central agri_advisor not found - train first!")

    # 4. Build all caches
    build_all_caches()

    # 5. Report artifact sizes
    log.info("── Edge Artifact Sizes ──")
    for item in sorted(EDGE_ARTIFACTS.rglob("*")):
        if item.is_file():
            size_kb = item.stat().st_size / 1024
            log.info("  %s  ->  %.1f KB", item.relative_to(EDGE_ARTIFACTS), size_kb)

    log.info("Edge build completed in %.1fs", time.time() - t0)


if __name__ == "__main__":
    compress_all_edge_models()
