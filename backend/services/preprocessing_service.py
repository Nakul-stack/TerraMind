"""
TerraMind - Preprocessing Service

Transforms cleaned user inputs into model-ready feature vectors
for each of the three models.
"""
from __future__ import annotations

import numpy as np

from backend.core.config import CROP_REC_FEATURES
from backend.core.logging_config import log


class PreprocessingService:
    """Converts normalised user input dict -> feature arrays for each model."""

    @staticmethod
    def for_crop_recommender(inputs: dict, scaler) -> np.ndarray:
        """Build feature vector for Model 1."""
        features = [float(inputs.get(f, 0)) for f in CROP_REC_FEATURES]
        X = np.array(features).reshape(1, -1)
        return scaler.transform(X)

    @staticmethod
    def for_yield_predictor(
        inputs: dict,
        crop: str,
        scaler,
        le_crop, le_state, le_district, le_season,
        metadata: dict,
    ) -> np.ndarray:
        """
        Build feature vector for Model 2.
        Uses label encoders, falling back to the most common class
        when a value is unseen.
        """
        def safe_encode(le, val: str) -> int:
            val = str(val).lower().strip()
            if val in le.classes_:
                return int(le.transform([val])[0])
            log.warning("Unseen label '%s' for encoder - using fallback 0", val)
            return 0

        crop_enc     = safe_encode(le_crop, crop)
        state_enc    = safe_encode(le_state, inputs.get("state", ""))
        district_enc = safe_encode(le_district, inputs.get("district", ""))
        season_enc   = safe_encode(le_season, inputs.get("season", ""))
        area_log     = float(np.log1p(max(float(inputs.get("area") or 1.0), 0)))

        # For lag features, use 0 (will be filled from cache or default)
        # In production, these would come from cached district stats
        features = [
            crop_enc, state_enc, district_enc, season_enc,
            area_log,
            0.0,  # yield_lag1  - filled by inference pipeline from cache
            0.0,  # yield_lag2
            0.0,  # yield_lag3
            0.0,  # yield_rolling3_mean
            0.0,  # yield_rolling5_mean
            0.0,  # area_lag1
            0.0,  # yield_trend_slope
        ]
        # Remove duplicates from feature list based on metadata
        expected_n = len(metadata.get("features", features))
        features = features[:expected_n]

        X = np.array(features).reshape(1, -1)
        return scaler.transform(X)

    @staticmethod
    def for_agri_advisor(
        inputs: dict,
        crop: str,
        scaler,
        le_crop, le_soil_type, le_season,
    ) -> np.ndarray:
        """Build feature vector for Model 3."""
        def safe_encode(le, val: str) -> int:
            val = str(val).lower().strip()
            if val in le.classes_:
                return int(le.transform([val])[0])
            return 0

        features = [
            float(inputs.get("ph", 6.5)),
            float(inputs.get("temperature", 25.0)),
            float(inputs.get("humidity", 60.0)),
            float(inputs.get("rainfall", 100.0)),
            safe_encode(le_crop, crop),
            safe_encode(le_soil_type, inputs.get("soil_type", "")),
            safe_encode(le_season, inputs.get("season", "")),
        ]
        X = np.array(features).reshape(1, -1)
        return scaler.transform(X)

    @staticmethod
    def enrich_yield_features_from_cache(
        X: np.ndarray,
        state: str,
        district: str,
        crop: str,
        crop_stats_cache: dict,
    ) -> np.ndarray:
        """
        Fill lag/rolling features from cached district stats.
        Modifies positions 5–11 in the feature vector.
        """
        key = f"{state}|{district}|{crop}"
        stats = crop_stats_cache.get(key, {})

        if stats:
            mean_yield = stats.get("mean_yield", 0)
            std_yield  = stats.get("std_yield", 0)
            X[0, 5]  = mean_yield          # yield_lag1 proxy
            X[0, 6]  = mean_yield * 0.95   # yield_lag2 proxy
            X[0, 7]  = mean_yield * 0.90   # yield_lag3 proxy
            X[0, 8]  = mean_yield          # rolling3_mean proxy
            X[0, 9]  = mean_yield          # rolling5_mean proxy
            X[0, 10] = stats.get("mean_area", 0)  # area_lag1
            # yield_trend is already index 11
        return X
