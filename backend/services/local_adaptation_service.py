"""
TerraMind - Local Adaptation Service

Bounded post-prediction module that applies district/state/agro-climatic
adjustments to crop recommender probabilities.

Strategy:
  1. Run global crop recommender -> top-K crops + probabilities
  2. Apply small bounded local adjustments
  3. Re-rank cautiously
  4. Never let local logic completely override strong global evidence

Adjustment sources:
  - District historical crop frequency prior
  - District-season crop suitability
  - State/agro-climatic prior
  - Irrigation availability compatibility
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np

from backend.core.config import (
    EDGE_ARTIFACTS, LOCAL_ADAPT_MAX_SHIFT,
    LOCAL_ADAPT_DECAY, ADAPTATION_MIN_SAMPLES,
)
from backend.core.logging_config import log
from backend.utils.partitioning import get_zone_for_state


class LocalAdaptationService:
    """Applies bounded local priors to global crop probabilities."""

    def __init__(self):
        self._load_caches()

    def _load_caches(self):
        """Load all cached priors from edge artifacts."""
        self.crop_freq   = self._load_json("district_crop_frequency.json")
        self.crop_stats  = self._load_json("district_crop_stats.json")
        self.best_season = self._load_json("district_crop_best_season.json")
        self.agro_zones  = self._load_json("agro_climatic_zones.json")

    def _load_json(self, name: str) -> dict:
        path = EDGE_ARTIFACTS / name
        if path.exists():
            with open(path) as f:
                return json.load(f)
        return {}

    def adapt(
        self,
        state: str,
        district: str,
        season: str,
        crop_probabilities: dict[str, float],
    ) -> dict:
        """
        Apply bounded local adaptation to crop probabilities.

        Args:
            state: normalised state name
            district: normalised district name
            season: normalised season
            crop_probabilities: {crop_name: probability} from global model

        Returns:
            {
              "original_probs": {...},
              "adapted_probs": {...},
              "adjustments": {...},
              "adaptation_applied": bool,
              "top_factors": [...],
              "notes": [...]
            }
        """
        result = {
            "original_probs": dict(crop_probabilities),
            "adapted_probs": dict(crop_probabilities),
            "adjustments": {},
            "adaptation_applied": False,
            "top_factors": [],
            "notes": [],
        }

        key_sd = f"{state}|{district}"
        freq_list = self.crop_freq.get(key_sd, [])

        if not freq_list:
            result["notes"].append(
                f"No district frequency data for {district}, {state} - "
                f"no local adaptation applied"
            )
            return result

        # Build area-share lookup for this district
        area_shares = {e["crop"]: e["area_share"] for e in freq_list}

        adapted = dict(crop_probabilities)
        adjustments = {}
        factors = []

        for crop, base_prob in crop_probabilities.items():
            total_shift = 0.0

            # Factor 1: District crop frequency prior
            share = area_shares.get(crop, 0.0)
            if share > 0:
                # Proportional nudge: higher share -> stronger nudge, but bounded
                freq_shift = min(share * 0.3, LOCAL_ADAPT_MAX_SHIFT * 0.5)
                total_shift += freq_shift
                factors.append(f"{crop}: frequency prior +{freq_shift:.3f} (area share={share:.1%})")

            # Factor 2: Season suitability
            season_key = f"{state}|{district}|{crop}"
            best_s = self.best_season.get(season_key)
            if best_s and best_s == season:
                season_shift = LOCAL_ADAPT_MAX_SHIFT * 0.3
                total_shift += season_shift
                factors.append(f"{crop}: season match +{season_shift:.3f}")
            elif best_s and best_s != season:
                season_penalty = -LOCAL_ADAPT_MAX_SHIFT * 0.15
                total_shift += season_penalty
                factors.append(f"{crop}: season mismatch {season_penalty:.3f}")

            # Factor 3: Yield history quality
            stats_key = f"{state}|{district}|{crop}"
            stats = self.crop_stats.get(stats_key)
            if stats and stats.get("n_years", 0) >= 5:
                # Crop has solid history -> mild boost
                history_shift = LOCAL_ADAPT_MAX_SHIFT * 0.1
                total_shift += history_shift
                factors.append(f"{crop}: strong history +{history_shift:.3f}")

            # Bound the total shift
            total_shift = np.clip(total_shift, -LOCAL_ADAPT_MAX_SHIFT, LOCAL_ADAPT_MAX_SHIFT)
            adjustments[crop] = round(float(total_shift), 4)
            adapted[crop] = base_prob + total_shift

        # Normalise to valid probability distribution
        values = np.array(list(adapted.values()))
        values = np.clip(values, 0.01, None)    # floor at 1%
        values = values / values.sum()           # re-normalise
        adapted = dict(zip(adapted.keys(), [round(float(v), 4) for v in values]))

        result["adapted_probs"] = adapted
        result["adjustments"] = adjustments
        result["adaptation_applied"] = any(abs(v) > 0.001 for v in adjustments.values())
        result["top_factors"] = factors[:10]

        if result["adaptation_applied"]:
            log.info("Local adaptation applied for %s, %s - %d factors",
                     district, state, len(factors))

        return result
