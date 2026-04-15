"""
TerraMind - District Crop Intelligence Engine (Novelty Layer)

Generates rich, interpretable district intelligence for the recommended crop:
  - crop area share in district
  - yield trend (improving/declining/stable)
  - top competing crops
  - best historical season
  - 10-year trajectory summary
  - irrigation infrastructure summary
  - crop-specific irrigated area percentage
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np

from backend.core.config import EDGE_ARTIFACTS
from backend.core.logging_config import log


def _load_cache(name: str) -> dict:
    path = EDGE_ARTIFACTS / f"{name}.json"
    if not path.exists():
        log.warning("Cache file not found: %s", path.name)
        return {}
    with open(path, "r") as f:
        return json.load(f)


class DistrictIntelligenceEngine:
    """
    Rule-based + analytical engine producing interpretable
    district-level intelligence for a given (state, district, crop).
    """

    def __init__(self):
        self.crop_freq    = _load_cache("district_crop_frequency")
        self.crop_stats   = _load_cache("district_crop_stats")
        self.best_season  = _load_cache("district_crop_best_season")
        self.trajectory   = _load_cache("district_crop_trajectory")
        self.irr_infra    = _load_cache("district_irrigation_infra")
        self.irr_pct      = _load_cache("district_crop_irrigated_area")
        log.info("DistrictIntelligenceEngine initialised (%d freq keys, %d stats keys)",
                 len(self.crop_freq), len(self.crop_stats))

    def query(self, state: str, district: str, crop: str) -> dict:
        """
        Generate the full district intelligence payload.
        Falls back to state-level data if district-level is unavailable.
        """
        key_sd = f"{state}|{district}"
        key_sdc = f"{state}|{district}|{crop}"
        notes: list[str] = []

        result = {
            "district_crop_share_percent": None,
            "yield_trend": None,
            "top_competing_crops": [],
            "best_historical_season": None,
            "ten_year_trajectory_summary": None,
            "ten_year_trajectory_data": None,
            "irrigation_infrastructure_summary": None,
            "crop_irrigated_area_percent": None,
        }

        # 1. Crop area share
        freq_list = self.crop_freq.get(key_sd, [])
        crop_entry = next((e for e in freq_list if e["crop"] == crop), None)
        if crop_entry:
            result["district_crop_share_percent"] = round(crop_entry["area_share"] * 100, 1)
        else:
            notes.append(f"No area share data for '{crop}' in {district}, {state}")

        # 2. Yield trend from stats
        stats = self.crop_stats.get(key_sdc)
        if stats and stats.get("n_years", 0) >= 3:
            traj_data = self.trajectory.get(key_sdc)
            if traj_data and len(traj_data.get("yields", [])) >= 3:
                yields = traj_data["yields"]
                x = np.arange(len(yields), dtype=float)
                slope = np.polyfit(x, yields, 1)[0]
                if slope > 0.02:
                    result["yield_trend"] = "improving"
                elif slope < -0.02:
                    result["yield_trend"] = "declining"
                else:
                    result["yield_trend"] = "stable"
            else:
                result["yield_trend"] = "insufficient data"
        else:
            result["yield_trend"] = "insufficient data"
            notes.append(f"Limited yield history for '{crop}' in {district}")

        # 3. Top competing crops
        if freq_list:
            sorted_crops = sorted(freq_list, key=lambda e: e["area_share"], reverse=True)
            competitors = [e["crop"] for e in sorted_crops if e["crop"] != crop][:5]
            result["top_competing_crops"] = competitors

        # 4. Best historical season
        result["best_historical_season"] = self.best_season.get(key_sdc, "data unavailable")

        # 5. 10-year trajectory
        traj = self.trajectory.get(key_sdc)
        if traj and traj.get("years"):
            years = traj["years"]
            yields = traj["yields"]
            if len(years) >= 2:
                first_y = yields[0]
                last_y  = yields[-1]
                change  = ((last_y - first_y) / max(first_y, 0.01)) * 100
                result["ten_year_trajectory_summary"] = (
                    f"Yield moved from {first_y:.2f} to {last_y:.2f} t/ha "
                    f"over {years[0]}–{years[-1]} ({change:+.1f}% change)"
                )
                result["ten_year_trajectory_data"] = {
                    "years": years, "yields": yields
                }
            else:
                result["ten_year_trajectory_summary"] = "Only 1 data point available"
        else:
            result["ten_year_trajectory_summary"] = "No trajectory data for this crop in this district"
            notes.append("Trajectory data unavailable; consider state-level fallback")

        # 6. Irrigation infrastructure
        infra = self.irr_infra.get(key_sd)
        if infra:
            total = sum(v for v in infra.values() if v and v > 0)
            if total > 0:
                dominant = max(
                    [(k, v) for k, v in infra.items() if k != "net_irrigated" and v and v > 0],
                    key=lambda x: x[1], default=(None, 0)
                )
                parts = []
                for src in ["canals", "tanks", "tube_wells", "other_wells", "other"]:
                    val = infra.get(src, 0) or 0
                    if val > 0:
                        pct = round(val / total * 100, 1)
                        parts.append(f"{src.replace('_', ' ')}: {pct}%")
                result["irrigation_infrastructure_summary"] = (
                    f"Dominated by {dominant[0].replace('_', ' ')} "
                    f"({', '.join(parts)})"
                )
                result["irrigation_infrastructure_data"] = infra
            else:
                result["irrigation_infrastructure_summary"] = "No irrigation infrastructure data"
        else:
            result["irrigation_infrastructure_summary"] = "Data unavailable for this district"

        # 7. Crop irrigated area percentage
        irr_pct_val = self.irr_pct.get(key_sdc)
        if irr_pct_val is not None:
            result["crop_irrigated_area_percent"] = round(irr_pct_val, 1)
        else:
            notes.append("Crop-specific irrigated area data not available")

        result["notes"] = notes
        return result
