"""
TerraMind - Cache Builder

Precomputes and serialises all district-level lookup tables needed
for edge inference and the district intelligence novelty layer.

Outputs JSON files under artifacts/edge/:
  - district_crop_frequency.json
  - district_crop_stats.json
  - district_irrigation_infra.json
  - district_crop_irrigated_area.json
  - state_district_map.json
  - agro_climatic_zones.json (stub - configurable later)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from backend.core.config import EDGE_ARTIFACTS, CENTRAL_ARTIFACTS
from backend.core.logging_config import log
from backend.utils.data_loader import (
    load_combined_yield_data,
    load_icrisat_main,
    load_icrisat_source,
    load_icrisat_irrigation,
)
from backend.utils.feature_engineering import (
    build_crop_frequency_prior,
    compute_district_crop_stats,
)
from backend.utils.naming_maps import ICRISAT_CROP_PREFIX_MAP, normalize_crop


class NumpyEncoder(json.JSONEncoder):
    """Handle numpy types in JSON serialisation."""
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if pd.isna(obj):
            return None
        return super().default(obj)


def _save_json(data, path: Path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, cls=NumpyEncoder, indent=2)
    log.info("Saved cache: %s  (%d bytes)", path.name, path.stat().st_size)


def build_all_caches():
    """Build every cache file needed for edge / district intelligence."""
    log.info("=== Building edge caches ===")

    # 1. Yield data -> crop frequency + crop stats
    yield_df = load_combined_yield_data()

    freq = build_crop_frequency_prior(yield_df)
    freq_dict: dict = {}
    for _, row in freq.iterrows():
        key = f"{row['State']}|{row['District']}"
        freq_dict.setdefault(key, []).append({
            "crop": row["Crop"],
            "area_share": round(float(row["area_share"]), 4),
            "rank": int(row["frequency_rank"]),
        })
    _save_json(freq_dict, EDGE_ARTIFACTS / "district_crop_frequency.json")

    stats = compute_district_crop_stats(yield_df)
    stats_dict: dict = {}
    for _, row in stats.iterrows():
        key = f"{row['State']}|{row['District']}|{row['Crop']}"
        stats_dict[key] = {
            "mean_yield": round(float(row["mean_yield"]), 3),
            "std_yield": round(float(row["std_yield"]), 3),
            "median_yield": round(float(row["median_yield"]), 3),
            "n_years": int(row["n_years"]),
            "first_year": int(row["first_year"]) if pd.notna(row["first_year"]) else None,
            "last_year": int(row["last_year"]) if pd.notna(row["last_year"]) else None,
        }
    _save_json(stats_dict, EDGE_ARTIFACTS / "district_crop_stats.json")

    # State-district map
    sd_map: dict = {}
    for state in yield_df["State"].dropna().unique():
        districts = sorted(yield_df.loc[yield_df["State"] == state, "District"].dropna().unique().tolist())
        if districts:
            sd_map[state] = districts
    _save_json(sd_map, EDGE_ARTIFACTS / "state_district_map.json")
    _save_json(sd_map, CENTRAL_ARTIFACTS / "state_district_map.json")

    # 2. ICRISAT source -> irrigation infrastructure
    src = load_icrisat_source()
    if src is not None:
        infra_dict: dict = {}
        for (state, dist), grp in src.groupby(["State", "District"], observed=True):
            latest = grp.sort_values("Year").iloc[-1]

            def safe_float(val):
                if pd.isna(val):
                    return 0.0
                try:
                    return float(val)
                except (ValueError, TypeError):
                    return 0.0

            infra_dict[f"{state}|{dist}"] = {
                "canals":      safe_float(latest.get("CANALS AREA (1000 ha)")),
                "tanks":       safe_float(latest.get("TANKS AREA (1000 ha)")),
                "tube_wells":  safe_float(latest.get("TUBE WELLS AREA (1000 ha)")),
                "other_wells": safe_float(latest.get("OTHER WELLS AREA (1000 ha)")),
                "other":       safe_float(latest.get("OTHER SOURCES AREA (1000 ha)")),
                "net_irrigated": safe_float(latest.get("NET AREA (1000 ha)")),
            }
        _save_json(infra_dict, EDGE_ARTIFACTS / "district_irrigation_infra.json")

    # 3. ICRISAT irrigation -> crop irrigated area percentages
    irr = load_icrisat_irrigation()
    icr_main = load_icrisat_main()
    if irr is not None and icr_main is not None:
        irr_pct_dict: dict = {}
        for (state, dist), irr_grp in irr.groupby(["State", "District"], observed=True):
            main_grp = icr_main[(icr_main["State"] == state) & (icr_main["District"] == dist)]
            if main_grp.empty:
                continue
            latest_irr = irr_grp.sort_values("Year").iloc[-1]
            latest_main = main_grp.sort_values("Year").iloc[-1]
            for prefix, crop_name in ICRISAT_CROP_PREFIX_MAP.items():
                irr_col = f"{prefix} IRRIGATED AREA (1000 ha)"
                area_col = f"{prefix} AREA (1000 ha)"
                if irr_col in latest_irr.index and area_col in latest_main.index:
                    irr_val = latest_irr.get(irr_col)
                    area_val = latest_main.get(area_col)
                    if pd.notna(irr_val) and pd.notna(area_val) and area_val > 0:
                        pct = round(float(irr_val / area_val * 100), 1)
                        key = f"{state}|{dist}|{crop_name}"
                        irr_pct_dict[key] = min(pct, 100.0)
        _save_json(irr_pct_dict, EDGE_ARTIFACTS / "district_crop_irrigated_area.json")

    # 4. Best season per (district, crop)
    season_stats = yield_df.groupby(
        ["State", "District", "Crop", "Season"], observed=True
    ).agg(
        mean_yield=("Yield", "mean"),
        count=("Yield", "count"),
    ).reset_index()
    best_season: dict = {}
    for (state, dist, crop), grp in season_stats.groupby(["State", "District", "Crop"], observed=True):
        if grp.empty:
            continue
        best = grp.loc[grp["mean_yield"].idxmax()]
        best_season[f"{state}|{dist}|{crop}"] = best["Season"]
    _save_json(best_season, EDGE_ARTIFACTS / "district_crop_best_season.json")

    # 5. 10-year trajectory per (district, crop)
    trajectory: dict = {}
    recent = yield_df[yield_df["Year_Num"] >= yield_df["Year_Num"].max() - 10].copy()
    for (state, dist, crop), grp in recent.groupby(["State", "District", "Crop"], observed=True):
        yearly = grp.groupby("Year_Num", observed=True)["Yield"].mean().sort_index()
        trajectory[f"{state}|{dist}|{crop}"] = {
            "years": yearly.index.astype(int).tolist(),
            "yields": [round(float(v), 3) for v in yearly.values],
        }
    _save_json(trajectory, EDGE_ARTIFACTS / "district_crop_trajectory.json")

    # 6. Stub agro-climatic zones - configurable mapping file
    agro_zones = {
        "western himalayan": ["jammu and kashmir", "himachal pradesh", "uttarakhand"],
        "eastern himalayan": ["assam", "sikkim", "meghalaya", "arunachal pradesh", "nagaland", "manipur", "mizoram", "tripura"],
        "upper gangetic plains": ["uttar pradesh"],
        "lower gangetic plains": ["west bengal", "bihar"],
        "middle gangetic plains": ["bihar", "jharkhand"],
        "trans gangetic plains": ["punjab", "haryana", "delhi", "chandigarh"],
        "eastern plateau and hills": ["chhattisgarh", "jharkhand", "odisha"],
        "central plateau and hills": ["madhya pradesh", "rajasthan", "uttar pradesh"],
        "western plateau and hills": ["maharashtra"],
        "southern plateau and hills": ["karnataka", "andhra pradesh", "telangana"],
        "east coast": ["tamil nadu", "andhra pradesh", "odisha"],
        "west coast": ["goa", "kerala", "karnataka"],
        "gujarat plains": ["gujarat"],
        "western dry region": ["rajasthan"],
        "island region": ["andaman and nicobar", "lakshadweep"],
    }
    _save_json(agro_zones, EDGE_ARTIFACTS / "agro_climatic_zones.json")

    log.info("=== All edge caches built successfully ===")


if __name__ == "__main__":
    build_all_caches()
