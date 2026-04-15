"""
TerraMind - Feature Engineering utilities.

Provides feature transformations for the yield predictor and
district intelligence modules.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from backend.core.logging_config import log


def engineer_yield_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Given a combined yield DataFrame (State, District, Crop, Year_Num, Season, Area, Production, Yield),
    compute historical lag/rolling features per (State, District, Crop, Season) group.

    Returns enriched DataFrame with additional columns:
        - yield_lag1, yield_lag2, yield_lag3
        - yield_rolling3_mean, yield_rolling5_mean
        - area_lag1
        - production_lag1
        - yield_trend_slope  (linear slope over last 5 years)
    """
    df = df.copy()
    df.sort_values(["State", "District", "Crop", "Season", "Year_Num"], inplace=True)

    grp = df.groupby(["State", "District", "Crop", "Season"], observed=True)

    # Lag features
    df["yield_lag1"] = grp["Yield"].shift(1)
    df["yield_lag2"] = grp["Yield"].shift(2)
    df["yield_lag3"] = grp["Yield"].shift(3)

    # Rolling means
    df["yield_rolling3_mean"] = grp["Yield"].transform(
        lambda s: s.shift(1).rolling(3, min_periods=1).mean()
    )
    df["yield_rolling5_mean"] = grp["Yield"].transform(
        lambda s: s.shift(1).rolling(5, min_periods=2).mean()
    )

    # Area / production lag
    df["area_lag1"]       = grp["Area"].shift(1)
    df["production_lag1"] = grp["Production"].shift(1)

    # Linear trend slope over last 5 observations
    def _slope(series):
        vals = series.dropna().values[-5:]
        if len(vals) < 3:
            return np.nan
        x = np.arange(len(vals), dtype=float)
        slope = np.polyfit(x, vals, 1)[0]
        return slope

    df["yield_trend_slope"] = grp["Yield"].transform(
        lambda s: s.expanding(min_periods=3).apply(_slope, raw=False)
    )

    log.info("Engineered yield features - %d rows, %d cols", len(df), len(df.columns))
    return df


def compute_district_crop_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Precompute summary statistics per (State, District, Crop) for use
    in the yield predictor and district intelligence.

    Returns DataFrame with one row per (State, District, Crop):
        - mean_yield, std_yield, median_yield
        - total_area, mean_area
        - n_years
        - first_year, last_year
    """
    grp = df.groupby(["State", "District", "Crop"], observed=True)

    stats = grp.agg(
        mean_yield    = ("Yield", "mean"),
        std_yield     = ("Yield", "std"),
        median_yield  = ("Yield", "median"),
        total_area    = ("Area", "sum"),
        mean_area     = ("Area", "mean"),
        n_years       = ("Year_Num", "nunique"),
        first_year    = ("Year_Num", "min"),
        last_year     = ("Year_Num", "max"),
    ).reset_index()

    stats["std_yield"] = stats["std_yield"].fillna(0)
    return stats


def build_crop_frequency_prior(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build per-district crop frequency table.

    Returns DataFrame: State, District, Crop, area_share, frequency_rank
    """
    # Total area per district
    dist_total = df.groupby(["State", "District"], observed=True)["Area"].sum().reset_index()
    dist_total.rename(columns={"Area": "district_total_area"}, inplace=True)

    # Crop area per district
    crop_area = df.groupby(["State", "District", "Crop"], observed=True)["Area"].sum().reset_index()
    crop_area.rename(columns={"Area": "crop_total_area"}, inplace=True)

    merged = crop_area.merge(dist_total, on=["State", "District"], how="left")
    merged["area_share"] = (merged["crop_total_area"] / merged["district_total_area"]).clip(0, 1)
    merged["frequency_rank"] = merged.groupby(["State", "District"], observed=True)["crop_total_area"].rank(
        ascending=False, method="min"
    )

    return merged[["State", "District", "Crop", "area_share", "frequency_rank", "crop_total_area"]]
