"""
TerraMind - Robust Data Loaders

Loads every dataset from ``./dataset before sowing/`` using relative paths.
Handles CSV, XLSX, and XLS gracefully with fallback logging.
"""
from __future__ import annotations

import warnings
from typing import Optional

import pandas as pd

from backend.core.config import (
    CROP_DATASET, IRRIGATION_DATASET, INDIA_AGRI_CSV,
    CROP_PROD_XLSX, ICRISAT_MAIN, ICRISAT_SOURCE,
    ICRISAT_IRRIGATION, MERGE_XLS,
)
from backend.core.logging_config import log
from backend.utils.naming_maps import (
    normalize_state, normalize_district, normalize_crop, normalize_season,
)


# ── Helpers ─────────────────────────────────────────────────────────────────

def _safe_read_csv(path, **kw) -> Optional[pd.DataFrame]:
    try:
        df = pd.read_csv(path, **kw)
        log.info("Loaded %s -> %s rows x %s cols", path.name, len(df), len(df.columns))
        return df
    except Exception as exc:
        log.error("Failed to load %s: %s", path, exc)
        return None


def _safe_read_excel(path, engine=None, **kw) -> Optional[pd.DataFrame]:
    try:
        df = pd.read_excel(path, engine=engine, **kw)
        log.info("Loaded %s  ->  %s rows x %s cols", path.name, len(df), len(df.columns))
        return df
    except Exception as exc:
        log.warning("Could not load %s (engine=%s): %s - skipping.", path.name, engine, exc)
        return None


# ── Public loaders ──────────────────────────────────────────────────────────

def load_crop_dataset() -> pd.DataFrame:
    """Load crop_dataset_rebuilt.csv for Model 1 (Crop Recommender)."""
    df = _safe_read_csv(CROP_DATASET)
    if df is None:
        raise FileNotFoundError(f"Critical dataset missing: {CROP_DATASET}")
    df.columns = [c.strip() for c in df.columns]
    df["label"] = df["label"].apply(normalize_crop)
    df.dropna(subset=["label"], inplace=True)
    return df


def load_irrigation_dataset() -> pd.DataFrame:
    """Load irrigation_prediction.csv for Model 3 (Agri-Condition Advisor)."""
    df = _safe_read_csv(IRRIGATION_DATASET)
    if df is None:
        raise FileNotFoundError(f"Critical dataset missing: {IRRIGATION_DATASET}")
    df.columns = [c.strip().lower() for c in df.columns]
    df["crop"] = df["crop"].apply(normalize_crop)
    df["season"] = df["season"].apply(normalize_season)
    return df


def load_india_agri_production() -> Optional[pd.DataFrame]:
    """Load India Agriculture Crop Production.csv for yield + intelligence."""
    df = _safe_read_csv(INDIA_AGRI_CSV)
    if df is None:
        return None
    df.columns = [c.strip() for c in df.columns]
    # Standardise names
    df["State"]    = df["State"].apply(normalize_state)
    df["District"] = df["District"].apply(normalize_district)
    df["Crop"]     = df["Crop"].apply(normalize_crop)
    df["Season"]   = df["Season"].apply(normalize_season)
    # Parse year - format "2001-02" -> 2001
    df["Year_Num"] = df["Year"].astype(str).str[:4]
    df["Year_Num"] = pd.to_numeric(df["Year_Num"], errors="coerce")
    # Compute yield safely where missing
    if "Yield" in df.columns:
        mask_missing = df["Yield"].isna()
        safe = (df["Area"].notna()) & (df["Area"] > 0) & (df["Production"].notna())
        df.loc[mask_missing & safe, "Yield"] = (
            df.loc[mask_missing & safe, "Production"] / df.loc[mask_missing & safe, "Area"]
        )
    return df


def load_crop_production_xlsx() -> Optional[pd.DataFrame]:
    """Load crop_production.csv.xlsx - secondary yield source."""
    df = _safe_read_excel(CROP_PROD_XLSX, engine="openpyxl")
    if df is None:
        return None
    df.columns = [c.strip() for c in df.columns]
    rename = {
        "State_Name":    "State",
        "District_Name": "District",
        "Crop_Year":     "Year_Num",
        "label":         "Crop",
    }
    df.rename(columns=rename, inplace=True)
    df["State"]    = df["State"].apply(normalize_state)
    df["District"] = df["District"].apply(normalize_district)
    df["Crop"]     = df["Crop"].apply(normalize_crop)
    df["Season"]   = df["Season"].apply(normalize_season)
    df["Year_Num"] = pd.to_numeric(df["Year_Num"], errors="coerce")
    # Derive yield
    safe = (df["Area"].notna()) & (df["Area"] > 0) & (df["Production"].notna())
    df.loc[safe, "Yield"] = df.loc[safe, "Production"] / df.loc[safe, "Area"]
    return df


def load_icrisat_main() -> Optional[pd.DataFrame]:
    """Load ICRISAT-District Level Data.csv (wide-format crop area/prod/yield)."""
    df = _safe_read_csv(ICRISAT_MAIN)
    if df is None:
        return None
    df.columns = [c.strip() for c in df.columns]
    df.rename(columns={"State Name": "State", "Dist Name": "District"}, inplace=True)
    df["State"]    = df["State"].apply(normalize_state)
    df["District"] = df["District"].apply(normalize_district)
    # Replace sentinel -1 with NaN
    numeric_cols = df.select_dtypes("number").columns
    df[numeric_cols] = df[numeric_cols].replace(-1, pd.NA)
    return df


def load_icrisat_source() -> Optional[pd.DataFrame]:
    """Load ICRISAT-District Level Data Source.csv (irrigation infra sources)."""
    df = _safe_read_csv(ICRISAT_SOURCE)
    if df is None:
        return None
    df.columns = [c.strip() for c in df.columns]
    df.rename(columns={"State Name": "State", "Dist Name": "District"}, inplace=True)
    df["State"]    = df["State"].apply(normalize_state)
    df["District"] = df["District"].apply(normalize_district)
    numeric_cols = df.select_dtypes("number").columns
    df[numeric_cols] = df[numeric_cols].replace(-1, pd.NA)
    return df


def load_icrisat_irrigation() -> Optional[pd.DataFrame]:
    """Load ICRISAT-District Level Data Irrigation.csv (crop irrigated areas)."""
    df = _safe_read_csv(ICRISAT_IRRIGATION)
    if df is None:
        return None
    df.columns = [c.strip() for c in df.columns]
    df.rename(columns={"State Name": "State", "Dist Name": "District"}, inplace=True)
    df["State"]    = df["State"].apply(normalize_state)
    df["District"] = df["District"].apply(normalize_district)
    numeric_cols = df.select_dtypes("number").columns
    df[numeric_cols] = df[numeric_cols].replace(-1, pd.NA)
    return df


def load_merge_xls() -> Optional[pd.DataFrame]:
    """Attempt to load the .xls merged file; skip gracefully if xlrd unavailable."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = _safe_read_excel(MERGE_XLS, engine="xlrd")
    if df is not None:
        df.columns = [c.strip() for c in df.columns]
    return df


def load_combined_yield_data() -> pd.DataFrame:
    """
    Merge India Agri CSV + crop_production XLSX into a single yield dataset.
    Returns DataFrame with columns:
        State, District, Crop, Year_Num, Season, Area, Production, Yield
    """
    frames = []

    ia = load_india_agri_production()
    if ia is not None:
        cols = ["State", "District", "Crop", "Year_Num", "Season", "Area", "Production", "Yield"]
        ia = ia[[c for c in cols if c in ia.columns]]
        frames.append(ia)

    cp = load_crop_production_xlsx()
    if cp is not None:
        cols = ["State", "District", "Crop", "Year_Num", "Season", "Area", "Production", "Yield"]
        cp = cp[[c for c in cols if c in cp.columns]]
        frames.append(cp)

    if not frames:
        raise FileNotFoundError("No yield datasets could be loaded.")

    combined = pd.concat(frames, ignore_index=True)
    combined.drop_duplicates(subset=["State", "District", "Crop", "Year_Num", "Season"], inplace=True)
    combined.dropna(subset=["Yield"], inplace=True)

    # Remove extreme outliers (yield > 99.5th percentile globally)
    q995 = combined["Yield"].quantile(0.995)
    combined = combined[combined["Yield"] <= q995]

    log.info("Combined yield dataset: %d rows", len(combined))
    return combined
