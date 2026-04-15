"""
TerraMind - Name normalisation maps and utilities.

Provides canonical mappings for states, districts, crops, and seasons
so that heterogeneous datasets can be joined safely.
"""
from __future__ import annotations
import re
from typing import Optional

# ── Season normalisation ────────────────────────────────────────────────────
SEASON_MAP: dict[str, str] = {
    "kharif":     "kharif",
    "rabi":       "rabi",
    "whole year": "whole year",
    "annual":     "whole year",
    "autumn":     "autumn",
    "summer":     "summer",
    "winter":     "winter",
    "zaid":       "summer",
}

# ── Crop alias map (lowercase -> canonical lowercase) ────────────────────────
CROP_ALIAS: dict[str, str] = {
    "paddy":        "rice",
    "arhar/tur":    "pigeonpea",
    "tur":          "pigeonpea",
    "arhar":        "pigeonpea",
    "gram":         "chickpea",
    "bengal gram":  "chickpea",
    "bajra":        "pearl millet",
    "jowar":        "sorghum",
    "ragi":         "finger millet",
    "rapeseed":     "rapeseed and mustard",
    "mustard":      "rapeseed and mustard",
    "rape & mustard": "rapeseed and mustard",
    "ground nut":   "groundnut",
    "ginger":       "ginger",
    "mesta":        "mesta",
    "jute":         "jute",
    "cotton(lint)": "cotton",
    "cotton lint":  "cotton",
    "sesamum":      "sesamum",
    "sesame":       "sesamum",
    "linseed":      "linseed",
    "sunflower":    "sunflower",
    "soyabean":     "soybean",
    "soybean":      "soybean",
    "sugar cane":   "sugarcane",
    "tobacco":      "tobacco",
    "millets":      "millets",
    "pulses":       "pulses",
    "oilseeds":     "oilseeds",
}

# ── State alias map ─────────────────────────────────────────────────────────
STATE_ALIAS: dict[str, str] = {
    "andaman and nicobar islands":  "andaman and nicobar",
    "andaman & nicobar islands":    "andaman and nicobar",
    "andaman and nicobar":          "andaman and nicobar",
    "dadra and nagar haveli":       "dadra and nagar haveli",
    "dadra & nagar haveli":         "dadra and nagar haveli",
    "daman and diu":                "daman and diu",
    "daman & diu":                  "daman and diu",
    "jammu and kashmir":            "jammu and kashmir",
    "jammu & kashmir":              "jammu and kashmir",
    "nct of delhi":                 "delhi",
    "delhi":                        "delhi",
    "odisha":                       "odisha",
    "orissa":                       "odisha",
    "chhattisgarh":                 "chhattisgarh",
    "chattisgarh":                  "chhattisgarh",
    "pondicherry":                  "puducherry",
    "puducherry":                   "puducherry",
    "uttaranchal":                  "uttarakhand",
    "uttarakhand":                  "uttarakhand",
    "telangana":                    "telangana",
}


# ── ICRISAT crop-column -> canonical crop map ─────────────────────────────
# Column prefixes in ICRISAT-District Level Data.csv -> canonical crop name
ICRISAT_CROP_PREFIX_MAP: dict[str, str] = {
    "RICE":              "rice",
    "WHEAT":             "wheat",
    "KHARIF SORGHUM":    "sorghum",
    "RABI SORGHUM":      "sorghum",
    "SORGHUM":           "sorghum",
    "PEARL MILLET":      "pearl millet",
    "MAIZE":             "maize",
    "FINGER MILLET":     "finger millet",
    "BARLEY":            "barley",
    "CHICKPEA":          "chickpea",
    "PIGEONPEA":         "pigeonpea",
    "MINOR PULSES":      "pulses",
    "GROUNDNUT":         "groundnut",
    "SESAMUM":           "sesamum",
    "RAPESEED AND MUSTARD": "rapeseed and mustard",
    "SAFFLOWER":         "safflower",
    "CASTOR":            "castor",
    "LINSEED":           "linseed",
    "SUNFLOWER":         "sunflower",
    "SOYABEAN":          "soybean",
    "OILSEEDS":          "oilseeds",
    "SUGARCANE":         "sugarcane",
    "COTTON":            "cotton",
}


def _clean(text: str) -> str:
    """Lowercase, strip, collapse whitespace, remove trailing punctuation."""
    text = str(text).lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s&]", "", text)
    return text.strip()


def normalize_state(name: Optional[str]) -> str:
    """Return canonical lowercase state name."""
    if not name or str(name).lower() in ("nan", "none", ""):
        return ""
    key = _clean(name)
    return STATE_ALIAS.get(key, key)


def normalize_district(name: Optional[str]) -> str:
    """Return cleaned district name (lowercase, stripped)."""
    if not name or str(name).lower() in ("nan", "none", ""):
        return ""
    return _clean(name)


def normalize_crop(name: Optional[str]) -> str:
    """Return canonical lowercase crop name."""
    if not name or str(name).lower() in ("nan", "none", ""):
        return ""
    key = _clean(name)
    return CROP_ALIAS.get(key, key)


def normalize_season(name: Optional[str]) -> str:
    """Return canonical lowercase season name."""
    if not name or str(name).lower() in ("nan", "none", ""):
        return ""
    key = _clean(name)
    return SEASON_MAP.get(key, key)
