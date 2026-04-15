"""
Centralised name-normalisation utilities for the Pre-Sowing Advisor.

Every dataset spells state / district / crop / season names differently.
These helpers bring everything into a canonical lowercase, trimmed form and
resolve the most common aliases found across the seven CSV/XLSX/XLS files.

Usage:
    from ml.pre_sowing_advisor.normalizers import (
        normalize_state_name,
        normalize_district_name,
        normalize_crop_name,
        normalize_season,
    )
"""

from __future__ import annotations

import re
import unicodedata
from typing import Optional

# ---------------------------------------------------------------------------
# Generic text cleaner
# ---------------------------------------------------------------------------

def _clean(text: Optional[str]) -> str:
    """Lowercase, strip, collapse whitespace, remove non-ASCII noise."""
    if text is None:
        return ""
    text = str(text).strip().lower()
    # Normalise unicode (e.g. accented chars)
    text = unicodedata.normalize("NFKD", text)
    # Remove chars that are not alphanumeric, space, or basic punctuation
    text = re.sub(r"[^\w\s\-&/()]", "", text)
    # Collapse multiple spaces
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ---------------------------------------------------------------------------
# State normalisation
# ---------------------------------------------------------------------------

_STATE_ALIASES: dict[str, str] = {
    "andaman & nicobar islands": "andaman and nicobar",
    "andaman & nicobar": "andaman and nicobar",
    "a & n islands": "andaman and nicobar",
    "jammu & kashmir": "jammu and kashmir",
    "j&k": "jammu and kashmir",
    "dadra & nagar haveli": "dadra and nagar haveli",
    "d & n haveli": "dadra and nagar haveli",
    "daman & diu": "daman and diu",
    "orissa": "odisha",
    "pondicherry": "puducherry",
    "uttaranchal": "uttarakhand",
    "chattisgarh": "chhattisgarh",
    "delhi": "nct of delhi",
    "new delhi": "nct of delhi",
}


def normalize_state_name(name: Optional[str]) -> str:
    """Canonicalise a state name."""
    cleaned = _clean(name)
    return _STATE_ALIASES.get(cleaned, cleaned)


# ---------------------------------------------------------------------------
# District normalisation
# ---------------------------------------------------------------------------

_DISTRICT_ALIASES: dict[str, str] = {
    "bangalore": "bengaluru",
    "bangalore urban": "bengaluru urban",
    "bangalore rural": "bengaluru rural",
    "bombay": "mumbai",
    "calcutta": "kolkata",
    "madras": "chennai",
    "trivandrum": "thiruvananthapuram",
    "baroda": "vadodara",
    "poona": "pune",
    "shimoga": "shivamogga",
    "mysore": "mysuru",
    "belgaum": "belagavi",
    "gulbarga": "kalaburagi",
    "bellary": "ballari",
    "tumkur": "tumakuru",
    "raichur": "raichuru",
}


def normalize_district_name(name: Optional[str]) -> str:
    """Canonicalise a district name."""
    cleaned = _clean(name)
    return _DISTRICT_ALIASES.get(cleaned, cleaned)


# ---------------------------------------------------------------------------
# Crop normalisation
# ---------------------------------------------------------------------------

_CROP_ALIASES: dict[str, str] = {
    "arhar/tur": "tur",
    "arhar": "tur",
    "moong(green gram)": "moong",
    "green gram": "moong",
    "urad(black gram)": "urad",
    "black gram": "urad",
    "masoor(lentil)": "lentil",
    "masoor": "lentil",
    "rapeseed &mustard": "rapeseed and mustard",
    "rapeseed": "rapeseed and mustard",
    "mustard": "rapeseed and mustard",
    "sesamum": "sesame",
    "til": "sesame",
    "groundnut": "groundnut",
    "g.nut": "groundnut",
    "gram": "chickpea",
    "bengal gram": "chickpea",
    "paddy": "rice",
    "maize (corn)": "maize",
    "corn": "maize",
    "jowar": "sorghum",
    "bajra": "pearl millet",
    "ragi": "finger millet",
    "jute & mesta": "jute",
    "mesta": "jute",
    "sunflower": "sunflower",
    "soyabean": "soybean",
    "soybean": "soybean",
}


def normalize_crop_name(name: Optional[str]) -> str:
    """Canonicalise a crop name."""
    cleaned = _clean(name)
    return _CROP_ALIASES.get(cleaned, cleaned)


# ---------------------------------------------------------------------------
# Season normalisation
# ---------------------------------------------------------------------------

_SEASON_ALIASES: dict[str, str] = {
    "whole year": "annual",
    "total": "annual",
    "summer": "zaid",
    "autumn": "kharif",
    "winter": "rabi",
}


def normalize_season(name: Optional[str]) -> str:
    """Canonicalise a season name."""
    cleaned = _clean(name)
    return _SEASON_ALIASES.get(cleaned, cleaned)
