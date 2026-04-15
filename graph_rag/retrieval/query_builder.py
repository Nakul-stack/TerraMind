from __future__ import annotations

import re
from typing import Optional

from .types import QueryProfile


_REGION_PATTERN = re.compile(r"\b(?:region|state|district|location)\s*[:=-]?\s*([A-Za-z\s]+)", re.IGNORECASE)
_GROWTH_PATTERN = re.compile(r"\b(?:growth\s*stage|stage)\s*[:=-]?\s*([A-Za-z\s]+)", re.IGNORECASE)


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()


def _extract_region(user_query: str) -> Optional[str]:
    m = _REGION_PATTERN.search(user_query or "")
    return _clean(m.group(1)) if m else None


def _extract_growth_stage(user_query: str) -> Optional[str]:
    m = _GROWTH_PATTERN.search(user_query or "")
    return _clean(m.group(1)) if m else None


def build_query_profile(user_query: str, parsed_intent) -> QueryProfile:
    base = _clean(user_query)
    crop = parsed_intent.crop.replace("_", " ") if getattr(parsed_intent, "crop", None) else None
    disease = parsed_intent.disease.replace("_", " ") if getattr(parsed_intent, "disease", None) else None
    pest = parsed_intent.pest.replace("_", " ") if getattr(parsed_intent, "pest", None) else None
    region = _extract_region(base)
    growth_stage = _extract_growth_stage(base)

    threat_term = disease or pest or "disease pest management"

    broad_query = _clean(base)
    phrase_query = f'"{_clean(" ".join(filter(None, [crop, threat_term, growth_stage, region])))}"'
    crop_query = _clean(" ".join(filter(None, [crop, growth_stage, "disease management", region])))
    threat_query = _clean(" ".join(filter(None, [crop, threat_term, growth_stage, "treatment", region])))
    region_query = _clean(" ".join(filter(None, [region, crop, threat_term, "agriculture"]))) if region else broad_query
    fallback_query = _clean(" ".join(filter(None, [crop, threat_term]))) or broad_query

    weather_terms = []
    lq = base.lower()
    if "humidity" in lq or "humid" in lq:
        weather_terms.append("high humidity")
    if "rain" in lq or "rainfall" in lq or "monsoon" in lq:
        weather_terms.append("rainfall")
    if "temp" in lq or "temperature" in lq:
        weather_terms.append("temperature")

    return QueryProfile(
        user_query=base,
        broad_query=broad_query,
        phrase_query=phrase_query,
        crop_query=crop_query,
        threat_query=threat_query,
        region_query=region_query,
        fallback_query=fallback_query,
        crop=crop,
        disease=disease,
        pest=pest,
        region=region,
        growth_stage=growth_stage,
        weather_terms=weather_terms,
    )
