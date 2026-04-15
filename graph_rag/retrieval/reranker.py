from __future__ import annotations

import datetime as _dt
import re
from typing import List

from .types import NormalizedDocument, QueryProfile


def _tokenize(value: str) -> List[str]:
    return re.findall(r"[a-zA-Z]{3,}", (value or "").lower())


def score_documents(profile: QueryProfile, docs: List[NormalizedDocument]) -> List[NormalizedDocument]:
    query_tokens = set(_tokenize(" ".join([
        profile.broad_query,
        profile.crop_query,
        profile.threat_query,
        profile.region_query,
    ])))

    current_year = _dt.datetime.utcnow().year

    for doc in docs:
        hay = " ".join([doc.title, doc.abstract, doc.snippet, doc.raw_text]).lower()
        hay_tokens = set(_tokenize(hay))

        overlap = len(query_tokens & hay_tokens)
        score = min(0.55, overlap * 0.04)

        if profile.crop and profile.crop.lower() in hay:
            score += 0.12
        if profile.disease and profile.disease.lower().replace("_", " ") in hay:
            score += 0.12
        if profile.pest and profile.pest.lower().replace("_", " ") in hay:
            score += 0.10
        if profile.growth_stage and profile.growth_stage.lower() in hay:
            score += 0.07
        if profile.region and profile.region.lower() in hay:
            score += 0.06

        for wt in profile.weather_terms:
            if wt.lower() in hay:
                score += 0.03

        if doc.evidence_quality == "high":
            score += 0.08
        elif doc.evidence_quality == "medium":
            score += 0.04

        if doc.year.isdigit():
            y = int(doc.year)
            if y >= current_year - 3:
                score += 0.05
            elif y >= current_year - 8:
                score += 0.02
            elif y < 2015:
                score -= 0.03

        source_lc = doc.source.lower()
        if source_lc == "agris":
            # AGRIS remains the primary evidence source.
            score += 0.08
        elif source_lc in {
            "agricola",
            "pubag",
            "cabi",
            "agecon",
            "asabe",
            "faostat",
            "cgiar",
            "climatedata",
            "soildata",
        }:
            score += 0.03

        doc.retrieval_confidence = max(0.0, min(1.0, score))

    docs.sort(key=lambda d: d.retrieval_confidence, reverse=True)

    filtered: List[NormalizedDocument] = []
    for doc in docs:
        if doc.retrieval_confidence < 0.12:
            continue
        filtered.append(doc)

    return filtered
