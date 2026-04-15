from __future__ import annotations

import re
from typing import Dict, List, Tuple

from ..types import QueryProfile, SourceCallLog


def _tokenize(value: str) -> List[str]:
    return re.findall(r"[a-zA-Z]{3,}", (value or "").lower())


def run_catalog_search(
    source_name: str,
    source_url: str,
    profile: QueryProfile,
    catalog_records: List[Dict],
    max_results: int = 8,
) -> Tuple[List[Dict], List[SourceCallLog]]:
    query_text = profile.threat_query or profile.broad_query or profile.fallback_query
    tokens = set(
        _tokenize(
            " ".join(
                [
                    profile.broad_query,
                    profile.crop_query,
                    profile.threat_query,
                    profile.region_query,
                    profile.fallback_query,
                ]
            )
        )
    )

    scored: List[Tuple[int, Dict]] = []
    for rec in catalog_records:
        hay = " ".join(
            [
                str(rec.get("title") or ""),
                str(rec.get("abstract") or ""),
                str(rec.get("snippet") or ""),
                str(rec.get("year") or ""),
                " ".join([str(x) for x in rec.get("tags", [])]),
            ]
        ).lower()

        overlap = sum(1 for t in tokens if t in hay)
        if overlap > 0:
            scored.append((overlap, rec))

    scored.sort(key=lambda x: x[0], reverse=True)
    records = [x[1] for x in scored[:max_results]]

    call = SourceCallLog(
        source=source_name,
        query=query_text,
        url=source_url,
        method="CATALOG",
        payload={"mode": "curated_enrichment_catalog"},
        status_code=200,
        response_type="catalog",
        parsed_item_count=len(records),
        preview_items=records[:3],
    )

    return records, [call]
