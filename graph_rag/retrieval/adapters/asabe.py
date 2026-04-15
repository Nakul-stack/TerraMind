from __future__ import annotations

from typing import Dict, List

from .base import AdapterCapability, SourceAdapter
from ..types import QueryProfile


class AsabeAdapter(SourceAdapter):
    source_name = "ASABE"

    def capability(self) -> AdapterCapability:
        return AdapterCapability(
            source="ASABE",
            access_type="search-page (publisher platform)",
            expected_result_type="metadata/abstract",
            full_text_likely=False,
            metadata_only_likely=True,
            reliability="low-medium",
            source_group="supporting_research",
            enrichment_only=True,
            notes="TODO: verify search endpoint and anti-bot controls on ASABE library site.",
        )

    def build_requests(self, profile: QueryProfile) -> List[Dict]:
        return [
            {
                "url": "https://elibrary.asabe.org/search",
                "params": {"q": profile.threat_query or profile.fallback_query},
                "query": profile.threat_query or profile.fallback_query,
            }
        ]
