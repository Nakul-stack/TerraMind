from __future__ import annotations

from typing import Dict, List

from .base import AdapterCapability, SourceAdapter
from ..types import QueryProfile


class AgricolaAdapter(SourceAdapter):
    source_name = "AGRICOLA"

    def capability(self) -> AdapterCapability:
        return AdapterCapability(
            source="AGRICOLA",
            access_type="API/search-page",
            expected_result_type="bibliographic metadata",
            full_text_likely=False,
            metadata_only_likely=True,
            reliability="medium",
            source_group="supporting_research",
            enrichment_only=True,
            notes="TODO: pin stable USDA NAL endpoint/schema version for production.",
        )

    def build_requests(self, profile: QueryProfile) -> List[Dict]:
        return [
            {
                "url": "https://catalog.nal.usda.gov/api/v1/",
                "params": {"query": profile.crop_query or profile.broad_query, "format": "json"},
                "query": profile.crop_query or profile.broad_query,
            },
            {
                "url": "https://catalog.nal.usda.gov/api/v1/",
                "params": {"query": profile.fallback_query, "format": "json"},
                "query": profile.fallback_query,
            },
        ]
