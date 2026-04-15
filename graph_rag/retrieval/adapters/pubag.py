from __future__ import annotations

from typing import Dict, List

from .base import AdapterCapability, SourceAdapter
from ..types import QueryProfile


class PubAgAdapter(SourceAdapter):
    source_name = "PubAg"

    def capability(self) -> AdapterCapability:
        return AdapterCapability(
            source="PubAg",
            access_type="search-page",
            expected_result_type="metadata/result cards",
            full_text_likely=False,
            metadata_only_likely=True,
            reliability="low-medium",
            source_group="supporting_research",
            enrichment_only=True,
            notes="TODO: confirm query params/selectors and legal scraping constraints.",
        )

    def build_requests(self, profile: QueryProfile) -> List[Dict]:
        return [
            {
                "url": "https://pubag.nal.usda.gov/",
                "params": {"q": profile.threat_query or profile.broad_query},
                "query": profile.threat_query or profile.broad_query,
            }
        ]
