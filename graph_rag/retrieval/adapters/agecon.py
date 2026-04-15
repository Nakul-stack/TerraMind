from __future__ import annotations

from typing import Dict, List

from .base import AdapterCapability, SourceAdapter
from ..types import QueryProfile


class AgEconAdapter(SourceAdapter):
    source_name = "AgEcon"

    def capability(self) -> AdapterCapability:
        return AdapterCapability(
            source="AgEcon",
            access_type="search-page",
            expected_result_type="metadata/full text links",
            full_text_likely=True,
            metadata_only_likely=False,
            reliability="medium",
            source_group="supporting_research",
            enrichment_only=True,
            notes="TODO: validate stable selectors and PDF link extraction.",
        )

    def build_requests(self, profile: QueryProfile) -> List[Dict]:
        return [
            {
                "url": "https://ageconsearch.umn.edu/search",
                "params": {"q": profile.region_query or profile.broad_query},
                "query": profile.region_query or profile.broad_query,
            }
        ]
