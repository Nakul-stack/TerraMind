from __future__ import annotations

from typing import Dict, List

from .base import AdapterCapability, SourceAdapter
from ..types import QueryProfile


class CabiAdapter(SourceAdapter):
    source_name = "CABI"

    def capability(self) -> AdapterCapability:
        return AdapterCapability(
            source="CABI",
            access_type="search-page (often gated)",
            expected_result_type="bibliographic metadata",
            full_text_likely=False,
            metadata_only_likely=True,
            reliability="low",
            source_group="supporting_research",
            enrichment_only=True,
            notes="TODO: many CABI resources are subscription-protected; verify entitlement and API options.",
        )

    def build_requests(self, profile: QueryProfile) -> List[Dict]:
        return [
            {
                "url": "https://www.cabidigitallibrary.org/action/doSearch",
                "params": {"AllField": profile.broad_query},
                "query": profile.broad_query,
            }
        ]
