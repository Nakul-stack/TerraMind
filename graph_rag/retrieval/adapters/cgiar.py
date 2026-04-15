from __future__ import annotations

from typing import Dict, List

from .base import AdapterCapability, SourceAdapter
from .catalog_utils import run_catalog_search
from ..types import QueryProfile, SourceCallLog


_CGIAR_CATALOG: List[Dict] = [
    {
        "title": "CGIAR GARDIAN Open Knowledge Platform",
        "abstract": "CGIAR knowledge discovery platform for agricultural research outputs, projects, and linked datasets.",
        "year": "2024",
        "url": "https://gardian.cgiar.org/",
        "document_type": "dataset",
        "tags": ["cgiar", "research", "crop", "agriculture"],
    },
    {
        "title": "CIMMYT Data Repository",
        "abstract": "Open datasets related to wheat, maize, agronomy, and climate-resilient cropping systems.",
        "year": "2024",
        "url": "https://data.cimmyt.org/",
        "document_type": "dataset",
        "tags": ["wheat", "maize", "crop yield", "agronomy"],
    },
    {
        "title": "ICRISAT Data Management and Open Access Resources",
        "abstract": "Research data and indicators covering dryland agriculture, climate risks, and crop performance.",
        "year": "2023",
        "url": "https://www.icrisat.org/",
        "document_type": "dataset",
        "tags": ["dryland", "climate", "sorghum", "millet", "yield"],
    },
    {
        "title": "IFPRI Data and Tools",
        "abstract": "Agricultural economics, food systems, and policy datasets to support evidence-based decisions.",
        "year": "2024",
        "url": "https://www.ifpri.org/data",
        "document_type": "dataset",
        "tags": ["economics", "food system", "supply chain", "risk"],
    },
]


class CgiarAdapter(SourceAdapter):
    source_name = "CGIAR"

    def capability(self) -> AdapterCapability:
        return AdapterCapability(
            source="CGIAR",
            access_type="curated CGIAR dataset catalog",
            expected_result_type="dataset metadata",
            full_text_likely=False,
            metadata_only_likely=True,
            reliability="high",
            source_group="cgiar_dataset",
            enrichment_only=True,
            notes="Enrichment-only CGIAR datasets for cross-source validation and graph connectivity.",
        )

    def build_requests(self, profile: QueryProfile) -> List[Dict]:
        return []

    def search(self, profile: QueryProfile) -> (List[Dict], List[SourceCallLog]):
        return run_catalog_search(
            source_name=self.source_name,
            source_url="https://gardian.cgiar.org/",
            profile=profile,
            catalog_records=_CGIAR_CATALOG,
            max_results=4,
        )
