from __future__ import annotations

from typing import Dict, List

from .base import AdapterCapability, SourceAdapter
from .catalog_utils import run_catalog_search
from ..types import QueryProfile, SourceCallLog


_FAOSTAT_CATALOG: List[Dict] = [
    {
        "title": "FAOSTAT Crops and Livestock Products Database",
        "abstract": "Global agricultural production, harvested area, and yield indicators for crops and livestock products.",
        "year": "2024",
        "url": "https://www.fao.org/faostat/en/#data/QCL",
        "document_type": "dataset",
        "tags": ["crop", "yield", "production", "agriculture", "indicator"],
    },
    {
        "title": "FAOSTAT Fertilizers by Nutrient",
        "abstract": "Country-level fertilizer nutrient statistics useful for nutrient management and agronomic planning.",
        "year": "2024",
        "url": "https://www.fao.org/faostat/en/#data/RFN",
        "document_type": "dataset",
        "tags": ["fertilizer", "nutrient", "soil", "crop nutrition"],
    },
    {
        "title": "FAOSTAT Pesticides Use",
        "abstract": "Pesticide use indicators for agricultural systems, supporting crop protection and risk analysis.",
        "year": "2024",
        "url": "https://www.fao.org/faostat/en/#data/RP",
        "document_type": "dataset",
        "tags": ["pesticide", "plant disease", "pest", "risk"],
    },
    {
        "title": "AQUASTAT Global Water and Irrigation Information",
        "abstract": "FAO water and irrigation indicators for agricultural water management and planning.",
        "year": "2024",
        "url": "https://www.fao.org/aquastat/en/",
        "document_type": "dataset",
        "tags": ["irrigation", "water", "climate", "agriculture"],
    },
    {
        "title": "FAO WaPOR Water Productivity through Open Access of Remotely Sensed Data",
        "abstract": "Remote sensing based evapotranspiration and water productivity metrics for crop monitoring.",
        "year": "2023",
        "url": "https://www.fao.org/in-action/remote-sensing-for-water-productivity/wapor/en/",
        "document_type": "dataset",
        "tags": ["remote sensing", "irrigation", "crop monitoring", "water productivity"],
    },
]


class FaostatAdapter(SourceAdapter):
    source_name = "FAOSTAT"

    def capability(self) -> AdapterCapability:
        return AdapterCapability(
            source="FAOSTAT",
            access_type="curated FAO dataset catalog",
            expected_result_type="dataset metadata",
            full_text_likely=False,
            metadata_only_likely=True,
            reliability="high",
            source_group="fao_dataset",
            enrichment_only=True,
            notes="Enrichment-only FAO datasets used to strengthen graph relationships around irrigation, fertilizer, and yield indicators.",
        )

    def build_requests(self, profile: QueryProfile) -> List[Dict]:
        return []

    def search(self, profile: QueryProfile) -> (List[Dict], List[SourceCallLog]):
        return run_catalog_search(
            source_name=self.source_name,
            source_url="https://www.fao.org/faostat/en/",
            profile=profile,
            catalog_records=_FAOSTAT_CATALOG,
            max_results=5,
        )
