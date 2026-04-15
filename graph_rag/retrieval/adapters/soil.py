from __future__ import annotations

from typing import Dict, List

from .base import AdapterCapability, SourceAdapter
from .catalog_utils import run_catalog_search
from ..types import QueryProfile, SourceCallLog


_SOIL_CATALOG: List[Dict] = [
    {
        "title": "ISRIC SoilGrids",
        "abstract": "Global gridded soil information with properties such as pH, organic carbon, and texture for agronomic decisions.",
        "year": "2024",
        "url": "https://soilgrids.org/",
        "document_type": "dataset",
        "tags": ["soil", "ph", "organic carbon", "texture", "nutrient"],
    },
    {
        "title": "OpenLandMap Global Soil Properties",
        "abstract": "Open geospatial soil property layers used in precision agriculture and land suitability analysis.",
        "year": "2023",
        "url": "https://openlandmap.org/",
        "document_type": "dataset",
        "tags": ["soil fertility", "precision agriculture", "soil map"],
    },
    {
        "title": "Harmonized World Soil Database",
        "abstract": "Reference global soil database combining multiple regional and global soil information sources.",
        "year": "2023",
        "url": "https://www.iiasa.ac.at/web/home/research/researchPrograms/EcosystemsServicesandManagement/HWSD.html",
        "document_type": "dataset",
        "tags": ["soil type", "soil science", "land suitability"],
    },
    {
        "title": "FAO Global Soil Partnership Resources",
        "abstract": "Global soil health resources and datasets supporting sustainable soil management.",
        "year": "2024",
        "url": "https://www.fao.org/global-soil-partnership/en/",
        "document_type": "dataset",
        "tags": ["soil health", "sustainability", "fertility"],
    },
]


class SoilAdapter(SourceAdapter):
    source_name = "SoilData"

    def capability(self) -> AdapterCapability:
        return AdapterCapability(
            source="SoilData",
            access_type="curated soil dataset catalog",
            expected_result_type="dataset metadata",
            full_text_likely=False,
            metadata_only_likely=True,
            reliability="high",
            source_group="soil_dataset",
            enrichment_only=True,
            notes="Enrichment-only soil datasets for nutrient, pH, and soil-health graph augmentation.",
        )

    def build_requests(self, profile: QueryProfile) -> List[Dict]:
        return []

    def search(self, profile: QueryProfile) -> (List[Dict], List[SourceCallLog]):
        return run_catalog_search(
            source_name=self.source_name,
            source_url="https://soilgrids.org/",
            profile=profile,
            catalog_records=_SOIL_CATALOG,
            max_results=4,
        )
