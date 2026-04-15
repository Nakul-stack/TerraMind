from __future__ import annotations

from typing import Dict, List

from .base import AdapterCapability, SourceAdapter
from .catalog_utils import run_catalog_search
from ..types import QueryProfile, SourceCallLog


_CLIMATE_CATALOG: List[Dict] = [
    {
        "title": "NASA POWER Agroclimatology Data Access",
        "abstract": "Daily and monthly climate variables for agriculture, including temperature, humidity, rainfall, and solar radiation.",
        "year": "2024",
        "url": "https://power.larc.nasa.gov/",
        "document_type": "dataset",
        "tags": ["climate", "temperature", "humidity", "rainfall", "solar"],
    },
    {
        "title": "CHIRPS Rainfall Dataset",
        "abstract": "Quasi-global rainfall time series commonly used for drought, crop risk, and rainfall anomaly analysis.",
        "year": "2024",
        "url": "https://www.chc.ucsb.edu/data/chirps",
        "document_type": "dataset",
        "tags": ["rainfall", "drought", "climate risk", "crop"],
    },
    {
        "title": "ERA5 Reanalysis (Copernicus Climate Data Store)",
        "abstract": "Global atmospheric reanalysis useful for climate impact modeling in agriculture and yield forecasting.",
        "year": "2024",
        "url": "https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels",
        "document_type": "dataset",
        "tags": ["climate change", "temperature", "wind", "moisture", "yield"],
    },
    {
        "title": "WorldClim Version 2.1 Climate Surfaces",
        "abstract": "High-resolution global climate layers for crop suitability and agro-ecological modeling.",
        "year": "2023",
        "url": "https://www.worldclim.org/data/worldclim21.html",
        "document_type": "dataset",
        "tags": ["crop suitability", "agroecology", "climate layers"],
    },
]


class ClimateAdapter(SourceAdapter):
    source_name = "ClimateData"

    def capability(self) -> AdapterCapability:
        return AdapterCapability(
            source="ClimateData",
            access_type="curated climate dataset catalog",
            expected_result_type="dataset metadata",
            full_text_likely=False,
            metadata_only_likely=True,
            reliability="high",
            source_group="climate_dataset",
            enrichment_only=True,
            notes="Enrichment-only climate datasets used to strengthen weather and climate-to-yield graph edges.",
        )

    def build_requests(self, profile: QueryProfile) -> List[Dict]:
        return []

    def search(self, profile: QueryProfile) -> (List[Dict], List[SourceCallLog]):
        return run_catalog_search(
            source_name=self.source_name,
            source_url="https://power.larc.nasa.gov/",
            profile=profile,
            catalog_records=_CLIMATE_CATALOG,
            max_results=4,
        )
