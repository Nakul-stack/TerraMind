from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class QueryProfile:
    user_query: str
    broad_query: str
    phrase_query: str
    crop_query: str
    threat_query: str
    region_query: str
    fallback_query: str
    crop: Optional[str] = None
    disease: Optional[str] = None
    pest: Optional[str] = None
    region: Optional[str] = None
    growth_stage: Optional[str] = None
    weather_terms: List[str] = field(default_factory=list)


@dataclass
class NormalizedDocument:
    source: str = ""
    source_group: str = "research"
    enrichment_only: bool = False
    title: str = ""
    abstract: str = ""
    snippet: str = ""
    authors: List[str] = field(default_factory=list)
    year: str = ""
    doi: str = ""
    url: str = ""
    pdf_url: str = ""
    document_type: str = "metadata"
    crop_tags: List[str] = field(default_factory=list)
    disease_tags: List[str] = field(default_factory=list)
    pest_tags: List[str] = field(default_factory=list)
    region_tags: List[str] = field(default_factory=list)
    evidence_quality: str = "low"
    raw_text: str = ""
    retrieval_confidence: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "source_group": self.source_group,
            "enrichment_only": self.enrichment_only,
            "title": self.title,
            "abstract": self.abstract,
            "snippet": self.snippet,
            "authors": self.authors,
            "year": self.year,
            "doi": self.doi,
            "url": self.url,
            "pdf_url": self.pdf_url,
            "document_type": self.document_type,
            "crop_tags": self.crop_tags,
            "disease_tags": self.disease_tags,
            "pest_tags": self.pest_tags,
            "region_tags": self.region_tags,
            "evidence_quality": self.evidence_quality,
            "raw_text": self.raw_text,
            "retrieval_confidence": round(float(self.retrieval_confidence), 4),
        }


@dataclass
class SourceCallLog:
    source: str
    query: str
    url: str
    method: str = "GET"
    payload: Dict[str, Any] = field(default_factory=dict)
    status_code: Optional[int] = None
    response_type: str = "unknown"
    raw_sample: str = ""
    parsed_item_count: int = 0
    normalized_item_count: int = 0
    parser_errors: List[str] = field(default_factory=list)
    timeout_error: bool = False
    blocked_error: bool = False
    other_error: str = ""
    preview_items: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "query": self.query,
            "url": self.url,
            "method": self.method,
            "payload": self.payload,
            "status_code": self.status_code,
            "response_type": self.response_type,
            "raw_sample": self.raw_sample,
            "parsed_item_count": self.parsed_item_count,
            "normalized_item_count": self.normalized_item_count,
            "parser_errors": self.parser_errors,
            "timeout_error": self.timeout_error,
            "blocked_error": self.blocked_error,
            "other_error": self.other_error,
            "preview_items": self.preview_items,
        }


@dataclass
class RetrievalResult:
    query_profile: QueryProfile
    documents: List[NormalizedDocument]
    source_counts: Dict[str, int]
    source_logs: List[SourceCallLog]

    @property
    def total_docs(self) -> int:
        return len(self.documents)

    @property
    def metadata_only(self) -> bool:
        return bool(self.documents) and all(d.evidence_quality in {"low", "metadata_only"} for d in self.documents)
