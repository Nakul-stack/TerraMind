from __future__ import annotations

import re
from typing import Any, Dict, List

from .types import NormalizedDocument, QueryProfile


def _to_list(value: Any) -> List[str]:
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
    if value:
        return [str(value).strip()]
    return []


def _extract_year(record: Dict[str, Any]) -> str:
    for key in ["year", "publication_year", "pub_year", "date", "issued"]:
        val = record.get(key)
        if not val:
            continue
        m = re.search(r"(19|20)\d{2}", str(val))
        if m:
            return m.group(0)
    return ""


def _extract_doi(record: Dict[str, Any]) -> str:
    for key in ["doi", "DOI", "identifier", "id"]:
        val = record.get(key)
        if not val:
            continue
        s = str(val)
        m = re.search(r"10\.\d{4,9}/[-._;()/:A-Za-z0-9]+", s)
        if m:
            return m.group(0)
    return ""


def normalize_record(
    source: str,
    record: Dict[str, Any],
    profile: QueryProfile,
    source_group: str = "research",
    enrichment_only: bool = False,
) -> NormalizedDocument:
    title = str(record.get("title") or record.get("dcTitle") or "").strip()
    abstract = str(record.get("abstract") or record.get("description") or record.get("dcDescription") or "").strip()
    snippet = str(record.get("snippet") or abstract[:350]).strip()

    raw_text = "\n".join([title, abstract, snippet]).strip()

    evidence_quality = "metadata_only"
    if len(abstract) > 350 or len(raw_text) > 700:
        evidence_quality = "medium"
    if len(raw_text) > 1400:
        evidence_quality = "high"

    doc = NormalizedDocument(
        source=source,
        source_group=source_group,
        enrichment_only=bool(enrichment_only),
        title=title,
        abstract=abstract,
        snippet=snippet,
        authors=_to_list(record.get("authors") or record.get("author")),
        year=_extract_year(record),
        doi=_extract_doi(record),
        url=str(record.get("url") or record.get("link") or "").strip(),
        pdf_url=str(record.get("pdf_url") or record.get("pdf") or "").strip(),
        document_type=str(record.get("document_type") or "metadata").strip() or "metadata",
        crop_tags=[profile.crop] if profile.crop else [],
        disease_tags=[profile.disease] if profile.disease else [],
        pest_tags=[profile.pest] if profile.pest else [],
        region_tags=[profile.region] if profile.region else [],
        evidence_quality=evidence_quality,
        raw_text=raw_text,
        retrieval_confidence=0.0,
    )

    return doc


def normalize_records(
    source: str,
    records: List[Dict[str, Any]],
    profile: QueryProfile,
    source_group: str = "research",
    enrichment_only: bool = False,
) -> List[NormalizedDocument]:
    docs: List[NormalizedDocument] = []
    seen = set()
    for record in records:
        try:
            doc = normalize_record(
                source,
                record,
                profile,
                source_group=source_group,
                enrichment_only=enrichment_only,
            )
        except Exception:
            continue
        key = (doc.source.lower(), doc.title.lower(), doc.url.lower())
        if key in seen:
            continue
        seen.add(key)
        if len(doc.title) < 6 and len(doc.abstract) < 30:
            continue
        docs.append(doc)
    return docs
