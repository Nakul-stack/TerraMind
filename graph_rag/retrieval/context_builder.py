from __future__ import annotations

from typing import List

from .types import NormalizedDocument


def build_external_context(docs: List[NormalizedDocument], max_docs: int = 8, max_chars: int = 800) -> str:
    if not docs:
        return ""

    lines = ["EXTERNAL EVIDENCE SUMMARY:"]
    for idx, doc in enumerate(docs[:max_docs], start=1):
        title = (doc.title or "Untitled")[:180]
        snippet = (doc.abstract or doc.snippet or doc.raw_text or "")
        snippet = snippet.replace("\n", " ").strip()[:max_chars]
        url = doc.url or doc.pdf_url or ""
        tier = "enrichment" if doc.enrichment_only else "primary"
        source_group = doc.source_group or "research"

        lines.append(
            f"[{idx}] Tier={tier} | Group={source_group} | Source={doc.source} | Confidence={doc.retrieval_confidence:.2f} | Year={doc.year or 'n/a'} | Quality={doc.evidence_quality}"
        )
        lines.append(f"Title: {title}")
        if snippet:
            lines.append(f"Evidence: {snippet}")
        if url:
            lines.append(f"URL: {url}")
        lines.append("")

    return "\n".join(lines).strip()
