from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional

from .adapters import default_adapters
from .context_builder import build_external_context
from .grounding_policy import GroundingDecision, evaluate_grounding
from .normalizer import normalize_records
from .query_builder import build_query_profile
from .reranker import score_documents
from .structured_logger import RetrievalStructuredLogger
from .types import RetrievalResult, SourceCallLog


@dataclass
class RetrievalPipelineOutput:
    retrieval: RetrievalResult
    context_text: str
    grounding: GroundingDecision
    capability_matrix: List[Dict]


class ExternalRetrievalOrchestrator:
    """Model-agnostic external retrieval pipeline for Graph RAG grounding."""

    def __init__(self):
        self.adapters = default_adapters()
        self.logger = RetrievalStructuredLogger()

    def capability_matrix(self) -> List[Dict]:
        return [a.capability().__dict__ for a in self.adapters]

    def run(self, user_query: str, parsed_intent, has_local_kb_context: bool) -> RetrievalPipelineOutput:
        profile = build_query_profile(user_query, parsed_intent)

        all_docs = []
        source_counts: Dict[str, int] = defaultdict(int)
        all_logs: List[SourceCallLog] = []
        capability_by_source: Dict[str, Dict] = {}

        for adapter in self.adapters:
            capability = adapter.capability()
            capability_by_source[adapter.source_name] = capability.__dict__

            raw_records, logs = adapter.search(profile)
            normalized_docs = normalize_records(
                adapter.source_name,
                raw_records,
                profile,
                source_group=capability.source_group,
                enrichment_only=capability.enrichment_only,
            )
            for call in logs:
                call.normalized_item_count = len(normalized_docs)
                self.logger.log_call(call)
            all_logs.extend(logs)

            source_counts[adapter.source_name] += len(normalized_docs)
            all_docs.extend(normalized_docs)

        ranked_docs = score_documents(profile, all_docs)
        final_docs = self._apply_source_group_policy(ranked_docs)

        retrieval = RetrievalResult(
            query_profile=profile,
            documents=final_docs,
            source_counts=dict(source_counts),
            source_logs=all_logs,
        )
        self.logger.log_summary(profile.user_query, retrieval.source_counts, all_logs)

        context_text = build_external_context(final_docs)
        grounding = evaluate_grounding(
            total_external_docs=retrieval.total_docs,
            metadata_only=retrieval.metadata_only,
            has_local_kb_context=has_local_kb_context,
        )

        return RetrievalPipelineOutput(
            retrieval=retrieval,
            context_text=context_text,
            grounding=grounding,
            capability_matrix=list(capability_by_source.values()),
        )

    def _apply_source_group_policy(self, ranked_docs):
        primary_docs = [d for d in ranked_docs if not getattr(d, "enrichment_only", False)]
        enrichment_docs = [d for d in ranked_docs if getattr(d, "enrichment_only", False)]

        agris_primary = [d for d in primary_docs if (d.source or "").lower() == "agris"]
        other_primary = [d for d in primary_docs if (d.source or "").lower() != "agris"]

        # AGRIS is the primary evidence tier by design.
        if agris_primary:
            selected_primary = agris_primary + other_primary[:2]
            enrichment_cap = min(4, max(2, len(selected_primary) // 2))
            return selected_primary + enrichment_docs[:enrichment_cap]

        if primary_docs:
            enrichment_cap = min(4, max(2, len(primary_docs) // 2))
            return primary_docs + enrichment_docs[:enrichment_cap]

        # If primary evidence is unavailable, return a small enrichment fallback set.
        return enrichment_docs[:4]
