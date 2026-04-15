from __future__ import annotations

import json
from typing import List

from graph_rag.graph_builder import AgroKGBuilder
from graph_rag.intent_parser import IntentParser
from graph_rag.query_engine import GraphQueryEngine
from graph_rag.retrieval.orchestrator import ExternalRetrievalOrchestrator


TEST_QUERIES: List[str] = [
    "potato late blight humid conditions fungicide evidence",
    "tomato late blight metalaxyl resistance management",
    "wheat flowering Punjab humidity rust aphids fungicide",
    "rice blast high humidity management",
    "maize stem borer warm humid conditions",
]


def run_tests() -> None:
    kg_builder = AgroKGBuilder()
    kg_builder.build()
    parser = IntentParser(kg_builder)
    query_engine = GraphQueryEngine(kg_builder)
    orchestrator = ExternalRetrievalOrchestrator()

    for q in TEST_QUERIES:
        parsed = parser.parse(q)
        ctx = query_engine.query(
            crop_name=parsed.crop,
            pest_name=parsed.pest,
            disease_name=parsed.disease,
            climate_conditions=parsed.climate_conditions,
            soil_type=parsed.soil_type,
            pesticide_name=parsed.pesticide,
        )
        has_local_kb_context = bool(ctx.pests_found or ctx.diseases_found or ctx.treatments)

        output = orchestrator.run(q, parsed, has_local_kb_context=has_local_kb_context)

        result = {
            "query": q,
            "raw_retrieval_counts_per_source": output.retrieval.source_counts,
            "normalized_total_count": output.retrieval.total_docs,
            "final_context_count": min(8, output.retrieval.total_docs),
            "answer_allowed": output.grounding.allow_generation,
            "grounding_message": output.grounding.message,
            "metadata_only": output.retrieval.metadata_only,
        }
        print(json.dumps(result, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    run_tests()
