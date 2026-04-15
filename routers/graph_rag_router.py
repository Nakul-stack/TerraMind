from threading import Lock
from typing import Any, Dict

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from graph_rag.graph_rag_pipeline import GraphRAGPipeline


router = APIRouter()


class GraphRAGRequest(BaseModel):
    query: str = Field(..., min_length=3, description="Farmer query in natural language")
    use_llm: bool = Field(True, description="If false, return deterministic graph-only response")


class GraphRAGResponse(BaseModel):
    query: str
    parsed_intent: Dict[str, Any]
    context: Dict[str, Any]
    kg_context_text: str
    response: str
    engine: Dict[str, Any]


_pipeline = None
_pipeline_lock = Lock()


def get_pipeline() -> GraphRAGPipeline:
    global _pipeline
    if _pipeline is None:
        with _pipeline_lock:
            if _pipeline is None:
                _pipeline = GraphRAGPipeline()
    return _pipeline


@router.get("/health")
def graph_rag_health() -> Dict[str, Any]:
    pipeline = get_pipeline()
    return {
        "status": "ok",
        "engine": "graph_rag",
        "model": pipeline.ollama_model,
        "ollama_base_url": pipeline.ollama_base_url,
    }


@router.post("/query", response_model=GraphRAGResponse)
def query_graph_rag(payload: GraphRAGRequest):
    try:
        pipeline = get_pipeline()
        result = pipeline.run(payload.query, use_llm=payload.use_llm)
        return result
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"GraphRAG processing failed: {exc}") from exc
