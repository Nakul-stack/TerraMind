import os
import requests
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict

from backend.core.logging_config import log
from graph_rag.graph_rag_pipeline import GraphRAGPipeline

router = APIRouter()

_pipeline = None

def get_pipeline() -> GraphRAGPipeline:
    global _pipeline
    if _pipeline is None:
        log.info("Initializing GraphRAGPipeline...")
        _pipeline = GraphRAGPipeline()
    return _pipeline

class QueryRequest(BaseModel):
    query: str
    use_llm: bool = True
    model_config = ConfigDict(protected_namespaces=())

@router.post("/query")
def query_graph_rag(req: QueryRequest):
    try:
        pipeline = get_pipeline()
        result = pipeline.run(req.query, use_llm=req.use_llm)
        return result
    except Exception as e:
        log.exception("Graph RAG query failed")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
def graph_rag_health():
    try:
        pipeline = get_pipeline()
        status = "ok"
        model = pipeline.ollama_model
        
        try:
            url = f"{pipeline.ollama_base_url.rstrip('/')}/api/tags"
            r = requests.get(url, timeout=5)
            if r.status_code == 200:
                tags = r.json().get("models", [])
                # Check if model exists locally
                if not any(model in t.get("name", "") for t in tags):
                    status = "degraded (model not found)"
            else:
                status = "degraded (ollama error)"
        except Exception:
            status = "degraded (ollama config/connection error)"
            
        return {
            "status": status,
            "model": model,
            "kg_nodes": len(pipeline.kg_builder.G.nodes) if pipeline.kg_builder.G else 0
        }
    except Exception as e:
        log.exception("Graph RAG health check failed")
        return {"status": "error", "detail": str(e)}
