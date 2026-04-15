import logging
from typing import Optional

from app.core.config import RETRIEVAL_TOP_K, MAX_CONTEXT_CHUNKS
from app.chatbot.ingestion.embedder import embed_query
from app.chatbot import document_registry
from app.chatbot.router.types import RetrieverContext

logger = logging.getLogger(__name__)

class FAISSRetriever:
    """Retriever wrapper for the existing PDF/FAISS infrastructure."""
    
    async def retrieve(self, query: str, diag_env: Optional[dict] = None, top_k: Optional[int] = None) -> RetrieverContext:
        try:
            document_registry.ensure_loaded()
        except FileNotFoundError as exc:
            logger.error("Vector store not found: %s", exc)
            return RetrieverContext(
                text_content="The document index has not been built yet.",
                metadata={"error": "index_missing", "sources": []}
            )

        diag_env = diag_env or {}
        identified_crop = diag_env.get("identified_crop")
        identified_class = diag_env.get("identified_class")

        enriched_query = query
        if identified_crop:
            enriched_query = f"{identified_crop} {enriched_query}"
        if identified_class:
            label = identified_class.replace("___", " ").replace("__", " ").replace("_", " ")
            enriched_query = f"{label} {enriched_query}"

        query_vec = embed_query(enriched_query)
        k = top_k or RETRIEVAL_TOP_K
        results = document_registry.search(query_vec, top_k=k)
        
        limit = top_k or MAX_CONTEXT_CHUNKS
        context_chunks = results[:limit]
        
        text_content = "\n\n".join(c["text"] for c in context_chunks)
        
        sources = [
            {
                "file_name": c["file_name"],
                "page": c["page"],
                "snippet": c["text"][:200] + ("…" if len(c["text"]) > 200 else ""),
                "score": round(c["score"], 4),
            }
            for c in context_chunks
        ]
        
        return RetrieverContext(
            text_content=text_content,
            metadata={"sources": sources, "raw_chunks": context_chunks}
        )
