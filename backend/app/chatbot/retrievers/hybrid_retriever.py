import asyncio
import logging
from typing import Optional

from app.chatbot.router.types import RetrieverContext
from app.chatbot.retrievers.base import BaseRetriever

logger = logging.getLogger(__name__)

class HybridRetriever:
    """Orchestrates async fetching from both PDF and Graph retrievers."""
    
    def __init__(self, pdf_retriever: BaseRetriever, graph_retriever: BaseRetriever):
        self.pdf_retriever = pdf_retriever
        self.graph_retriever = graph_retriever
        
    async def retrieve(self, query: str, diag_env: Optional[dict] = None, top_k: Optional[int] = None) -> RetrieverContext:
        # Run both retrievals concurrently to minimize latency
        pdf_task = asyncio.create_task(self.pdf_retriever.retrieve(query, diag_env, top_k))
        graph_task = asyncio.create_task(self.graph_retriever.retrieve(query, diag_env, top_k))
        
        pdf_ctx, graph_ctx = await asyncio.gather(pdf_task, graph_task)
        
        # Combine text content
        combined_text = []
        if graph_ctx.text_content.strip():
            combined_text.append(graph_ctx.text_content)
        if pdf_ctx.text_content.strip():
            combined_text.append("DOCUMENT CONTEXT:\n" + pdf_ctx.text_content)
            
        final_text = "\n\n".join(combined_text)
        
        # Combine sources
        pdf_sources = pdf_ctx.metadata.get("sources", [])
        graph_sources = graph_ctx.metadata.get("sources", [])
        
        # Only attach graph source if it actually found something useful
        sources = graph_sources + pdf_sources
        
        metadata = {
            "pdf_results_count": len(pdf_ctx.metadata.get("raw_chunks", [])),
            "graph_nodes_found": graph_ctx.metadata.get("nodes_found", 0),
            "sources": sources
        }
        
        return RetrieverContext(text_content=final_text, metadata=metadata)
