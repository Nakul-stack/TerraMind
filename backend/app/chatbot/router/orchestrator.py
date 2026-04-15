import asyncio
import logging
import json
from datetime import datetime

from app.chatbot.router.types import RetrievalStrategy
from app.chatbot.router.intent_classifier import classify_intent
from app.chatbot.retrievers.pdf_retriever import FAISSRetriever
from app.chatbot.retrievers.graph_retriever import GraphRetriever
from app.chatbot.retrievers.hybrid_retriever import HybridRetriever
from app.chatbot.generators.answer_composer import compose_answer

logger = logging.getLogger(__name__)

class IntelligentOrchestrator:
    def __init__(self):
        self.pdf_retriever = FAISSRetriever()
        self.graph_retriever = GraphRetriever()
        self.hybrid_retriever = HybridRetriever(self.pdf_retriever, self.graph_retriever)

    async def ask(self, user_query: str, diag_env: dict = None, top_k: int = None) -> dict:
        diag_env = diag_env or {}
        
        # Step 1: Detect Intent
        routing_decision = await classify_intent(user_query)
        strategy = routing_decision.strategy
        
        logger.info(f"Router selected {strategy.value} (confidence: {routing_decision.confidence:.2f})")
        
        # Step 2: Retrieve Context
        if strategy == RetrievalStrategy.GRAPH_RAG:
            context = await self.graph_retriever.retrieve(user_query, diag_env, top_k)
            # Fallback if graph is empty
            if context.metadata.get("nodes_found", 0) == 0:
                logger.warning("Graph RAG returned 0 nodes, falling back to PDF RAG.")
                strategy = RetrievalStrategy.PDF_RAG
                context = await self.pdf_retriever.retrieve(user_query, diag_env, top_k)
        elif strategy == RetrievalStrategy.PDF_RAG:
            context = await self.pdf_retriever.retrieve(user_query, diag_env, top_k)
        else: # HYBRID_RAG
            context = await self.hybrid_retriever.retrieve(user_query, diag_env, top_k)
            
        # Step 3: Compose Natural Answer
        answer = await compose_answer(user_query, context, routing_decision)
        
        # Log decision metric
        self._log_decision(user_query, routing_decision, strategy, context.metadata)
        
        return {
            "answer": answer,
            "strategy": strategy.value,
            "sources": context.metadata.get("sources", []),
            "confidence": routing_decision.confidence
        }
        
    def _log_decision(self, query: str, decision, final_strategy: RetrievalStrategy, context_meta: dict):
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "query": query,
            "decision_strategy": decision.strategy.value,
            "final_strategy": final_strategy.value,
            "confidence": decision.confidence,
            "reason": decision.reason,
            "nodes_found": context_meta.get("nodes_found"),
            "pdf_results_count": context_meta.get("pdf_results_count") 
               if "pdf_results_count" in context_meta else len(context_meta.get("raw_chunks", []))
        }
        # In a real system, you would append this to an ELK stack or DB.
        # Here we try to log safely.
        try:
            with open("error.log", "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry) + "\n")
        except Exception:
            pass
            
# Create a singleton instance for the app
orchestrator = IntelligentOrchestrator()
