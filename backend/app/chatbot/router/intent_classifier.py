import json
import logging
import asyncio
from typing import Dict, Any

from app.chatbot.router.types import RoutingDecision, RetrievalStrategy
from app.chatbot.client import generate as ollama_generate, OllamaError

logger = logging.getLogger(__name__)

LLM_CLASSIFIER_PROMPT = """You are an expert intent router for an agriculture chatbot, TerraBot. 
Your job is to read the user's query and route it to the correct retrieval engine.

AVAILABLE ENGINES:
1. PDF_RAG: Choose this if the query asks for descriptive explanations, "how-to" procedural guides, background literature, summarization of documents, or general advice. (e.g., "Explain early blight", "Summarize this manual").
2. GRAPH_RAG: Choose this if the query requires highly structured data lookups, multi-hop relationship matching, conflicts, tank-mix safety, or direct trait linkage. (e.g., "What diseases peak in high humidity for rice?", "Can I mix Azoxystrobin with Copper?").
3. HYBRID_RAG: Choose this if the query explicitly requires both official descriptive guidance AND structured lookups. (e.g., "Recommend a treatment for powdery mildew and provide the official safety precautions", "What fungicide should I use for early blight and what precautions are mentioned in the manual?").

USER QUERY: "{user_query}"

OUTPUT FORMAT:
You must output ONLY valid JSON in the following format:
{{
  "strategy": "PDF_RAG" | "GRAPH_RAG" | "HYBRID_RAG",
  "confidence": <float between 0.0 and 1.0>,
  "reason": "<short explanation>"
}}
"""

def rule_based_fallback(query: str) -> RoutingDecision:
    """Fallback if LLM classification fails."""
    q = query.lower()
    
    graph_indicators = ["mix", "compatible", "tank", "high humidity", "weather", "temperature", "peaks during", "linked to", "associations", "conflicts"]
    pdf_indicators = ["explain", "summarize", "manual", "guide", "document", "describe", "what is", "how to"]
    
    has_graph = any(kw in q for kw in graph_indicators)
    has_pdf = any(kw in q for kw in pdf_indicators)
    
    if has_graph and has_pdf:
        return RoutingDecision(strategy=RetrievalStrategy.HYBRID_RAG, confidence=0.6, reason="Matched keywords for both graph and PDF.")
    elif "official guidance" in q or "literature" in q:
        return RoutingDecision(strategy=RetrievalStrategy.HYBRID_RAG, confidence=0.7, reason="Mentions official literature/guidance.")
    elif has_graph:
        return RoutingDecision(strategy=RetrievalStrategy.GRAPH_RAG, confidence=0.7, reason="Matched graph keywords (mix, climate, etc.).")
    elif has_pdf:
        return RoutingDecision(strategy=RetrievalStrategy.PDF_RAG, confidence=0.7, reason="Matched PDF keywords (explain, guide, etc.).")
    else:
        # Default to Hybrid RAG if uncertain
        return RoutingDecision(strategy=RetrievalStrategy.HYBRID_RAG, confidence=0.4, reason="Default fallback due to uncertainty.")

async def classify_intent(query: str) -> RoutingDecision:
    """Classify user query and return routing decision using async Ollama call."""
    prompt = LLM_CLASSIFIER_PROMPT.format(user_query=query)
    try:
        raw_response = await asyncio.to_thread(ollama_generate, prompt, None, 800)
        
        # Clean response string: might have markdown code blocks
        clean_resp = raw_response.strip()
        if clean_resp.startswith("```json"):
            clean_resp = clean_resp[7:]
        elif clean_resp.startswith("```"):
            clean_resp = clean_resp[3:]
        if clean_resp.endswith("```"):
            clean_resp = clean_resp[:-3]
            
        data = json.loads(clean_resp.strip())
        
        strategy_str = data.get("strategy")
        try:
            strategy = RetrievalStrategy(strategy_str)
        except ValueError:
            strategy = RetrievalStrategy.HYBRID_RAG
            
        confidence = float(data.get("confidence", 0.5))
        reason = data.get("reason", "LLM decided.")
        
        # Guardrail on confidence
        if confidence < 0.50 and strategy != RetrievalStrategy.HYBRID_RAG:
            logger.info("Confidence < 0.50, defaulting to HYBRID_RAG.")
            strategy = RetrievalStrategy.HYBRID_RAG
            reason += " (low confidence override)"
            
        return RoutingDecision(strategy=strategy, confidence=confidence, reason=reason)
        
    except (OllamaError, json.JSONDecodeError, Exception) as exc:
        logger.warning(f"LLM classifier failed ({exc}), falling back to rule-based routing.")
        return rule_based_fallback(query)
