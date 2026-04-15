import asyncio
import logging
from app.chatbot.router.types import RetrieverContext, RoutingDecision
from app.chatbot.client import generate as ollama_generate, OllamaError

logger = logging.getLogger(__name__)

COMPOSER_PROMPT = """You are TerraBot, a helpful and expert agricultural assistant.
You have been provided with specific context retrieved from our knowledge graph and/or agronomic literature. 
Respond directly and naturally to the exact question asked by the user, using ONLY the context provided.
Do not generate large lists, headings, or tables unless the user explicitly requested them or they are the most concise way to display highly structured warnings.
If the context does not contain the answer, say "I don't have enough information about that right now."

USER QUERY: {user_query}

REASONING STRATEGY DETECTED: {strategy} ({reason})

<CONTEXT>
{context_content}
</CONTEXT>

Generate your expert, natural response:
"""

async def compose_answer(user_query: str, context: RetrieverContext, routing_decision: RoutingDecision) -> str:
    if not context.text_content.strip():
        return "I don't have enough information about that right now."
        
    prompt = COMPOSER_PROMPT.format(
        user_query=user_query,
        strategy=routing_decision.strategy.value,
        reason=routing_decision.reason,
        context_content=context.text_content
    )
    
    try:
        # Long timeout for final generation
        answer = await asyncio.to_thread(ollama_generate, prompt, None, None)
        return answer.strip()
    except OllamaError as exc:
        logger.error(f"Failed to generate final answer: {exc}")
        return "I encountered an error while synthesizing the answer. Please try again."
