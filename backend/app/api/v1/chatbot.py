"""
Chatbot API - v1 Endpoints
===========================
POST /api/v1/chatbot/ask        - main question endpoint
GET  /api/v1/chatbot/status     - health check
POST /api/v1/chatbot/rebuild    - trigger index rebuild
"""

import logging

from fastapi import APIRouter, HTTPException, status, Request
from backend.app.core.rate_limit import limiter
from backend.app.core.runtime_config import CHATBOT_ASK_RATE_LIMIT, CHATBOT_REBUILD_RATE_LIMIT

from backend.app.schemas.chatbot import (
    ChatRequest,
    ChatResponse,
    ChatStatusResponse,
)
from backend.app.chatbot.router.orchestrator import orchestrator
from backend.app.chatbot import document_registry
from backend.app.chatbot.client import is_available as ollama_available
from backend.app.core.config import OLLAMA_MODEL_NAME

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post(
    "/ask",
    response_model=ChatResponse,
    status_code=status.HTTP_200_OK,
    summary="Ask the TerraMind PDF Assistant",
    description=(
        "Send a question to the document-grounded chatbot.  "
        "Optionally include diagnosis context (crop / disease) to bias retrieval."
    ),
)
@limiter.limit(CHATBOT_ASK_RATE_LIMIT)
async def ask_chatbot(request: Request, body_request: ChatRequest):
    logger.info(
        "Chatbot question: %s  (top_k=%d, crop=%s, class=%s)",
        body_request.question[:80],
        body_request.top_k,
        body_request.identified_crop,
        body_request.identified_class,
    )

    # ── Gated Diagnosis Workflow Verification ─────────────────────────────
    from backend.app.services.diagnosis_service import downloaded_reports
    if not body_request.report_id or body_request.report_id not in downloaded_reports:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access Denied: You must download the diagnosis report before querying TerraBot.",
        )

    try:
        diag_env = {
            "identified_crop": body_request.identified_crop,
            "identified_class": body_request.identified_class
        }
        result = await orchestrator.ask(
            user_query=body_request.question, 
            diag_env=diag_env,
            top_k=body_request.top_k
        )
        
        # ensure keys match ChatResponse schema
        return ChatResponse(
            answer=result.get("answer"),
            allowed=True,
            reason="ok",
            sources=result.get("sources", []),
            intent=result.get("strategy")
        )

    except Exception as e:
        logger.error("Unexpected chatbot error: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Chatbot processing error: {e}",
        )


@router.get(
    "/status",
    response_model=ChatStatusResponse,
    summary="Chatbot system status",
    description="Check whether the vector index is loaded and Ollama is available.",
)
def chatbot_status():
    return ChatStatusResponse(
        index_loaded=document_registry.is_loaded(),
        total_chunks=document_registry.total_chunks(),
        ollama_available=ollama_available(),
        ollama_model=OLLAMA_MODEL_NAME,
    )


@router.post(
    "/rebuild",
    status_code=status.HTTP_200_OK,
    summary="Rebuild the document index",
    description="Re-run the full PDF ingestion pipeline and reload the FAISS index.",
)
@limiter.limit(CHATBOT_REBUILD_RATE_LIMIT)
def rebuild_index(request: Request):
    logger.info("Index rebuild requested via API.")
    try:
        # Run the build pipeline
        from backend.app.chatbot.ingestion.build_index import main as build_main
        build_main()
        # Reload the registry from disk
        document_registry.reload()
        return {
            "status": "ok",
            "message": f"Index rebuilt - {document_registry.total_chunks()} chunks loaded.",
        }
    except Exception as e:
        logger.error("Index rebuild failed: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Index rebuild failed: {e}",
        )
