from fastapi import APIRouter
from app.api.v1 import advisor, diagnosis, monitor, chatbot
from routers.graph_rag_router import router as graph_rag_router

api_router = APIRouter()

api_router.include_router(
    advisor.router,
    prefix="/advisor",
    tags=["Pre-sowing Advisor"]
)

api_router.include_router(
    diagnosis.router,
    prefix="/diagnosis",
    tags=["Post-Symptom Diagnosis"]
)

api_router.include_router(
    monitor.router,
    prefix="/monitor",
    tags=["Growth Stage Monitor"]
)

api_router.include_router(
    chatbot.router,
    prefix="/chatbot",
    tags=["PDF Chatbot Assistant"]
)

api_router.include_router(
    graph_rag_router,
    prefix="/graph-rag",
    tags=["Graph RAG Advisor"]
)
