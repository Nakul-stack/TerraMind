from typing import Optional, Protocol
from app.chatbot.router.types import RetrieverContext

class BaseRetriever(Protocol):
    """Interface for standardizing retrievers for TerraBot."""
    async def retrieve(self, query: str, diag_env: Optional[dict] = None, top_k: Optional[int] = None) -> RetrieverContext:
        ...
