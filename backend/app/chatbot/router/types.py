import json
from enum import Enum
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

class RetrievalStrategy(str, Enum):
    PDF_RAG = "PDF_RAG"
    GRAPH_RAG = "GRAPH_RAG"
    HYBRID_RAG = "HYBRID_RAG"

class RoutingDecision(BaseModel):
    strategy: RetrievalStrategy
    confidence: float
    reason: str

class RetrieverContext(BaseModel):
    """Normalized context format for answer generation."""
    text_content: str = ""
    metadata: Dict[str, Any] = Field(default_factory=dict)
