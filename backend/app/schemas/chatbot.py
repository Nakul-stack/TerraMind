"""
Chatbot Schemas
===============
Pydantic models for the chatbot API request / response.
"""

from typing import List, Optional, Union
from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """Incoming chat question - optionally with diagnosis context."""
    question: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="The user's question to ask about the PDF documents.",
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of document chunks to retrieve.",
    )
    identified_crop: Optional[str] = Field(
        default=None,
        description="Crop from diagnosis context (e.g. 'Tomato').",
    )
    identified_class: Optional[str] = Field(
        default=None,
        description="Disease class from diagnosis context (e.g. 'Tomato___Late_blight').",
    )
    report_id: Optional[str] = Field(
        default=None,
        description="ID of the downloaded diagnosis report to unlock TerraBot.",
    )


class SourceCitation(BaseModel):
    """A single source citation from a retrieved PDF chunk."""
    file_name: str = Field(..., description="Name of the PDF file.")
    page: Union[int, str] = Field(..., description="Page number (1-indexed) or '-' for graph results.")
    snippet: str = Field(..., description="Preview of the chunk text.")
    score: float = Field(default=0.0, description="Cosine similarity score.")


class ChatResponse(BaseModel):
    """Response from the chatbot endpoint."""
    answer: str = Field(..., description="The chatbot's answer.")
    allowed: bool = Field(
        ...,
        description="True if the answer is grounded in documents, False if refused.",
    )
    reason: str = Field(
        default="ok",
        description="Reason code: 'ok', 'off_topic', 'low_similarity', 'no_results', 'index_missing', 'llm_error'.",
    )
    sources: List[SourceCitation] = Field(
        default_factory=list,
        description="Source citations for the answer.",
    )
    intent: Optional[str] = Field(
        default=None,
        description="Detected user intent: 'pesticide', 'symptoms', 'cause', 'prevention', 'severity', 'general'.",
    )


class ChatStatusResponse(BaseModel):
    """Health check / status response for the chatbot subsystem."""
    index_loaded: bool
    total_chunks: int
    ollama_available: bool
    ollama_model: str
