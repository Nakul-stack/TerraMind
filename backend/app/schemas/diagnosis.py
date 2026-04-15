"""
Pydantic schemas for the Post-Symptom Diagnosis API.
"""

from typing import List, Optional
from pydantic import BaseModel, Field


class TopPrediction(BaseModel):
    """A single entry in the top-k predictions list."""
    crop: str = Field(..., description="Identified crop name")
    class_name: str = Field(..., alias="class", description="Full disease class label")
    confidence: float = Field(..., ge=0, le=1, description="Prediction confidence (0–1)")

    class Config:
        populate_by_name = True


class DiagnosisResponse(BaseModel):
    """Response schema returned by the diagnosis prediction endpoint."""
    identified_crop: str = Field(..., description="Top-1 predicted crop name")
    identified_class: str = Field(..., description="Top-1 predicted disease class label")
    confidence: float = Field(..., ge=0, le=1, description="Top-1 prediction confidence")
    top_k_predictions: List[TopPrediction] = Field(
        ..., description="Ranked list of top-k predictions"
    )
    assistant_available: bool = Field(
        True, description="Whether the smart assistant chatbot is available for follow-up"
    )
    report_id: Optional[str] = Field(
        None, description="Report tracking token for background LLM report generation"
    )

