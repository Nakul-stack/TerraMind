"""
Service layer for the Post-Symptom Diagnosis module.

Bridges the FastAPI endpoint with the ML inference code, converting raw
prediction dicts into validated Pydantic response objects.

Also provides:
- ``report_store``: in-memory dict tracking background LLM report status.
- ``run_report_generation()``: async background task that produces
  structured reports via the Graph RAG + Ollama pipeline.
"""

import asyncio
import logging
import uuid
from typing import Any, Dict, Optional

from backend.app.core.runtime_config import DIAGNOSIS_REPORT_TTL_SECONDS
from app.schemas.diagnosis import DiagnosisResponse, TopPrediction
from ml.post_symptom_diagnosis.inference.predict import predict_disease

logger = logging.getLogger(__name__)

# ── In-memory report store ───────────────────────────────────────────────────
# Key: report_id (UUID string)
# Value: None  -> report still generating
#        dict  -> completed report (11 keys on success, or {"error": ...})
report_store: Dict[str, Optional[Dict[str, Any]]] = {}

# ── In-memory report download status ─────────────────────────────────────────
# Stores report IDs that have been explicitly marked as downloaded by the user.
# Used to unlock the TerraBot smart assistant functionality.
downloaded_reports: set = set()

def run_diagnosis(image_bytes: bytes, top_k: int = 3) -> DiagnosisResponse:
    """
    Execute the full diagnosis pipeline.

    Parameters
    ----------
    image_bytes : bytes
        Raw bytes of the uploaded image.
    top_k : int
        Number of top predictions to include.

    Returns
    -------
    DiagnosisResponse
        Validated Pydantic response.

    Raises
    ------
    ValueError
        For preprocessing / image errors.
    RuntimeError
        For model loading / inference failures.
    """
    logger.info("Diagnosis service - running inference (top_k=%d)", top_k)

    raw: Dict[str, Any] = predict_disease(image_bytes, top_k=top_k)

    # Convert raw dicts into Pydantic models
    top_predictions = [
        TopPrediction(
            crop=pred["crop"],
            # Use the Pydantic alias "class" via the field name "class_name"
            **{"class": pred["class"]},
            confidence=pred["confidence"],
        )
        for pred in raw["top_k_predictions"]
    ]

    response = DiagnosisResponse(
        identified_crop=raw["identified_crop"],
        identified_class=raw["identified_class"],
        confidence=raw["confidence"],
        top_k_predictions=top_predictions,
        assistant_available=raw.get("assistant_available", True),
    )

    logger.info(
        "Diagnosis complete - %s / %s (%.1f%%)",
        response.identified_crop,
        response.identified_class,
        response.confidence * 100,
    )
    return response


async def run_report_generation(
    report_id: str, crop: str, disease: str, confidence: float
) -> None:
    """
    Background task: generate a structured diagnosis report via
    Graph RAG + Ollama and store the result in ``report_store``.

    After storing the result, schedules automatic cleanup of the entry
    after 30 minutes to prevent unbounded memory growth.

    Parameters
    ----------
    report_id : str
        UUID key in ``report_store``.
    crop : str
        Top-1 predicted crop name.
    disease : str
        Top-1 predicted disease class label.
    confidence : float
        Top-1 CNN confidence (0–1).
    """
    logger.info("Background report task started - report_id=%s", report_id)
    try:
        from graph_rag.query_engine import generate_diagnosis_report

        result = await generate_diagnosis_report(crop, disease, confidence)
        report_store[report_id] = result

        if "error" in result:
            logger.warning(
                "Report generation completed with error - report_id=%s: %s",
                report_id, result["error"],
            )
        else:
            logger.info("Report generation succeeded - report_id=%s", report_id)

    except Exception as exc:
        logger.error(
            "Report generation crashed - report_id=%s: %s", report_id, exc,
        )
        report_store[report_id] = {"error": f"Report generation failed: {exc}"}

    # Schedule automatic cleanup for report session data.
    try:
        loop = asyncio.get_running_loop()
        
        def cleanup_report():
            report_store.pop(report_id, None)
            downloaded_reports.discard(report_id)
            
        loop.call_later(DIAGNOSIS_REPORT_TTL_SECONDS, cleanup_report)
        logger.info(
            "Scheduled report cleanup in %ss - report_id=%s",
            DIAGNOSIS_REPORT_TTL_SECONDS,
            report_id,
        )
    except RuntimeError:
        # Not inside a running loop (shouldn't happen in FastAPI, but be safe)
        logger.warning("Could not schedule cleanup for report_id=%s", report_id)
