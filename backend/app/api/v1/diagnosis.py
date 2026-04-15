"""
API endpoint for the Post-Symptom Diagnosis module.

POST /api/v1/diagnosis/predict
  - Accepts: multipart/form-data with an image file
  - Optional query param: top_k (default 3)
  - Returns: DiagnosisResponse JSON (includes report_id for async LLM report)

GET  /api/v1/diagnosis/report/{report_id}
  - Returns: report status and data when ready
"""

import logging
import uuid

from fastapi import APIRouter, File, HTTPException, Query, UploadFile, status, Request
from fastapi.responses import JSONResponse

from backend.app.core.rate_limit import limiter
from backend.app.core.runtime_config import (
    DIAGNOSIS_ALLOWED_CONTENT_TYPES,
    DIAGNOSIS_RATE_LIMIT,
    DIAGNOSIS_TOP_K_DEFAULT,
    DIAGNOSIS_TOP_K_MAX,
    DIAGNOSIS_TOP_K_MIN,
)
from backend.app.schemas.diagnosis import DiagnosisResponse
from backend.app.services.diagnosis_service import (
    run_diagnosis,
    run_report_generation,
    report_store,
)

router = APIRouter()
logger = logging.getLogger(__name__)

ALLOWED_CONTENT_TYPES = set(DIAGNOSIS_ALLOWED_CONTENT_TYPES)


@router.post(
    "/predict",
    response_model=DiagnosisResponse,
    status_code=status.HTTP_200_OK,
    summary="Predict Plant Disease from Image",
    description=(
        "Upload a leaf/plant image and receive identified crop, disease class, "
        "confidence score, and top-k ranked predictions.  Also returns a "
        "``report_id`` for polling the background LLM expert report."
    ),
)
@limiter.limit(DIAGNOSIS_RATE_LIMIT)
async def predict_diagnosis(
    request: Request,
    file: UploadFile = File(..., description="Leaf or plant image (JPEG / PNG / WebP)"),
    top_k: int = Query(
        DIAGNOSIS_TOP_K_DEFAULT,
        ge=DIAGNOSIS_TOP_K_MIN,
        le=DIAGNOSIS_TOP_K_MAX,
        description="Number of top predictions to return",
    ),
):
    """Diagnose plant disease from an uploaded image."""

    logger.info("Received diagnosis request - filename=%s, top_k=%d", file.filename, top_k)

    # ── Validate file presence ────────────────────────────────────────────
    if file.filename is None or file.filename == "":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No file was uploaded.",
        )

    # ── Validate content type ─────────────────────────────────────────────
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Invalid image type '{file.content_type}'. "
                f"Allowed types: {', '.join(sorted(ALLOWED_CONTENT_TYPES))}"
            ),
        )

    # ── Read image bytes ──────────────────────────────────────────────────
    try:
        image_bytes = await file.read()
        if len(image_bytes) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded file is empty.",
            )
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Failed to read uploaded file: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to read the uploaded file: {exc}",
        )

    # ── Run inference ─────────────────────────────────────────────────────
    try:
        result = run_diagnosis(image_bytes, top_k=top_k)
    except ValueError as exc:
        # Preprocessing / image errors
        logger.error("Image preprocessing error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        )
    except FileNotFoundError as exc:
        # Missing model artifacts
        logger.error("Model artifacts missing: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Diagnosis model is not available. Please contact the administrator.",
        )
    except RuntimeError as exc:
        logger.error("Model inference error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Model inference failed: {exc}",
        )
    except Exception as exc:
        logger.error("Unexpected diagnosis error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred during diagnosis.",
        )

    # ── Schedule background LLM report generation ─────────────────────────
    report_id = str(uuid.uuid4())
    report_store[report_id] = None  # Sentinel: "in progress"
    result.report_id = report_id

    import asyncio
    asyncio.ensure_future(
        run_report_generation(
            report_id=report_id,
            crop=result.identified_crop,
            disease=result.identified_class,
            confidence=result.confidence,
        )
    )

    logger.info(
        "Diagnosis prediction returned. report_id=%s scheduled for background LLM report.",
        report_id,
    )
    return result


@router.get(
    "/report/{report_id}",
    status_code=status.HTTP_200_OK,
    summary="Get LLM Diagnosis Report Status",
    description=(
        "Poll the status of a background-generated expert diagnosis report. "
        "Returns ``processing`` while the LLM is working, ``ready`` with "
        "the full report data when done, or ``error`` if generation failed."
    ),
)
async def get_report(report_id: str):
    """Return the status and data of a background LLM report."""

    if report_id not in report_store:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Report ID '{report_id}' not found.",
        )

    result = report_store[report_id]

    # Still generating
    if result is None:
        return JSONResponse(content={"status": "processing"})

    # Completed but with error
    if isinstance(result, dict) and "error" in result:
        return JSONResponse(
            content={"status": "error", "message": result["error"]}
        )

    # Success
    return JSONResponse(content={"status": "ready", "data": result})

@router.post(
    "/report/{report_id}/download",
    status_code=status.HTTP_200_OK,
    summary="Mark Report as Downloaded",
    description="Marks the diagnosis report as downloaded in the current session to unlock TerraBot.",
)
async def mark_report_downloaded(report_id: str):
    """Mark a generated report as downloaded."""
    from backend.app.services.diagnosis_service import report_store, downloaded_reports
    
    if report_id not in report_store:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Report ID '{report_id}' not found.",
        )
        
    downloaded_reports.add(report_id)
    return {"status": "ok", "message": "Report marked as downloaded"}
