"""
TerraMind - Sync API routes.
"""
from __future__ import annotations

from fastapi import APIRouter
from backend.services.sync_service import get_sync_status, pull_from_central
from backend.schemas.benchmark_schema import SyncStatusResponse, SyncPullResponse

router = APIRouter(prefix="/sync", tags=["sync"])


@router.get("/status", response_model=SyncStatusResponse)
async def sync_status():
    """Show local artifact status and central version metadata."""
    return get_sync_status()


@router.post("/pull", response_model=SyncPullResponse)
async def sync_pull():
    """Pull latest central artifacts into edge package."""
    result = pull_from_central()
    return result
