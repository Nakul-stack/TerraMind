"""
TerraMind - Benchmark & Sync schemas.
"""
from __future__ import annotations
from typing import Optional
from pydantic import BaseModel


class BenchmarkResult(BaseModel):
    systems: dict
    target_gap_pp: float = 5.0
    edge_within_target: Optional[bool] = None


class SyncStatusResponse(BaseModel):
    central_version: str = "unknown"
    edge_version: str = "unknown"
    last_sync: Optional[str] = None
    stale: bool = True
    hours_since_sync: Optional[float] = None
    central_trained_at: Optional[str] = None
    edge_compressed: Optional[bool] = None


class SyncPullResponse(BaseModel):
    success: bool
    synced_at: Optional[str] = None
    error: Optional[str] = None
