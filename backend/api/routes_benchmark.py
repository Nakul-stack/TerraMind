"""
TerraMind - Benchmark API routes.
"""
from __future__ import annotations

from fastapi import APIRouter, BackgroundTasks
from backend.core.logging_config import log

router = APIRouter(prefix="/benchmark", tags=["benchmark"])


@router.post("/all")
async def benchmark_all_route(background_tasks: BackgroundTasks):
    """Run benchmark comparing centralized vs edge vs local models."""
    from backend.services.benchmark_service import benchmark_all
    background_tasks.add_task(benchmark_all)
    return {"status": "benchmark_started", "message": "Benchmark running in background"}


@router.get("/results")
async def benchmark_results():
    """Return the latest benchmark report."""
    import json
    from backend.core.config import CENTRAL_ARTIFACTS
    path = CENTRAL_ARTIFACTS / "benchmark_report.json"
    if not path.exists():
        return {"error": "No benchmark report found. Run POST /benchmark/all first."}
    with open(path) as f:
        return json.load(f)


@router.post("/edge-assets")
async def build_edge_assets(background_tasks: BackgroundTasks):
    """Build edge-deployable model artifacts and caches."""
    from backend.models.compress_edge_model import compress_all_edge_models
    background_tasks.add_task(compress_all_edge_models)
    return {"status": "edge_build_started", "message": "Edge assets building in background"}
