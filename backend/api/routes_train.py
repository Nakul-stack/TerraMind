"""
TerraMind - Training API routes.
"""
from __future__ import annotations

import asyncio
from fastapi import APIRouter, BackgroundTasks
from backend.core.logging_config import log

router = APIRouter(prefix="/train", tags=["training"])


def _train_all():
    """Synchronous training wrapper."""
    from backend.models.train_crop_recommender import train_crop_recommender
    from backend.models.train_yield_predictor import train_yield_predictor
    from backend.models.train_agri_advisor import train_agri_advisor

    log.info("Starting full training pipeline...")
    train_crop_recommender()
    train_yield_predictor()
    train_agri_advisor()
    log.info("All models trained successfully!")


@router.post("/all")
async def train_all(background_tasks: BackgroundTasks):
    """Train all three models. Runs in background."""
    background_tasks.add_task(_train_all)
    return {"status": "training_started", "message": "All models are being trained in the background"}


@router.post("/crop-recommender")
async def train_crop_rec(background_tasks: BackgroundTasks):
    from backend.models.train_crop_recommender import train_crop_recommender
    background_tasks.add_task(train_crop_recommender)
    return {"status": "training_started", "model": "crop_recommender"}


@router.post("/yield-predictor")
async def train_yield(background_tasks: BackgroundTasks):
    from backend.models.train_yield_predictor import train_yield_predictor
    background_tasks.add_task(train_yield_predictor)
    return {"status": "training_started", "model": "yield_predictor"}


@router.post("/agri-advisor")
async def train_agri(background_tasks: BackgroundTasks):
    from backend.models.train_agri_advisor import train_agri_advisor
    background_tasks.add_task(train_agri_advisor)
    return {"status": "training_started", "model": "agri_advisor"}
