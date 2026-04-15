"""
TerraMind - Sync Service

Manages model version tracking, cache staleness, and
optional central ↔ edge synchronisation workflow.
"""
from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from backend.core.config import CENTRAL_ARTIFACTS, EDGE_ARTIFACTS
from backend.core.logging_config import log


SYNC_STATE_FILE = EDGE_ARTIFACTS / "sync_state.json"


def _read_sync_state() -> dict:
    if SYNC_STATE_FILE.exists():
        with open(SYNC_STATE_FILE) as f:
            return json.load(f)
    return {
        "last_sync": None,
        "central_version": "unknown",
        "edge_version": "unknown",
        "status": "never_synced",
    }


def _write_sync_state(state: dict):
    with open(SYNC_STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def get_sync_status() -> dict:
    """Return current sync state."""
    state = _read_sync_state()

    # Check central version
    central_meta = CENTRAL_ARTIFACTS / "crop_recommender" / "metadata.json"
    if central_meta.exists():
        with open(central_meta) as f:
            meta = json.load(f)
        state["central_version"] = meta.get("version", "unknown")
        state["central_trained_at"] = meta.get("trained_at", "unknown")

    # Check edge version
    edge_meta = EDGE_ARTIFACTS / "crop_recommender" / "metadata.json"
    if edge_meta.exists():
        with open(edge_meta) as f:
            meta = json.load(f)
        state["edge_version"] = meta.get("version", "unknown")
        state["edge_compressed"] = meta.get("edge_compressed", False)

    # Determine staleness
    if state.get("last_sync"):
        try:
            sync_dt = datetime.fromisoformat(state["last_sync"])
            age_hours = (datetime.now(timezone.utc) - sync_dt.replace(tzinfo=timezone.utc)).total_seconds() / 3600
            state["hours_since_sync"] = round(age_hours, 1)
            state["stale"] = age_hours > 168  # 1 week
        except Exception:
            state["stale"] = True
    else:
        state["stale"] = True

    return state


def pull_from_central() -> dict:
    """
    Sync: copy central artifacts to edge, rebuild caches.
    Returns sync result.
    """
    log.info("Starting sync: pulling central -> edge")

    if not CENTRAL_ARTIFACTS.exists():
        return {"success": False, "error": "Central artifacts not found"}

    # Backup current edge
    backup_dir = EDGE_ARTIFACTS.parent / "edge_backup"
    if EDGE_ARTIFACTS.exists():
        if backup_dir.exists():
            shutil.rmtree(backup_dir)
        shutil.copytree(EDGE_ARTIFACTS, backup_dir)
        log.info("Edge backup created at %s", backup_dir)

    try:
        # Copy model directories
        for model_dir in ["crop_recommender", "yield_predictor", "agri_advisor"]:
            src = CENTRAL_ARTIFACTS / model_dir
            dst = EDGE_ARTIFACTS / model_dir
            if src.exists():
                if dst.exists():
                    shutil.rmtree(dst)
                shutil.copytree(src, dst)

        # Update sync state
        state = _read_sync_state()
        state["last_sync"] = datetime.now(timezone.utc).isoformat()
        state["status"] = "synced"
        _write_sync_state(state)

        log.info("Sync completed successfully")
        return {"success": True, "synced_at": state["last_sync"]}

    except Exception as exc:
        log.error("Sync failed: %s - rolling back", exc)
        # Rollback
        if backup_dir.exists():
            if EDGE_ARTIFACTS.exists():
                shutil.rmtree(EDGE_ARTIFACTS)
            shutil.copytree(backup_dir, EDGE_ARTIFACTS)
        return {"success": False, "error": str(exc)}
