"""
TerraMind - Data partitioning for local-only benchmarks and edge adaptation.

Implements hierarchical partitioning:
  Level 1: Agro-climatic zone grouping
  Level 2: State-level grouping (default)
  Level 3: District calibration layer
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import pandas as pd

from backend.core.config import EDGE_ARTIFACTS
from backend.core.logging_config import log


def load_agro_climatic_zones() -> dict[str, list[str]]:
    """Load agro-climatic zone -> states mapping."""
    path = EDGE_ARTIFACTS / "agro_climatic_zones.json"
    if path.exists():
        with open(path, "r") as f:
            return json.load(f)
    log.warning("Agro-climatic zone file not found; falling back to state-level grouping.")
    return {}


def get_zone_for_state(state: str) -> Optional[str]:
    """Return the agro-climatic zone name for a given state, or None."""
    zones = load_agro_climatic_zones()
    state_lower = state.lower().strip()
    for zone_name, states in zones.items():
        if state_lower in [s.lower() for s in states]:
            return zone_name
    return None


def partition_by_state(df: pd.DataFrame, state_col: str = "State") -> dict[str, pd.DataFrame]:
    """Partition a DataFrame into per-state sub-DataFrames."""
    partitions = {}
    for state, grp in df.groupby(state_col, observed=True):
        if len(grp) >= 10:  # minimum viable partition size
            partitions[state] = grp.copy()
    log.info("Partitioned into %d state groups", len(partitions))
    return partitions


def partition_by_zone(df: pd.DataFrame, state_col: str = "State") -> dict[str, pd.DataFrame]:
    """
    Partition by agro-climatic zone.  Falls back to state if zone info unavailable.
    """
    zones = load_agro_climatic_zones()
    if not zones:
        return partition_by_state(df, state_col)

    # Build reverse map: state -> zone
    state_to_zone: dict[str, str] = {}
    for zone_name, states in zones.items():
        for s in states:
            state_to_zone[s.lower()] = zone_name

    df = df.copy()
    df["_zone"] = df[state_col].str.lower().map(state_to_zone).fillna("unknown")

    partitions = {}
    for zone, grp in df.groupby("_zone", observed=True):
        if len(grp) >= 20:
            partitions[zone] = grp.drop(columns=["_zone"]).copy()

    log.info("Partitioned into %d agro-climatic zones", len(partitions))
    return partitions
