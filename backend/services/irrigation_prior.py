"""
TerraMind - Irrigation Prior Service

Post-prediction prior that nudges irrigation_type predictions
based on district-level infrastructure availability from ICRISAT data.
This is a calibrated infrastructure prior, NOT a replacement model.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from backend.core.config import EDGE_ARTIFACTS
from backend.core.logging_config import log


# Mapping from irrigation infrastructure source -> compatible irrigation types
INFRA_TO_TYPE_COMPATIBILITY = {
    "canals":      ["canal", "surface", "flood"],
    "tanks":       ["surface", "flood"],
    "tube_wells":  ["drip", "sprinkler", "surface"],
    "other_wells": ["drip", "surface", "sprinkler"],
    "other":       ["rainfed", "surface"],
}


class IrrigationPriorService:
    """
    Applies a gentle district-level infrastructure prior to
    irrigation_type predictions.
    """

    def __init__(self):
        path = EDGE_ARTIFACTS / "district_irrigation_infra.json"
        self.infra_data: dict = {}
        if path.exists():
            with open(path) as f:
                self.infra_data = json.load(f)
        self.irr_pct_path = EDGE_ARTIFACTS / "district_crop_irrigated_area.json"
        self.irr_pct: dict = {}
        if self.irr_pct_path.exists():
            with open(self.irr_pct_path) as f:
                self.irr_pct = json.load(f)

    def apply_prior(
        self,
        state: str,
        district: str,
        crop: str,
        predicted_type: str,
        type_probabilities: dict[str, float] | None = None,
    ) -> dict:
        """
        Apply district irrigation infrastructure prior.

        Returns:
            {
              "adjusted_type": str,
              "original_type": str,
              "prior_used": bool,
              "reasoning": str,
              "district_irrigation_summary": str,
              "crop_irrigated_pct": float | None,
            }
        """
        key_sd = f"{state}|{district}"
        key_sdc = f"{state}|{district}|{crop}"

        result = {
            "adjusted_type": predicted_type,
            "original_type": predicted_type,
            "prior_used": False,
            "reasoning": "No district irrigation infrastructure data available",
            "district_irrigation_summary": "",
            "crop_irrigated_pct": self.irr_pct.get(key_sdc),
        }

        infra = self.infra_data.get(key_sd)
        if not infra:
            return result

        # Determine dominant source
        sources = {k: v for k, v in infra.items() if k != "net_irrigated" and v and v > 0}
        if not sources:
            result["reasoning"] = "District has no significant irrigation infrastructure recorded"
            return result

        total = sum(sources.values())
        dominant_src = max(sources, key=sources.get)
        dominant_pct = round(sources[dominant_src] / total * 100, 1)

        result["district_irrigation_summary"] = (
            f"Dominant source: {dominant_src.replace('_', ' ')} ({dominant_pct}% of irrigated area)"
        )

        # Check if predicted type is compatible with dominant infrastructure
        compatible_types = INFRA_TO_TYPE_COMPATIBILITY.get(dominant_src, [])

        if predicted_type.lower() in [t.lower() for t in compatible_types]:
            result["prior_used"] = True
            result["reasoning"] = (
                f"Predicted irrigation type '{predicted_type}' is compatible with "
                f"dominant district infrastructure ({dominant_src.replace('_', ' ')} at {dominant_pct}%)"
            )
        else:
            # Gentle nudge - only add reasoning, don't override unless very strong prior
            if dominant_pct > 70 and type_probabilities:
                # Check if any compatible type has reasonable probability
                for comp_type in compatible_types:
                    prob = type_probabilities.get(comp_type, 0)
                    if prob > 0.15:  # Only nudge if alternative has some support
                        result["adjusted_type"] = comp_type
                        result["prior_used"] = True
                        result["reasoning"] = (
                            f"Nudged from '{predicted_type}' to '{comp_type}' - "
                            f"district infrastructure strongly dominated by "
                            f"{dominant_src.replace('_', ' ')} ({dominant_pct}%), "
                            f"and '{comp_type}' had {prob:.0%} model probability"
                        )
                        break
                else:
                    result["prior_used"] = True
                    result["reasoning"] = (
                        f"Predicted '{predicted_type}' may not align with dominant "
                        f"district infrastructure ({dominant_src.replace('_', ' ')} at {dominant_pct}%), "
                        f"but no compatible alternative had sufficient model support"
                    )
            else:
                result["reasoning"] = (
                    f"Predicted '{predicted_type}'; district infrastructure is "
                    f"{dominant_src.replace('_', ' ')} ({dominant_pct}%) - "
                    f"no adjustment applied (prior not strong enough)"
                )

        return result
