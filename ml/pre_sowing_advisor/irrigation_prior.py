"""
Irrigation Infrastructure Prior Module (v2).

Uses the structured ICRISAT Source breakdown (Canals, Tanks, Tube Wells, Wells)
from the District Intelligence Engine to gently nudge Model 3's irrigation_type
prediction towards historically plausible infrastructure.

This is a calibrated prior, NOT a replacement model.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

from ml.pre_sowing_advisor.normalizers import normalize_district_name, normalize_state_name

logger = logging.getLogger(__name__)

# Mapping from ICRISAT infrastructure names to irrigation-type labels
_INFRA_TO_IRR_TYPE: Dict[str, str] = {
    "canals": "canal",
    "tube wells": "drip",          # modern tube well districts tend to use drip/sprinkler
    "tanks": "flood",              # tank-fed systems are typically flood irrigation
    "other wells": "sprinkler",
    "other sources": "sprinkler",
}


def apply_irrigation_prior(
    irrigation_result: Dict[str, Any],
    state: str,
    district: str,
    crop: str,
    district_intelligence: Dict[str, Any],
) -> Dict[str, Any]:
    """Apply district irrigation infrastructure prior to model prediction.

    Augments the Model 3 output with:
      - district_prior_used: bool
      - district_irrigation_summary: str
      - irrigation_reasoning: str (human-readable explanation)

    Does NOT override the model's prediction — only adds context.
    """
    state = normalize_state_name(state)
    district = normalize_district_name(district)

    result = dict(irrigation_result)
    result["district_prior_used"] = False
    result["district_irrigation_summary"] = ""
    result["irrigation_reasoning"] = ""

    infra_summary = district_intelligence.get("irrigation_infrastructure_summary", "")
    infra_breakdown = district_intelligence.get("irrigation_infrastructure_breakdown", {})
    crop_irr_pct = district_intelligence.get("crop_irrigated_area_percent")

    if not infra_summary or not infra_breakdown:
        result["district_irrigation_summary"] = "No district irrigation data available."
        return result

    result["district_prior_used"] = True
    result["district_irrigation_summary"] = infra_summary

    # ── Build human-readable reasoning ──────────────────────────────
    reasoning_parts = []
    predicted_type = result.get("irrigation_type", "").lower()

    # Find dominant source
    if infra_breakdown:
        dominant_source = max(infra_breakdown, key=infra_breakdown.get)
        dominant_pct = infra_breakdown[dominant_source]
        dominant_irr_type = _INFRA_TO_IRR_TYPE.get(dominant_source.lower(), dominant_source.lower())

        if predicted_type == dominant_irr_type or predicted_type in dominant_source.lower():
            reasoning_parts.append(
                f"Model prediction ({predicted_type}) aligns with district's dominant "
                f"infrastructure ({dominant_source}: {dominant_pct}%)."
            )
        else:
            reasoning_parts.append(
                f"Note: district infrastructure is dominated by {dominant_source} ({dominant_pct}%), "
                f"while model predicted '{predicted_type}'. "
                f"Consider local infrastructure feasibility."
            )

    if crop_irr_pct is not None:
        if crop_irr_pct > 70:
            reasoning_parts.append(
                f"This crop is well-irrigated here ({crop_irr_pct}% of area is irrigated)."
            )
        elif crop_irr_pct > 30:
            reasoning_parts.append(
                f"Moderate irrigation coverage ({crop_irr_pct}% of {crop} area is irrigated)."
            )
        else:
            reasoning_parts.append(
                f"Low irrigation coverage ({crop_irr_pct}% irrigated). "
                f"Rain-fed cultivation is common for this crop here."
            )

    reasoning_parts.append(f"District infrastructure: {infra_summary}")

    result["irrigation_reasoning"] = " ".join(reasoning_parts)

    logger.info(
        "Irrigation prior applied: %s/%s → prior_used=%s, dominant=%s",
        state, district, result["district_prior_used"],
        list(infra_breakdown.keys())[0] if infra_breakdown else "unknown",
    )
    return result
