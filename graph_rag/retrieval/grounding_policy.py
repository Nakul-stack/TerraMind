from __future__ import annotations

from dataclasses import dataclass


@dataclass
class GroundingDecision:
    allow_generation: bool
    message: str
    conservative_mode: bool
    metadata_limited: bool


def evaluate_grounding(total_external_docs: int, metadata_only: bool, has_local_kb_context: bool) -> GroundingDecision:
    # Hard rule for zero external evidence with no trusted local graph context.
    if total_external_docs == 0 and not has_local_kb_context:
        return GroundingDecision(
            allow_generation=False,
            message="No reliable external evidence retrieved from configured sources.",
            conservative_mode=True,
            metadata_limited=True,
        )

    # Fallback to local trusted KB is allowed, but mark confidence limits clearly.
    if total_external_docs == 0 and has_local_kb_context:
        return GroundingDecision(
            allow_generation=True,
            message="No reliable external evidence retrieved from configured sources. Answer grounded only in trusted local knowledge base.",
            conservative_mode=True,
            metadata_limited=True,
        )

    if metadata_only:
        return GroundingDecision(
            allow_generation=True,
            message="External evidence retrieved is metadata-limited; response is conservative and avoids unverified specifics.",
            conservative_mode=True,
            metadata_limited=True,
        )

    return GroundingDecision(
        allow_generation=True,
        message="External evidence retrieved and grounded.",
        conservative_mode=False,
        metadata_limited=False,
    )
