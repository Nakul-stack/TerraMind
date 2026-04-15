"""
Retrieval Guardrails
====================
Three layers of restriction ensuring the chatbot never answers outside
the provided PDF context.

Layer 1 - Similarity threshold:  reject if best score < threshold
Layer 2 - Off-topic keyword filter:  reject obvious non-agriculture queries
Layer 3 - Empty retrieval:  reject if FAISS returns no results at all
"""

import logging
import re
from typing import List, Tuple

from app.core.config import SIMILARITY_THRESHOLD

logger = logging.getLogger(__name__)

# ── Layer 2: Off-topic categories ────────────────────────────────────────────
# Simple keyword sets.  Not exhaustive - the LLM system prompt is the main
# defence - but this catches the most blatant off-topic queries cheaply.
_OFF_TOPIC_PATTERNS: List[re.Pattern] = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"\b(stock|bitcoin|crypto|forex|invest|trading)\b",
        r"\b(python|javascript|java|code|programming|algorithm|sql|html|css)\b",
        r"\b(president|election|congress|parliament|politics|democrat|republican)\b",
        r"\b(movie|celebrity|football|basketball|nba|fifa|song|album)\b",
        r"\b(recipe|cook|bake|restaurant|hotel)\b",
        r"\bwrite me (a|an|the)?\s?(poem|essay|story|song|code)\b",
    ]
]


def check_off_topic(question: str) -> bool:
    """Return ``True`` if the question appears to be off-topic."""
    for pattern in _OFF_TOPIC_PATTERNS:
        if pattern.search(question):
            logger.info("Off-topic filter triggered on: %r", question[:80])
            return True
    return False


def check_retrieval_quality(
    results: List[dict],
) -> Tuple[bool, str]:
    """
    Evaluate retrieval results and decide whether to allow an answer.

    Returns
    -------
    (allowed, reason)
        ``allowed`` is True if at least one chunk meets the similarity
        threshold.  ``reason`` explains why the answer is blocked (if so).
    """
    if not results:
        return False, "no_results"

    best_score = results[0].get("score", 0.0)
    if best_score < SIMILARITY_THRESHOLD:
        logger.info(
            "Best retrieval score %.3f < threshold %.3f -> refusing",
            best_score,
            SIMILARITY_THRESHOLD,
        )
        return False, "low_similarity"

    return True, "ok"


def run_guardrails(
    question: str,
    results: List[dict],
) -> Tuple[bool, str]:
    """
    Run all three guardrail layers.

    Returns
    -------
    (allowed, reason)
    """
    # Layer 2 - off-topic keyword check (runs before retrieval quality
    # because it's cheaper and catches blatant misuse)
    if check_off_topic(question):
        return False, "off_topic"

    # Layer 1 + 3 - retrieval quality / empty results
    return check_retrieval_quality(results)
