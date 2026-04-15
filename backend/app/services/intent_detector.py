"""
Intent Detector
===============
Analyses the user question and returns a classified intent so the
chatbot can tailor its response to ONLY what the user asked.

Supported intents
-----------------
- pesticide   : user asks about pesticide / fungicide / treatment / medicine
- symptoms    : user asks about symptoms / signs / indicators
- cause       : user asks about cause / reason / why it happens
- prevention  : user asks about prevention / how to prevent / avoid
- severity    : user asks about severity / how serious / how bad
- general     : broad / open-ended question (full explanation allowed)
"""

import re
from typing import Literal

Intent = Literal[
    "pesticide",
    "symptoms",
    "cause",
    "prevention",
    "severity",
    "general",
]

# ── Keyword patterns per intent ──────────────────────────────────────────────
# Order matters - first match wins, so more specific intents go first.

_INTENT_PATTERNS: list[tuple[Intent, re.Pattern]] = [
    (
        "pesticide",
        re.compile(
            r"\b("
            r"pesticides?|fungicides?|insecticides?|herbicides?|spray|chemicals?|"
            r"treatments?|treat|medicines?|remedy|cure|manage|management|"
            r"control|what.*(apply|use|spray)"
            r")\b",
            re.IGNORECASE,
        ),
    ),
    (
        "symptoms",
        re.compile(
            r"\b("
            r"symptoms?|signs?|indicators?|look\s*like|identify|"
            r"how.*know|how.*tell|what.*happen|appears?|"
            r"visual|visible|spots?|discoloration|wilts?|lesions?"
            r")\b",
            re.IGNORECASE,
        ),
    ),
    (
        "cause",
        re.compile(
            r"\b("
            r"causes?|reasons?|why|how.*start|how.*spread|"
            r"origin|source|pathogens?|bacteria|virus|fungus|"
            r"what\s*causes|caused\s*by"
            r")\b",
            re.IGNORECASE,
        ),
    ),
    (
        "prevention",
        re.compile(
            r"\b("
            r"prevents?|prevention|avoid|protect|safeguard|"
            r"precautions?|how.*stop|keep.*safe|"
            r"reduce.*risk|resistanc|resistant"
            r")\b",
            re.IGNORECASE,
        ),
    ),
    (
        "severity",
        re.compile(
            r"\b("
            r"sever\w*|serious|how\s*bad|dangerous|fatal|"
            r"risk.*level|damage.*level|threats?|"
            r"how\s*much\s*damage|yield\s*loss|critical"
            r")\b",
            re.IGNORECASE,
        ),
    ),
]


def detect_user_intent(query: str) -> Intent:
    """
    Classify a user query into one of the supported intent categories.

    Falls back to ``"general"`` when no specific intent pattern matches,
    which is the ONLY case where a multi-section response is allowed.
    """
    q = query.strip()
    for intent, pattern in _INTENT_PATTERNS:
        if pattern.search(q):
            return intent
    return "general"
