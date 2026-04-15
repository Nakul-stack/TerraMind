"""
Context Builder
===============
Assembles the final LLM prompt from:
 - the system prompt (loaded from file)
 - an intent-specific instruction block
 - retrieved document chunks
 - optional diagnosis context (crop / disease)
 - the user question
"""

import logging
from pathlib import Path
from typing import List, Optional

from app.services.intent_detector import Intent

logger = logging.getLogger(__name__)

_PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"


def _load_prompt(filename: str) -> str:
    path = _PROMPTS_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}")
    return path.read_text(encoding="utf-8").strip()


def get_system_prompt() -> str:
    """Return the system prompt that enforces document-only answers."""
    return _load_prompt("system_prompt.txt")


def get_refusal_message() -> str:
    """Return the polite refusal template."""
    return _load_prompt("refusal_prompt.txt")


# ── Intent-specific instruction blocks ──────────────────────────────────────
# These are injected into the prompt so the LLM knows EXACTLY which sections
# to include and which to omit.

_INTENT_INSTRUCTIONS: dict[str, str] = {
    "pesticide": (
        "TASK: The user wants to know about pesticide or treatment.\n"
        "Reply with ONLY: pesticide name, dosage, and how to apply. Nothing else."
    ),
    "symptoms": (
        "TASK: The user wants to know about symptoms.\n"
        "Reply with ONLY: symptoms as bullet points. Nothing else."
    ),
    "cause": (
        "TASK: The user wants to know the cause.\n"
        "Reply with ONLY: what causes this condition. Nothing else."
    ),
    "prevention": (
        "TASK: The user wants to know about prevention.\n"
        "Reply with ONLY: prevention steps as bullet points. Nothing else."
    ),
    "severity": (
        "TASK: The user wants to know how serious this is.\n"
        "Reply with ONLY: severity level and quick action advice. Nothing else."
    ),
    "general": (
        "TASK: The user wants a full explanation.\n"
        "Reply with: overview, key symptoms, and treatment. Keep it concise."
    ),
}


def get_intent_instruction(intent: Intent) -> str:
    """Return the intent-specific instruction block."""
    return _INTENT_INSTRUCTIONS.get(intent, _INTENT_INSTRUCTIONS["general"])


def build_context_block(
    chunks: List[dict],
    identified_crop: Optional[str] = None,
    identified_class: Optional[str] = None,
) -> str:
    """
    Format retrieved chunks (and optional diagnosis context) into a text
    block that will be inserted between the system prompt and the user
    question.
    """
    parts: list[str] = []

    # If diagnosis context is available, put it first to bias the LLM
    if identified_crop or identified_class:
        diag_lines = ["[DIAGNOSIS CONTEXT]"]
        if identified_crop:
            diag_lines.append(f"Crop: {identified_crop}")
        if identified_class:
            label = identified_class.replace("___", " — ").replace("__", " — ").replace("_", " ")
            diag_lines.append(f"Condition: {label}")
        diag_lines.append("")
        parts.append("\n".join(diag_lines))

    # Append retrieved document chunks (strip document metadata from header)
    for i, chunk in enumerate(chunks, 1):
        parts.append(f"[Passage {i}]\n{chunk['text']}")

    return "\n\n".join(parts)


def build_full_prompt(
    question: str,
    chunks: List[dict],
    intent: Intent,
    identified_crop: Optional[str] = None,
    identified_class: Optional[str] = None,
) -> str:
    """
    Assemble the complete prompt string sent to Ollama.

    Structure
    ---------
    SYSTEM PROMPT
    ---
    INTENT INSTRUCTION
    ---
    CONTEXT (diagnosis + document chunks)
    ---
    USER QUESTION
    """
    system = get_system_prompt()
    intent_block = get_intent_instruction(intent)
    context = build_context_block(chunks, identified_crop, identified_class)

    return (
        f"{system}\n\n"
        f"---\n"
        f"{intent_block}\n"
        f"---\n"
        f"DOCUMENT CONTEXT:\n{context}\n"
        f"---\n\n"
        f"USER QUESTION: {question}"
    )
