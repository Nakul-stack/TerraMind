"""
Structured LLM prompt builder for diagnosis reports.
=====================================================
Builds the system and user prompts for the Graph RAG diagnosis report,
and parses/validates the structured JSON response from the LLM.
"""

import json
import logging
import re
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

# The 11 required keys in the report, in canonical order.
REQUIRED_KEYS: List[str] = [
    "crop_identified",
    "disease_identified",
    "disease_overview",
    "symptoms",
    "causes",
    "severity",
    "immediate_steps",
    "treatment",
    "prevention",
    "possible_impact",
    "monitoring_advice",
]


def build_system_prompt() -> str:
    """
    Return the system prompt that constrains the LLM to produce
    a clean JSON object with exactly 11 keys and no markdown.

    Returns
    -------
    str
        The full system prompt string.
    """
    return (
        "You are an agricultural plant pathologist. "
        "Reply ONLY with a JSON object having these 11 keys: "
        "crop_identified, disease_identified, disease_overview, "
        "symptoms, causes, severity, immediate_steps, treatment, "
        "prevention, possible_impact, monitoring_advice. "
        "Values: plain strings, 1-3 sentences each. "
        "Do not use asterisks, double-asterisks, or any markdown formatting "
        "anywhere in your response. No backticks, no code fences."
    )


def build_user_prompt(
    crop: str,
    disease: str,
    confidence: float,
    context_chunks: List[str],
) -> str:
    """
    Build the user prompt that provides context and asks the LLM to
    fill all 11 report fields.

    Parameters
    ----------
    crop : str
        The identified crop name.
    disease : str
        The identified disease class label.
    confidence : float
        CNN prediction confidence (0–1 scale, displayed as percentage).
    context_chunks : list[str]
        Retrieved knowledge context strings from the Graph RAG engine.

    Returns
    -------
    str
        The assembled user prompt.
    """
    disease_display = disease.replace("___", " — ").replace("__", " ").replace("_", " ")
    conf_pct = confidence * 100

    context_block = "\n\n".join(
        f"[Source {i + 1}]\n{chunk}" for i, chunk in enumerate(context_chunks)
    ) if context_chunks else "(No additional context available.)"

    return (
        f"A CNN model identified the crop as \"{crop}\" with the disease "
        f"\"{disease_display}\" at {conf_pct:.1f}% confidence.\n\n"
        f"Below is relevant agricultural knowledge context:\n\n"
        f"{context_block}\n\n"
        f"Using only the information above and your expert knowledge, "
        f"produce a comprehensive diagnostic report as a JSON object with "
        f"these 11 fields:\n"
        f"1. crop_identified — the crop name\n"
        f"2. disease_identified — the disease name in plain English\n"
        f"3. disease_overview — a detailed overview of this disease\n"
        f"4. symptoms — visible symptoms farmers should look for\n"
        f"5. causes — what causes this disease (pathogens, environment)\n"
        f"6. severity — how severe this disease typically is\n"
        f"7. immediate_steps — urgent actions a farmer should take now\n"
        f"8. treatment — recommended chemical or biological treatments\n"
        f"9. prevention — preventive measures for future seasons\n"
        f"10. possible_impact — economic and yield impact if untreated\n"
        f"11. monitoring_advice — what to monitor in the coming days\n\n"
        f"Return ONLY the JSON object. No explanation, no markdown."
    )


def parse_llm_response(raw_text: str) -> Dict[str, Any]:
    """
    Parse and validate the LLM's raw text output into a structured dict.

    Strips any JSON code fences, leading/trailing whitespace, and stray
    asterisks from all string values.  Validates that all 11 required
    keys are present.

    Parameters
    ----------
    raw_text : str
        The raw text response from the LLM.

    Returns
    -------
    dict
        Parsed and cleaned report dictionary with all 11 keys.

    Raises
    ------
    ValueError
        If JSON parsing fails or required keys are missing.
    """
    cleaned = raw_text.strip()

    # Strip ```json ... ``` or ``` ... ``` fences
    cleaned = re.sub(r"^```(?:json)?\s*\n?", "", cleaned)
    cleaned = re.sub(r"\n?```\s*$", "", cleaned)
    cleaned = cleaned.strip()

    # Try to extract the JSON object if there's surrounding text
    json_match = re.search(r"\{[\s\S]*\}", cleaned)
    if json_match:
        cleaned = json_match.group(0)

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Failed to parse LLM response as JSON: {exc}\n"
            f"Raw text:\n{raw_text}"
        )

    if not isinstance(parsed, dict):
        raise ValueError(
            f"LLM response is not a JSON object (got {type(parsed).__name__}).\n"
            f"Raw text:\n{raw_text}"
        )

    # Strip asterisks and markdown formatting from all string values
    for key, value in parsed.items():
        if isinstance(value, str):
            # Remove **, *, and stray backticks
            value = value.replace("**", "").replace("*", "")
            value = value.replace("`", "")
            parsed[key] = value.strip()

    # Validate required keys
    missing = [k for k in REQUIRED_KEYS if k not in parsed]
    if missing:
        raise ValueError(
            f"LLM response missing required keys: {missing}\n"
            f"Raw text:\n{raw_text}"
        )

    logger.info("LLM response parsed successfully — all 11 keys present.")
    return parsed
