"""
Ollama Client
=============
Thin HTTP wrapper around the Ollama REST API (``/api/generate``).

Uses ``httpx`` for a clean sync interface with configurable timeouts.
The MX330 GPU is too small for LLM inference, so the model runs on CPU;
we use a generous timeout to account for that.
"""

import logging
from typing import Optional

import httpx

from app.core.config import (
    OLLAMA_BASE_URL,
    OLLAMA_MODEL_NAME,
    OLLAMA_TEMPERATURE,
    OLLAMA_NUM_CTX,
    OLLAMA_TIMEOUT_SECONDS,
)

logger = logging.getLogger(__name__)


class OllamaError(RuntimeError):
    """Raised when Ollama is unreachable or returns an error."""


def generate(prompt: str, model: Optional[str] = None, num_predict: Optional[int] = None) -> str:
    """
    Send a prompt to Ollama and return the generated text.

    Parameters
    ----------
    prompt : str
        The full prompt (system + context + question), already assembled.
    model : str, optional
        Override the default model name.
    num_predict : int, optional
        Maximum number of tokens to generate.  When ``None``, Ollama uses
        its own default (unlimited).  Set to ~1200-1500 for capped output.

    Returns
    -------
    str
        The LLM-generated answer text.
    """
    model_name = model or OLLAMA_MODEL_NAME
    url = f"{OLLAMA_BASE_URL}/api/generate"

    options = {
        "temperature": OLLAMA_TEMPERATURE,
        "num_ctx": OLLAMA_NUM_CTX,
    }
    if num_predict is not None:
        options["num_predict"] = num_predict

    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False,
        "options": options,
    }

    try:
        logger.info("Sending request to Ollama [%s] …", model_name)
        with httpx.Client(timeout=OLLAMA_TIMEOUT_SECONDS) as client:
            resp = client.post(url, json=payload)
            resp.raise_for_status()
    except httpx.ConnectError:
        raise OllamaError(
            f"Cannot connect to Ollama at {OLLAMA_BASE_URL}. "
            "Is Ollama running?  Start it with: ollama serve"
        )
    except httpx.TimeoutException:
        raise OllamaError(
            f"Ollama request timed out after {OLLAMA_TIMEOUT_SECONDS}s. "
            "The model may be too large for your hardware or still loading."
        )
    except httpx.HTTPStatusError as exc:
        raise OllamaError(f"Ollama returned HTTP {exc.response.status_code}: {exc.response.text}")

    data = resp.json()
    answer = data.get("response", "").strip()
    logger.info(
        "Ollama responded — %d chars, eval_duration=%.1fs",
        len(answer),
        data.get("eval_duration", 0) / 1e9,
    )
    return answer


def is_available() -> bool:
    """Check whether Ollama is reachable and the configured model is present."""
    try:
        with httpx.Client(timeout=5) as client:
            resp = client.get(f"{OLLAMA_BASE_URL}/api/tags")
            if resp.status_code != 200:
                return False
            models = [m["name"] for m in resp.json().get("models", [])]
            # Ollama model names may have a ":latest" suffix
            return any(
                OLLAMA_MODEL_NAME in m for m in models
            )
    except Exception:
        return False
