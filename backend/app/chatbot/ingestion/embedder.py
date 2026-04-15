"""
Embedder
========
Encodes text chunks into dense vectors using sentence-transformers.

Model: ``all-MiniLM-L6-v2``
 - 80 MB download, 384-dim output
 - ~1 000 chunks/sec on CPU  →  entire 57-PDF corpus in <30 s
 - Top-tier quality for its size on MTEB retrieval benchmarks
 - No GPU needed — runs on CPU only
"""

import logging
from typing import List

import numpy as np

from app.core.config import EMBEDDING_MODEL_NAME

logger = logging.getLogger(__name__)

# Module-level cache so the model is loaded once per process
_model = None


def _get_model():
    """Lazy-load the sentence-transformer model (once)."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        logger.info("Loading embedding model: %s …", EMBEDDING_MODEL_NAME)
        _model = SentenceTransformer(EMBEDDING_MODEL_NAME, device="cpu")
        logger.info("Embedding model loaded.")
    return _model


def embed_texts(texts: List[str], batch_size: int = 64) -> np.ndarray:
    """
    Encode a list of strings into L2-normalised float32 vectors.

    Returns
    -------
    np.ndarray
        Shape ``(len(texts), 384)`` — ready for FAISS ``IndexFlatIP``.
    """
    model = _get_model()
    logger.info("Embedding %d texts (batch_size=%d) …", len(texts), batch_size)

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,  # L2-normalise → dot product = cosine sim
    )
    logger.info("Embeddings shape: %s", embeddings.shape)
    return embeddings.astype(np.float32)


def embed_query(query: str) -> np.ndarray:
    """Embed a single query string.  Returns shape ``(1, 384)``."""
    model = _get_model()
    vec = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return vec.astype(np.float32)
