"""
FAISS Vector Store
==================
Manages building, saving, loading, and querying a FAISS flat inner-product
index.  Because the embeddings are L2-normalised, inner-product scores equal
cosine similarity.

Why FAISS flat index?
 - For ~15 000 chunks × 384 dims the index is ~23 MB — fits in RAM trivially.
 - Flat (brute-force) search is exact and still <5 ms for top-5.
 - No training / quantisation needed → simpler and more predictable.
"""

import json
import logging
from pathlib import Path
from typing import List, Tuple

import faiss
import numpy as np

from app.core.config import VECTOR_STORE_DIR

logger = logging.getLogger(__name__)

INDEX_FILE = "faiss_index.bin"
META_FILE = "chunk_metadata.json"


def build_and_save(
    embeddings: np.ndarray,
    metadata: List[dict],
    store_dir: str | None = None,
) -> None:
    """
    Build a FAISS ``IndexFlatIP`` and persist it alongside chunk metadata.

    Parameters
    ----------
    embeddings : np.ndarray
        Shape ``(n, dim)`` — L2-normalised vectors.
    metadata : list[dict]
        One dict per row containing at minimum ``chunk_id``, ``file_name``,
        ``page``, and ``text``.
    store_dir : str, optional
        Directory to write index + metadata into (default from config).
    """
    directory = Path(store_dir or VECTOR_STORE_DIR)
    directory.mkdir(parents=True, exist_ok=True)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # inner product on L2-normed = cosine
    index.add(embeddings)

    index_path = directory / INDEX_FILE
    meta_path = directory / META_FILE

    faiss.write_index(index, str(index_path))
    meta_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    logger.info(
        "Vector store saved — %d vectors, dim=%d → %s",
        index.ntotal,
        dim,
        directory,
    )


def load_index(store_dir: str | None = None) -> Tuple[faiss.Index, List[dict]]:
    """Load the persisted FAISS index and chunk metadata from disk."""
    directory = Path(store_dir or VECTOR_STORE_DIR)
    index_path = directory / INDEX_FILE
    meta_path = directory / META_FILE

    if not index_path.exists():
        raise FileNotFoundError(
            f"FAISS index not found at {index_path}. "
            "Run the index builder first: python -m app.chatbot.ingestion.build_index"
        )
    if not meta_path.exists():
        raise FileNotFoundError(
            f"Chunk metadata not found at {meta_path}. "
            "Run the index builder first: python -m app.chatbot.ingestion.build_index"
        )

    index = faiss.read_index(str(index_path))
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))

    logger.info(
        "Vector store loaded — %d vectors from %s", index.ntotal, directory
    )
    return index, metadata


def search(
    index: faiss.Index,
    metadata: List[dict],
    query_embedding: np.ndarray,
    top_k: int = 5,
) -> List[dict]:
    """
    Search the FAISS index and return top-k results with scores and metadata.

    Returns
    -------
    list[dict]
        Each dict contains ``score``, ``chunk_id``, ``file_name``, ``page``,
        and ``text``.
    """
    scores, indices = index.search(query_embedding, top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue  # FAISS sentinel for "no more results"
        meta = metadata[idx].copy()
        meta["score"] = float(score)
        results.append(meta)

    return results
