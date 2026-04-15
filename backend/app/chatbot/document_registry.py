"""
Document Registry
=================
Singleton holder for the FAISS index and chunk metadata.  Lazy-loads on
first access so the index is not loaded at backend startup unless the
chatbot endpoint is actually called.
"""

import logging
from typing import List, Optional

import numpy as np

from app.chatbot.ingestion.vector_store import load_index, search as faiss_search

logger = logging.getLogger(__name__)


class _Registry:
    """Internal singleton — use module-level helpers instead."""

    def __init__(self):
        self._index = None
        self._metadata: Optional[List[dict]] = None
        self._loaded = False

    def ensure_loaded(self) -> None:
        if self._loaded:
            return
        logger.info("Loading FAISS index into memory …")
        self._index, self._metadata = load_index()
        self._loaded = True
        logger.info("FAISS index ready  (%d vectors).", self._index.ntotal)

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[dict]:
        self.ensure_loaded()
        return faiss_search(self._index, self._metadata, query_embedding, top_k)

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def total_chunks(self) -> int:
        if not self._loaded:
            return 0
        return self._index.ntotal

    def reload(self) -> None:
        """Force-reload the index from disk (e.g. after rebuild)."""
        self._loaded = False
        self._index = None
        self._metadata = None
        self.ensure_loaded()


# Module-level singleton
_registry = _Registry()

# Public API
ensure_loaded = _registry.ensure_loaded
search = _registry.search
reload = _registry.reload
is_loaded = lambda: _registry.is_loaded
total_chunks = lambda: _registry.total_chunks
