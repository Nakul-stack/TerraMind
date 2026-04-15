"""
Index Builder — CLI entry point
================================
Orchestrates the full ingestion pipeline:
  1. Load PDFs  →  2. Chunk  →  3. Embed  →  4. Build & save FAISS index

Usage
-----
    cd backend
    python -m app.chatbot.ingestion.build_index

The resulting index is persisted to ``backend/app/chatbot/storage/`` so it
only needs to be rebuilt when new PDFs are added.
"""

import logging
import sys
import time
from pathlib import Path

# Ensure project root is on sys.path so ``app.*`` imports resolve
_root = Path(__file__).resolve().parents[3]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from app.core.config import PDF_FOLDER_PATH, VECTOR_STORE_DIR
from app.chatbot.ingestion.pdf_loader import load_pdfs
from app.chatbot.ingestion.chunker import chunk_pages
from app.chatbot.ingestion.embedder import embed_texts
from app.chatbot.ingestion.vector_store import build_and_save

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)-40s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("build_index")


def main() -> None:
    t0 = time.perf_counter()

    # ── 1. Load PDFs ─────────────────────────────────────────────────────
    logger.info("═══ Step 1/4 — Loading PDFs from %s", PDF_FOLDER_PATH)
    pages = load_pdfs()
    if not pages:
        logger.error("No page records extracted.  Aborting.")
        sys.exit(1)

    # ── 2. Chunk ─────────────────────────────────────────────────────────
    logger.info("═══ Step 2/4 — Chunking text")
    chunks = chunk_pages(pages)
    if not chunks:
        logger.error("No chunks produced.  Aborting.")
        sys.exit(1)

    # ── 3. Embed ─────────────────────────────────────────────────────────
    logger.info("═══ Step 3/4 — Generating embeddings")
    texts = [c["text"] for c in chunks]
    embeddings = embed_texts(texts)

    # ── 4. Build & save FAISS index ──────────────────────────────────────
    logger.info("═══ Step 4/4 — Building FAISS index")
    metadata = [
        {
            "chunk_id": c["chunk_id"],
            "file_name": c["file_name"],
            "page": c["page"],
            "text": c["text"],
        }
        for c in chunks
    ]
    build_and_save(embeddings, metadata)

    elapsed = time.perf_counter() - t0
    logger.info(
        "═══ Done!  %d PDFs → %d pages → %d chunks → FAISS index  (%.1f s)",
        len({c["file_name"] for c in chunks}),
        len(pages),
        len(chunks),
        elapsed,
    )
    logger.info("Vector store written to: %s", VECTOR_STORE_DIR)


if __name__ == "__main__":
    main()
