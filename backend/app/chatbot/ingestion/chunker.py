"""
Text Chunker
=============
Splits page records into overlapping text chunks of a controlled size.

Design notes for hardware optimisation:
 - 500-char chunks (≈100 tokens) are small enough that the embedding model
   processes them in ~1 ms each and the FAISS index stays compact.
 - 50-char overlap prevents sentences from being cut mid-idea.
 - Each chunk keeps its source metadata so citations can be returned.
"""

import logging
import uuid
from typing import List, TypedDict

from app.core.config import CHUNK_SIZE, CHUNK_OVERLAP

logger = logging.getLogger(__name__)


class ChunkRecord(TypedDict):
    chunk_id: str
    file_name: str
    page: int
    text: str


def chunk_pages(
    page_records: list,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> List[ChunkRecord]:
    """
    Split page texts into overlapping chunks.

    Parameters
    ----------
    page_records : list[PageRecord]
        Output from ``pdf_loader.load_pdfs()``.
    chunk_size : int, optional
        Maximum characters per chunk (default from config).
    chunk_overlap : int, optional
        Character overlap between consecutive chunks (default from config).

    Returns
    -------
    list[ChunkRecord]
        Each entry is a chunk with a unique ID, source metadata, and text.
    """
    size = chunk_size or CHUNK_SIZE
    overlap = chunk_overlap or CHUNK_OVERLAP
    chunks: List[ChunkRecord] = []

    for rec in page_records:
        text = rec["text"]
        if len(text) <= size:
            chunks.append(
                ChunkRecord(
                    chunk_id=uuid.uuid4().hex[:12],
                    file_name=rec["file_name"],
                    page=rec["page"],
                    text=text,
                )
            )
            continue

        start = 0
        while start < len(text):
            end = start + size
            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append(
                    ChunkRecord(
                        chunk_id=uuid.uuid4().hex[:12],
                        file_name=rec["file_name"],
                        page=rec["page"],
                        text=chunk_text,
                    )
                )
            start += size - overlap

    logger.info(
        "Chunked %d page records → %d chunks  (size=%d, overlap=%d)",
        len(page_records),
        len(chunks),
        size,
        overlap,
    )
    return chunks
