"""
PDF Loader
==========
Scans the configured PDF folder, extracts text page-by-page using PyMuPDF
(fitz).  Returns a flat list of page records with metadata.

PyMuPDF is chosen because:
 - Pure C extension — fast even on modest CPUs
 - Handles malformed / password-protected PDFs gracefully
 - Extracts embedded text and OCR-friendly layout
"""

import logging
from pathlib import Path
from typing import List, TypedDict

import fitz  # PyMuPDF

from app.core.config import PDF_FOLDER_PATH

logger = logging.getLogger(__name__)


class PageRecord(TypedDict):
    file_name: str
    page: int        # 1-indexed page number
    text: str


def load_pdfs(pdf_folder: str | None = None) -> List[PageRecord]:
    """
    Scan *pdf_folder* for ``*.pdf`` files and extract text per page.

    Returns
    -------
    list[PageRecord]
        Each entry contains the source file name, page number, and raw text.
    """
    folder = Path(pdf_folder or PDF_FOLDER_PATH)
    if not folder.exists():
        logger.error("PDF folder does not exist: %s", folder)
        raise FileNotFoundError(f"PDF folder not found: {folder}")

    pdf_files = sorted(folder.glob("*.pdf"))
    if not pdf_files:
        logger.warning("No PDF files found in %s", folder)
        return []

    logger.info("Found %d PDF files in %s", len(pdf_files), folder)
    records: List[PageRecord] = []

    for pdf_path in pdf_files:
        try:
            doc = fitz.open(str(pdf_path))
        except Exception as exc:
            logger.warning("Skipping unreadable PDF %s: %s", pdf_path.name, exc)
            continue

        for page_num in range(len(doc)):
            try:
                page = doc[page_num]
                text = page.get_text("text")
                # Skip pages with no meaningful text
                if text and text.strip():
                    records.append(
                        PageRecord(
                            file_name=pdf_path.name,
                            page=page_num + 1,  # 1-indexed
                            text=text.strip(),
                        )
                    )
            except Exception as exc:
                logger.warning(
                    "Error extracting page %d of %s: %s",
                    page_num + 1,
                    pdf_path.name,
                    exc,
                )
        logger.info("  ✓ %s — %d pages extracted", pdf_path.name, len(doc))
        doc.close()

    logger.info("Total page records extracted: %d", len(records))
    return records
