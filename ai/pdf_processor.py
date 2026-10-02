"""PDF text extraction for the MedTrace AI module.

This is the first stage of the MedTrace evidence pipeline::

    PDF file -> per-page text -> (later) structured medical facts -> traceable evidence

Design rule: pages are NEVER merged into a single text blob. Each page is
returned as its own record so that, in later stages, every extracted medical
fact can be traced back to:

    document -> page number -> exact source text

This module deliberately contains no LLM calls, OCR, vector search, or web
API code. It only extracts and preserves document text.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pymupdf  # PyMuPDF (the old ``fitz`` module name is deprecated)


class PDFExtractionError(Exception):
    """Raised when a file exists but its PDF text cannot be extracted."""


def _validate_pdf_path(pdf_path: str) -> Path:
    """Validate and normalise an incoming PDF path.

    Args:
        pdf_path: Path to a local PDF file.

    Returns:
        The validated path as a :class:`pathlib.Path`.

    Raises:
        ValueError: If ``pdf_path`` is not a non-empty string.
        FileNotFoundError: If nothing exists at ``pdf_path``.
        IsADirectoryError: If ``pdf_path`` points to a directory.
    """
    if not isinstance(pdf_path, str) or not pdf_path.strip():
        raise ValueError("pdf_path must be a non-empty string")

    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    if path.is_dir():
        raise IsADirectoryError(f"Expected a PDF file but found a directory: {pdf_path}")
    return path


def extract_pdf_text(pdf_path: str) -> dict[str, Any]:
    """Extract text from a PDF file while preserving page boundaries.

    Each page is kept as a separate record. Pages are never concatenated, so
    downstream stages can always trace a fact back to one specific page of
    one specific document.

    Args:
        pdf_path: Path to a local PDF file.

    Returns:
        A structured extraction dictionary::

            {
                "document": "medical_report.pdf",
                "source_path": "/abs/path/to/medical_report.pdf",
                "total_pages": 3,
                "pages": [
                    {"page_number": 1, "text": "..."},
                    {"page_number": 2, "text": "..."},
                    {"page_number": 3, "text": "..."},
                ]
            }

    Raises:
        ValueError: If ``pdf_path`` is empty or not a string.
        FileNotFoundError: If the file does not exist.
        IsADirectoryError: If ``pdf_path`` is a directory.
        PDFExtractionError: If the file cannot be opened or read as a PDF
            (for example a corrupt file or a password-protected document).
    """
    path = _validate_pdf_path(pdf_path)

    try:
        document = pymupdf.open(str(path))
    except Exception as exc:  # PyMuPDF raises several error types here
        raise PDFExtractionError(f"Could not open PDF '{pdf_path}': {exc}") from exc

    try:
        if document.needs_pass:
            raise PDFExtractionError(f"PDF is password-protected: {pdf_path}")

        pages: list[dict[str, Any]] = []
        for index in range(document.page_count):
            page = document.load_page(index)
            # Raw text is preserved verbatim; no cleanup, so exact source
            # snippets remain quoteable later.
            pages.append(
                {
                    "page_number": index + 1,
                    "text": page.get_text("text"),
                }
            )
    except PDFExtractionError:
        raise
    except Exception as exc:
        raise PDFExtractionError(
            f"Failed to extract text from '{pdf_path}': {exc}"
        ) from exc
    finally:
        document.close()

    return {
        "document": path.name,
        "source_path": str(path.resolve()),
        "total_pages": len(pages),
        "pages": pages,
    }


def save_extraction_json(extraction: dict[str, Any], output_path: str) -> str:
    """Save an extraction result to a JSON file.

    The page-level structure is written to disk unchanged, so the evidence
    link (document -> page number -> exact text) survives the round trip.

    Args:
        extraction: A result dictionary from :func:`extract_pdf_text`.
        output_path: Destination file path. Missing parent directories are
            created automatically.

    Returns:
        The absolute path of the written file, as a string.

    Raises:
        ValueError: If ``output_path`` is empty or not a string.
    """
    if not isinstance(output_path, str) or not output_path.strip():
        raise ValueError("output_path must be a non-empty string")

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(extraction, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return str(destination.resolve())
