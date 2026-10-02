"""Evidence-preserving text chunking for the MedTrace AI module.

Stage 2 of the MedTrace evidence pipeline::

    PDF -> per-page text (ai.pdf_processor) -> per-page chunks (this module)
        -> (later) medical facts -> traceable evidence

Design rules:

- Chunks never span page boundaries: every chunk belongs to exactly one page.
- Chunk text is an exact substring of its source page text, so
  ``page_text[start_char:end_char] == chunk["text"]`` always holds.
- Chunk IDs are deterministic: ``<document stem>-p<page>-c<n>``, where ``n``
  restarts at 1 for every page.
- Nothing is normalized, rewritten, summarized, or paraphrased: chunking is
  a purely structural operation.

No external chunking library is used — only standard Python.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def _validate_extraction(extraction: Any) -> None:
    """Validate the top-level extraction structure.

    Args:
        extraction: Result returned by :func:`ai.pdf_processor.extract_pdf_text`.

    Raises:
        ValueError: If ``extraction`` is not a dict or has no ``pages`` list.
    """
    if not isinstance(extraction, dict):
        raise ValueError("extraction must be a dict returned by extract_pdf_text()")
    if "pages" not in extraction:
        raise ValueError("extraction is missing the required 'pages' key")
    if not isinstance(extraction["pages"], list):
        raise ValueError("extraction['pages'] must be a list of page records")


def _validate_settings(chunk_size: int, overlap: int) -> None:
    """Validate chunk size and overlap settings.

    Args:
        chunk_size: Maximum number of characters per chunk.
        overlap: Number of characters shared between consecutive chunks.

    Raises:
        ValueError: If ``chunk_size`` is not a positive integer, or
            ``overlap`` is negative or not smaller than ``chunk_size``.
    """
    if not isinstance(chunk_size, int) or isinstance(chunk_size, bool):
        raise ValueError(f"chunk_size must be an integer, got {chunk_size!r}")
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be greater than 0, got {chunk_size}")

    if not isinstance(overlap, int) or isinstance(overlap, bool):
        raise ValueError(f"overlap must be an integer, got {overlap!r}")
    if overlap < 0:
        raise ValueError(f"overlap must be greater than or equal to 0, got {overlap}")
    if overlap >= chunk_size:
        raise ValueError(
            f"overlap must be smaller than chunk_size, got overlap={overlap} "
            f"and chunk_size={chunk_size}"
        )


def _page_fields(page: Any, position: int) -> tuple[int, str]:
    """Extract and validate the fields of a single page record.

    Args:
        page: A page record from ``extraction["pages"]``.
        position: Index of the record, used in error messages.

    Returns:
        The page number and its raw text.

    Raises:
        ValueError: If the record is not a dict or lacks valid
            ``page_number`` / ``text`` fields.
    """
    if not isinstance(page, dict):
        raise ValueError(f"pages[{position}] must be a dict, got {type(page).__name__}")
    if "page_number" not in page or "text" not in page:
        raise ValueError(f"pages[{position}] must contain 'page_number' and 'text'")

    page_number = page["page_number"]
    text = page["text"]
    if not isinstance(page_number, int) or isinstance(page_number, bool):
        raise ValueError(
            f"pages[{position}].page_number must be an integer, got {page_number!r}"
        )
    if not isinstance(text, str):
        raise ValueError(f"pages[{position}].text must be a string, got {type(text).__name__}")
    return page_number, text


def _document_stem(extraction: dict[str, Any]) -> str:
    """Return the identifier used at the start of every chunk ID.

    Args:
        extraction: The extraction dictionary.

    Returns:
        The document filename without its extension (``report.pdf`` ->
        ``report``), or ``"document"`` if the name is missing or empty.
    """
    document = extraction.get("document", "")
    if not isinstance(document, str) or not document.strip():
        return "document"
    return Path(document).stem or "document"


def _chunk_spans(text_length: int, chunk_size: int, overlap: int) -> list[tuple[int, int]]:
    """Compute the ``(start_char, end_char)`` spans for one page.

    Spans advance by ``chunk_size - overlap`` characters, so consecutive
    chunks share ``overlap`` characters of context.

    Args:
        text_length: Length of the page text in characters.
        chunk_size: Maximum characters per chunk.
        overlap: Characters shared between consecutive chunks.

    Returns:
        A list of ``(start, end)`` pairs covering the text in order.
    """
    spans: list[tuple[int, int]] = []
    start = 0
    while start < text_length:
        end = min(start + chunk_size, text_length)
        spans.append((start, end))
        if end >= text_length:
            break
        start = end - overlap  # guaranteed to advance: overlap < chunk_size
    return spans


def chunk_extraction(
    extraction: dict,
    chunk_size: int = 1000,
    overlap: int = 200,
) -> dict[str, Any]:
    """Split each page's text into overlapping, evidence-preserving chunks.

    The input is the dictionary returned by
    :func:`ai.pdf_processor.extract_pdf_text`; this function never re-reads
    the PDF. Every chunk keeps its document, page number, character offsets,
    and an exact copy of the source text at those offsets, so a later
    AI-extracted fact can always be traced back to:

        document -> page number -> exact source text -> exact character range

    Chunks never combine pages: each page starts its own chunk sequence.

    Args:
        extraction: Output of :func:`ai.pdf_processor.extract_pdf_text`.
        chunk_size: Maximum number of characters per chunk. Must be > 0.
        overlap: Characters shared between consecutive chunks to avoid
            losing context at boundaries. Must be >= 0 and < ``chunk_size``.

    Returns:
        A dictionary::

            {
                "document": "hospital_record.pdf",
                "chunk_size": 1000,
                "overlap": 200,
                "total_chunks": 5,
                "chunks": [
                    {
                        "chunk_id": "hospital_record-p1-c1",
                        "page_number": 1,
                        "text": "...",
                        "start_char": 0,
                        "end_char": 800
                    },
                    ...
                ]
            }

    Raises:
        ValueError: If the extraction structure or the chunk settings are
            invalid.
    """
    _validate_extraction(extraction)
    _validate_settings(chunk_size, overlap)

    document_stem = _document_stem(extraction)
    chunks: list[dict[str, Any]] = []

    for position, page in enumerate(extraction["pages"]):
        page_number, text = _page_fields(page, position)

        # Pages with no extractable text produce no chunks.
        if not text.strip():
            continue

        spans = _chunk_spans(len(text), chunk_size, overlap)
        for index, (start, end) in enumerate(spans, start=1):
            chunks.append(
                {
                    "chunk_id": f"{document_stem}-p{page_number}-c{index}",
                    "page_number": page_number,
                    "text": text[start:end],
                    "start_char": start,
                    "end_char": end,
                }
            )

    return {
        "document": extraction.get("document", ""),
        "chunk_size": chunk_size,
        "overlap": overlap,
        "total_chunks": len(chunks),
        "chunks": chunks,
    }
