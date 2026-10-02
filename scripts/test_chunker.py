#!/usr/bin/env python3
"""Command-line smoke test for the MedTrace chunker.

Usage::

    python scripts/test_chunker.py path/to/document.pdf
    python scripts/test_chunker.py path/to/document.pdf --chunk-size 800 --overlap 150
    python scripts/test_chunker.py path/to/document.pdf -o chunks.json

Extracts the PDF with the existing PDF processor, chunks the pages with the
evidence-preserving chunker, prints a summary, and optionally saves the chunk
JSON. The existing PDF CLI (scripts/test_pdf_processor.py) is unaffected.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make the repository root importable when this file is run directly.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ai.chunker import chunk_extraction  # noqa: E402
from ai.pdf_processor import (  # noqa: E402
    PDFExtractionError,
    extract_pdf_text,
    save_extraction_json,
)


def main(argv: list[str] | None = None) -> int:
    """Run the chunker smoke test.

    Args:
        argv: Argument list (defaults to ``sys.argv[1:]``).

    Returns:
        Process exit code: 0 on success, 1 on a handled error.
    """
    parser = argparse.ArgumentParser(
        description="Extract a PDF and split each page into overlapping chunks."
    )
    parser.add_argument("pdf_path", help="Path to a local PDF file")
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1000,
        help="Maximum characters per chunk (default: 1000)",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=200,
        help="Characters shared between consecutive chunks (default: 200)",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Optional destination JSON path for the chunk output",
    )
    args = parser.parse_args(argv)

    try:
        extraction = extract_pdf_text(args.pdf_path)
        result = chunk_extraction(
            extraction,
            chunk_size=args.chunk_size,
            overlap=args.overlap,
        )
    except (ValueError, FileNotFoundError, IsADirectoryError, PDFExtractionError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    chunks_per_page: dict[int, int] = {}
    for chunk in result["chunks"]:
        page_number = chunk["page_number"]
        chunks_per_page[page_number] = chunks_per_page.get(page_number, 0) + 1

    print(f"Document: {result['document']}")
    print(f"Total pages: {extraction['total_pages']}")
    print(f"Total chunks: {result['total_chunks']}")
    print(f"Chunk size: {result['chunk_size']} | Overlap: {result['overlap']}")
    print("\nChunks per page:")
    for page_number in range(1, extraction["total_pages"] + 1):
        count = chunks_per_page.get(page_number, 0)
        print(f"  Page {page_number}: {count} chunk(s)")

    if args.output:
        saved_path = save_extraction_json(result, args.output)
        print(f"\nSaved chunks to: {saved_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
