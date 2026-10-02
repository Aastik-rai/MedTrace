#!/usr/bin/env python3
"""Command-line smoke test for the MedTrace PDF processor.

Usage::

    python scripts/test_pdf_processor.py path/to/document.pdf
    python scripts/test_pdf_processor.py path/to/document.pdf -o output.json

Prints a per-page summary (document name, page count, extracted characters
per page) and saves the full page-level extraction as JSON so it can be
inspected or handed to later pipeline stages.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make the repository root importable when this file is run directly.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ai.pdf_processor import (  # noqa: E402
    PDFExtractionError,
    extract_pdf_text,
    save_extraction_json,
)

DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "extracted"


def build_output_path(pdf_path: Path) -> Path:
    """Return the default JSON output path for a given PDF file."""
    return DEFAULT_OUTPUT_DIR / f"{pdf_path.stem}.extracted.json"


def main(argv: list[str] | None = None) -> int:
    """Run the command-line smoke test.

    Args:
        argv: Argument list (defaults to ``sys.argv[1:]``).

    Returns:
        Process exit code: 0 on success, 1 on a handled error.
    """
    parser = argparse.ArgumentParser(
        description="Extract text from a PDF and save it as structured JSON."
    )
    parser.add_argument("pdf_path", help="Path to a local PDF file")
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Destination JSON path (default: data/extracted/<name>.extracted.json)",
    )
    args = parser.parse_args(argv)

    try:
        extraction = extract_pdf_text(args.pdf_path)
    except (ValueError, FileNotFoundError, IsADirectoryError, PDFExtractionError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    output_path = Path(args.output) if args.output else build_output_path(Path(args.pdf_path))
    saved_path = save_extraction_json(extraction, str(output_path))

    print(f"Document: {extraction['document']}")
    print(f"Pages: {extraction['total_pages']}")
    for page in extraction["pages"]:
        print(f"\nPage {page['page_number']}:")
        print(f"[{len(page['text'])} characters]")
    print(f"\nSaved extraction to: {saved_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
