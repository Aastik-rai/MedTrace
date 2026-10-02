#!/usr/bin/env python3
"""Stage 4 smoke test: PDF -> chunks -> Gemini -> validated facts.

Usage::

    set GEMINI_API_KEY=...           # see .env.example
    python scripts/test_gemini_provider.py path/to/synthetic_report.pdf
    python scripts/test_gemini_provider.py report.pdf --model gemini-2.5-flash -o facts.json

Runs the real pipeline: extraction (Stage 1), chunking (Stage 2), one Gemini
request per chunk (Stage 4), and Stage 3 validation of every candidate.

For the hackathon demo, use synthetic medical records only.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make the repository root importable when this file is run directly.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ai import (  # noqa: E402
    PDFExtractionError,
    chunk_extraction,
    extract_facts,
    extract_pdf_text,
    save_extraction_json,
)
from ai.fact_extractor import FactValidationError  # noqa: E402
from ai.providers import GeminiFactExtractionProvider, GeminiProviderError  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    """Run the Stage 4 pipeline on a PDF.

    Returns:
        Process exit code: 0 on success, 1 on a handled error.
    """
    parser = argparse.ArgumentParser(
        description="Extract validated medical facts from a PDF using Gemini."
    )
    parser.add_argument("pdf_path", help="Path to a local (synthetic) PDF file")
    parser.add_argument(
        "--model", default=None, help="Gemini model name (default: gemini-2.5-flash)"
    )
    parser.add_argument("--chunk-size", type=int, default=1000)
    parser.add_argument("--overlap", type=int, default=200)
    parser.add_argument(
        "-o", "--output", default=None, help="Optional JSON file for validated facts"
    )
    args = parser.parse_args(argv)

    try:
        extraction = extract_pdf_text(args.pdf_path)
        chunks = chunk_extraction(
            extraction, chunk_size=args.chunk_size, overlap=args.overlap
        )
        # Reads GEMINI_API_KEY from the environment; raises a clear error
        # if it is missing. The key is never printed.
        provider = GeminiFactExtractionProvider(
            model=args.model, document=extraction["document"]
        )
        facts = extract_facts(chunks, provider)
    except (
        ValueError,
        FileNotFoundError,
        IsADirectoryError,
        PDFExtractionError,
        GeminiProviderError,
        FactValidationError,
    ) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Document: {extraction['document']}")
    print(f"Model: {provider.model}")
    print(f"Chunks processed: {chunks['total_chunks']}")
    print(f"Facts extracted: {len(facts)} (all validated against source chunks)")

    for index, fact in enumerate(facts, start=1):
        print(f"\n{index}. [{fact.fact_type}] {fact.name} (status={fact.status})")
        if fact.value:
            unit = f" {fact.unit}" if fact.unit else ""
            print(f"   Value: {fact.value}{unit}")
        print(
            f"   Evidence: {fact.evidence.document} "
            f"p.{fact.evidence.page_number} ({fact.evidence.chunk_id})"
        )
        print(f'   Quote: "{fact.evidence.quote}"')

    if args.output:
        result = {
            "document": extraction["document"],
            "model": provider.model,
            "chunks_processed": chunks["total_chunks"],
            "facts": [fact.to_dict() for fact in facts],
        }
        saved_path = save_extraction_json(result, args.output)
        print(f"\nSaved facts to: {saved_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
