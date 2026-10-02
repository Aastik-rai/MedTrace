"""Tests for the MedTrace evidence-preserving chunker.

Run from the repository root::

    python -m unittest discover -s tests -v
    # or
    pytest tests/test_chunker.py

All test text is generic placeholder content. No medical claims or
synthetic patient data are used.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

# Make the repository root importable regardless of the runner used.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ai.chunker import chunk_extraction  # noqa: E402

PAGE_PREFIXES = [
    "Page one contains sample text for chunking tests. ",
    "Page two contains sample text for chunking tests. ",
    "Page three contains sample text for chunking tests. ",
]


def build_text(prefix: str, size: int) -> str:
    """Build a generic test string of exactly ``size`` characters."""
    if size <= 0:
        return ""
    return (prefix * (size // len(prefix) + 1))[:size]


def make_extraction(page_texts: list[str], document: str = "sample_report.pdf") -> dict:
    """Build an extraction dict shaped like ``extract_pdf_text()`` output."""
    return {
        "document": document,
        "source_path": f"C:/data/{document}",
        "total_pages": len(page_texts),
        "pages": [
            {"page_number": index + 1, "text": text}
            for index, text in enumerate(page_texts)
        ],
    }


class TestChunkExtraction(unittest.TestCase):
    """Tests for :func:`ai.chunker.chunk_extraction`."""

    def test_basic_chunking(self) -> None:
        """A short page yields one chunk with the expected structure."""
        extraction = make_extraction([build_text(PAGE_PREFIXES[0], 500)])

        result = chunk_extraction(extraction)

        self.assertEqual(result["document"], "sample_report.pdf")
        self.assertEqual(result["total_chunks"], 1)
        self.assertEqual(len(result["chunks"]), 1)

        chunk = result["chunks"][0]
        self.assertEqual(
            set(chunk.keys()),
            {"chunk_id", "page_number", "text", "start_char", "end_char"},
        )
        self.assertEqual(chunk["chunk_id"], "sample_report-p1-c1")
        self.assertEqual(chunk["page_number"], 1)
        self.assertEqual(chunk["start_char"], 0)
        self.assertEqual(chunk["end_char"], 500)

    def test_multiple_chunks_on_a_page(self) -> None:
        """One long page is split into several sequential chunks."""
        text = build_text(PAGE_PREFIXES[0], 2500)
        extraction = make_extraction([text])

        result = chunk_extraction(extraction, chunk_size=1000, overlap=200)

        self.assertEqual(result["total_chunks"], 3)
        ids = [chunk["chunk_id"] for chunk in result["chunks"]]
        self.assertEqual(ids, [
            "sample_report-p1-c1",
            "sample_report-p1-c2",
            "sample_report-p1-c3",
        ])

    def test_page_boundaries_are_preserved(self) -> None:
        """Chunks never combine pages; each page restarts at c1."""
        page_texts = [
            build_text(PAGE_PREFIXES[0], 1500),
            build_text(PAGE_PREFIXES[1], 1500),
        ]
        extraction = make_extraction(page_texts)

        result = chunk_extraction(extraction, chunk_size=1000, overlap=200)

        # 1500 chars at 1000/200 -> spans 0-1000, 800-1500 -> 2 chunks each.
        self.assertEqual(result["total_chunks"], 4)
        self.assertEqual(
            [chunk["page_number"] for chunk in result["chunks"]],
            [1, 1, 2, 2],
        )
        self.assertEqual(
            [chunk["chunk_id"] for chunk in result["chunks"]],
            [
                "sample_report-p1-c1",
                "sample_report-p1-c2",
                "sample_report-p2-c1",
                "sample_report-p2-c2",
            ],
        )
        # Page 2 restarts its own chunk numbering.
        page_two_ids = [
            chunk["chunk_id"]
            for chunk in result["chunks"]
            if chunk["page_number"] == 2
        ]
        self.assertTrue(all("-p2-c" in chunk_id for chunk_id in page_two_ids))

    def test_chunk_size_is_respected(self) -> None:
        """No chunk exceeds chunk_size; full chunks are exactly chunk_size."""
        extraction = make_extraction([build_text(PAGE_PREFIXES[0], 2300)])

        result = chunk_extraction(extraction, chunk_size=1000, overlap=200)

        sizes = [
            chunk["end_char"] - chunk["start_char"]
            for chunk in result["chunks"]
        ]
        self.assertEqual(sizes, [1000, 1000, 700])
        for size in sizes:
            self.assertLessEqual(size, 1000)

        # A custom chunk size is honoured too.
        custom = chunk_extraction(extraction, chunk_size=500, overlap=50)
        self.assertEqual(custom["chunk_size"], 500)
        self.assertTrue(
            all(
                chunk["end_char"] - chunk["start_char"] <= 500
                for chunk in custom["chunks"]
            )
        )

    def test_overlap_between_consecutive_chunks(self) -> None:
        """Consecutive chunks on a page share exactly ``overlap`` chars."""
        extraction = make_extraction([build_text(PAGE_PREFIXES[0], 2500)])

        result = chunk_extraction(extraction, chunk_size=1000, overlap=200)

        for previous, current in zip(result["chunks"], result["chunks"][1:]):
            self.assertEqual(current["start_char"], previous["end_char"] - 200)
            self.assertEqual(
                previous["text"][-200:],
                current["text"][:200],
            )

        # Overlap is configurable: zero overlap means no shared characters.
        no_overlap = chunk_extraction(extraction, chunk_size=1000, overlap=0)
        for previous, current in zip(
            no_overlap["chunks"], no_overlap["chunks"][1:]
        ):
            self.assertEqual(current["start_char"], previous["end_char"])

    def test_character_offsets_are_valid(self) -> None:
        """Offsets are integers within the bounds of their own page text."""
        page_texts = [
            build_text(PAGE_PREFIXES[0], 2500),
            build_text(PAGE_PREFIXES[1], 900),
        ]
        extraction = make_extraction(page_texts)

        result = chunk_extraction(extraction, chunk_size=400, overlap=100)

        for chunk in result["chunks"]:
            page_text = page_texts[chunk["page_number"] - 1]
            self.assertIsInstance(chunk["start_char"], int)
            self.assertIsInstance(chunk["end_char"], int)
            self.assertGreaterEqual(chunk["start_char"], 0)
            self.assertLess(chunk["start_char"], chunk["end_char"])
            self.assertLessEqual(chunk["end_char"], len(page_text))

    def test_chunk_text_reconstructs_from_source_page_text(self) -> None:
        """page_text[start_char:end_char] equals the chunk text exactly."""
        page_texts = [
            build_text(PAGE_PREFIXES[0], 2100),
            build_text(PAGE_PREFIXES[1], 1300),
            build_text(PAGE_PREFIXES[2], 300),
        ]
        extraction = make_extraction(page_texts)

        result = chunk_extraction(extraction, chunk_size=700, overlap=150)

        self.assertGreater(result["total_chunks"], 0)
        for chunk in result["chunks"]:
            page_text = page_texts[chunk["page_number"] - 1]
            reconstructed = page_text[chunk["start_char"]:chunk["end_char"]]
            self.assertEqual(reconstructed, chunk["text"])

    def test_chunk_ids_are_deterministic(self) -> None:
        """The same input and settings always produce the same IDs."""
        extraction = make_extraction(
            [build_text(PAGE_PREFIXES[0], 2500), build_text(PAGE_PREFIXES[1], 700)]
        )

        first = chunk_extraction(extraction, chunk_size=800, overlap=100)
        second = chunk_extraction(extraction, chunk_size=800, overlap=100)

        first_ids = [chunk["chunk_id"] for chunk in first["chunks"]]
        second_ids = [chunk["chunk_id"] for chunk in second["chunks"]]
        self.assertEqual(first_ids, second_ids)

        # ID format: <document stem>-p<page>-c<n>
        expected_prefix = "sample_report-p1-c"
        self.assertTrue(first_ids[0].startswith(expected_prefix))

    def test_invalid_chunk_size_is_rejected(self) -> None:
        """Zero, negative, and non-integer chunk sizes raise ValueError."""
        extraction = make_extraction([build_text(PAGE_PREFIXES[0], 500)])

        for bad_size in (0, -100):
            with self.assertRaises(ValueError):
                chunk_extraction(extraction, chunk_size=bad_size, overlap=100)
        with self.assertRaises(ValueError):
            chunk_extraction(extraction, chunk_size="1000", overlap=100)  # type: ignore[arg-type]

    def test_invalid_overlap_is_rejected(self) -> None:
        """Negative overlap and overlap >= chunk_size raise ValueError."""
        extraction = make_extraction([build_text(PAGE_PREFIXES[0], 500)])

        with self.assertRaises(ValueError):
            chunk_extraction(extraction, chunk_size=1000, overlap=-1)
        with self.assertRaises(ValueError):
            chunk_extraction(extraction, chunk_size=1000, overlap=1000)
        with self.assertRaises(ValueError):
            chunk_extraction(extraction, chunk_size=1000, overlap=1500)
        with self.assertRaises(ValueError):
            chunk_extraction(extraction, chunk_size=1000, overlap="200")  # type: ignore[arg-type]

    def test_empty_and_missing_pages_are_handled(self) -> None:
        """Blank pages produce no chunks; missing/invalid pages raise."""
        # A page with empty text produces zero chunks.
        empty_page = make_extraction([""])
        result = chunk_extraction(empty_page)
        self.assertEqual(result["total_chunks"], 0)
        self.assertEqual(result["chunks"], [])

        # Whitespace-only pages are skipped as well.
        whitespace_page = make_extraction(["   \n\n  "])
        self.assertEqual(chunk_extraction(whitespace_page)["total_chunks"], 0)

        # A mixed extraction keeps only the non-empty page's chunks.
        mixed = make_extraction(["", build_text(PAGE_PREFIXES[1], 400), ""])
        self.assertEqual(chunk_extraction(mixed)["total_chunks"], 1)

        # An empty page list is valid and yields no chunks.
        no_pages = make_extraction([])
        self.assertEqual(chunk_extraction(no_pages)["total_chunks"], 0)

        # A missing 'pages' key is rejected.
        with self.assertRaises(ValueError):
            chunk_extraction({"document": "report.pdf"})

    def test_multiple_pages_are_chunked_independently(self) -> None:
        """Each page gets its own chunk sequence with correct counts."""
        page_texts = [
            build_text(PAGE_PREFIXES[0], 3000),   # 4 chunks at 1000/200
            build_text(PAGE_PREFIXES[1], 1000),   # 1 chunk
            build_text(PAGE_PREFIXES[2], 100),    # 1 chunk
        ]
        extraction = make_extraction(page_texts)

        result = chunk_extraction(extraction, chunk_size=1000, overlap=200)

        per_page: dict[int, int] = {}
        for chunk in result["chunks"]:
            per_page[chunk["page_number"]] = per_page.get(chunk["page_number"], 0) + 1

        self.assertEqual(result["total_chunks"], 6)
        self.assertEqual(per_page, {1: 4, 2: 1, 3: 1})
        # Chunk 1 of every page starts at offset 0.
        for chunk in result["chunks"]:
            if chunk["chunk_id"].endswith("-c1"):
                self.assertEqual(chunk["start_char"], 0)


if __name__ == "__main__":
    unittest.main()
