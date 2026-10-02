"""Basic tests for the MedTrace PDF processor.

Run from the repository root::

    python -m unittest discover -s tests -v
    # or
    pytest tests/test_pdf_processor.py

The tests only use generic placeholder text. No medical claims or synthetic
patient data are created here.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

# Make the repository root importable regardless of the runner used.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pymupdf  # PyMuPDF (the old ``fitz`` module name is deprecated)

from ai.pdf_processor import (  # noqa: E402
    PDFExtractionError,
    extract_pdf_text,
    save_extraction_json,
)

PAGE_TEXTS = [
    "Sample test page one. Generic placeholder text for unit testing.",
    "Sample test page two. Second generic placeholder line.",
    "Sample test page three. Third generic placeholder line.",
]


def create_test_pdf(path: Path, page_texts: list[str]) -> None:
    """Create a small PDF containing generic test text.

    Args:
        path: Destination file path.
        page_texts: One string of text to place on each new page.
    """
    document = pymupdf.open()
    for text in page_texts:
        page = document.new_page()
        page.insert_text((72, 72), text)
    document.save(str(path))
    document.close()


class TestExtractPdfText(unittest.TestCase):
    """Tests for :func:`extract_pdf_text`."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp_dir = Path(self._tmp.name)
        self.pdf_path = self.tmp_dir / "sample_report.pdf"
        create_test_pdf(self.pdf_path, PAGE_TEXTS)

    def test_returns_expected_structure(self) -> None:
        result = extract_pdf_text(str(self.pdf_path))

        self.assertIsInstance(result, dict)
        self.assertEqual(result["document"], "sample_report.pdf")
        self.assertEqual(result["total_pages"], 3)
        self.assertIsInstance(result["pages"], list)
        self.assertEqual(len(result["pages"]), 3)
        for page in result["pages"]:
            self.assertEqual(set(page.keys()), {"page_number", "text"})
            self.assertIsInstance(page["page_number"], int)
            self.assertIsInstance(page["text"], str)

    def test_page_numbers_are_preserved(self) -> None:
        result = extract_pdf_text(str(self.pdf_path))

        page_numbers = [page["page_number"] for page in result["pages"]]
        self.assertEqual(page_numbers, [1, 2, 3])

    def test_total_page_count_is_correct(self) -> None:
        result = extract_pdf_text(str(self.pdf_path))

        self.assertEqual(result["total_pages"], len(PAGE_TEXTS))
        self.assertEqual(result["total_pages"], len(result["pages"]))

    def test_pages_remain_separate_records(self) -> None:
        result = extract_pdf_text(str(self.pdf_path))

        # Every page holds its own text...
        for page, expected_text in zip(result["pages"], PAGE_TEXTS):
            self.assertIn(expected_text, page["text"])

        # ...and no page is merged with text from another page.
        self.assertNotIn("page two", result["pages"][0]["text"])
        self.assertNotIn("page one", result["pages"][1]["text"])

    def test_empty_path_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            extract_pdf_text("")

    def test_missing_file_is_rejected(self) -> None:
        missing = self.tmp_dir / "does_not_exist.pdf"
        with self.assertRaises(FileNotFoundError):
            extract_pdf_text(str(missing))

    def test_directory_path_is_rejected(self) -> None:
        with self.assertRaises(IsADirectoryError):
            extract_pdf_text(str(self.tmp_dir))

    def test_corrupt_file_raises_extraction_error(self) -> None:
        corrupt = self.tmp_dir / "corrupt.pdf"
        corrupt.write_bytes(b"this is definitely not a PDF file")

        with self.assertRaises(PDFExtractionError):
            extract_pdf_text(str(corrupt))


class TestSaveExtractionJson(unittest.TestCase):
    """Tests for :func:`save_extraction_json`."""

    def test_round_trip_preserves_page_structure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            pdf_path = tmp_path / "sample_report.pdf"
            create_test_pdf(pdf_path, PAGE_TEXTS)

            extraction = extract_pdf_text(str(pdf_path))
            output_path = tmp_path / "nested" / "output.json"
            saved_path = save_extraction_json(extraction, str(output_path))

            self.assertTrue(Path(saved_path).is_file())
            with Path(saved_path).open(encoding="utf-8") as handle:
                loaded = json.load(handle)

        self.assertEqual(loaded, extraction)
        self.assertEqual(loaded["total_pages"], 3)
        self.assertEqual(
            [page["page_number"] for page in loaded["pages"]], [1, 2, 3]
        )

    def test_empty_output_path_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            save_extraction_json({"pages": []}, "")


if __name__ == "__main__":
    unittest.main()
