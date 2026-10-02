"""MedTrace AI and document-processing package.

Public API for the first two ingestion stages::

    from ai import extract_pdf_text, save_extraction_json, chunk_extraction
"""

from .chunker import chunk_extraction
from .pdf_processor import PDFExtractionError, extract_pdf_text, save_extraction_json

__all__ = [
    "PDFExtractionError",
    "chunk_extraction",
    "extract_pdf_text",
    "save_extraction_json",
]
