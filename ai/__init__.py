"""MedTrace AI and document-processing package.

Public API for the first three ingestion stages::

    from ai import (
        extract_pdf_text,
        save_extraction_json,
        chunk_extraction,
        MedicalFact,
        Evidence,
        extract_facts,
    )
"""

from .chunker import chunk_extraction
from .fact_extractor import (
    SUPPORTED_FACT_TYPES,
    SUPPORTED_STATUSES,
    Evidence,
    FactExtractionProvider,
    FactValidationError,
    MedicalFact,
    extract_facts,
    facts_to_json,
    validate_fact,
)
from .pdf_processor import PDFExtractionError, extract_pdf_text, save_extraction_json

__all__ = [
    "SUPPORTED_FACT_TYPES",
    "SUPPORTED_STATUSES",
    "Evidence",
    "FactExtractionProvider",
    "FactValidationError",
    "MedicalFact",
    "PDFExtractionError",
    "chunk_extraction",
    "extract_facts",
    "extract_pdf_text",
    "facts_to_json",
    "save_extraction_json",
    "validate_fact",
]
