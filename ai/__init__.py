"""MedTrace AI and document-processing package.

Public API for the first five ingestion stages::

    from ai import (
        extract_pdf_text,
        save_extraction_json,
        chunk_extraction,
        MedicalFact,
        Evidence,
        extract_facts,
        PatientRecord,
        search_facts,
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
from .patient_record import (
    PatientRecord,
    PatientRecordError,
    get_facts_by_name,
    get_facts_by_type,
    get_facts_for_document,
    sort_facts_chronologically,
)
from .retrieval import RetrievalError, find_condition_history, search_facts

__all__ = [
    "SUPPORTED_FACT_TYPES",
    "SUPPORTED_STATUSES",
    "Evidence",
    "FactExtractionProvider",
    "FactValidationError",
    "MedicalFact",
    "PDFExtractionError",
    "PatientRecord",
    "PatientRecordError",
    "RetrievalError",
    "chunk_extraction",
    "extract_facts",
    "extract_pdf_text",
    "facts_to_json",
    "find_condition_history",
    "get_facts_by_name",
    "get_facts_by_type",
    "get_facts_for_document",
    "save_extraction_json",
    "search_facts",
    "sort_facts_chronologically",
    "validate_fact",
]
