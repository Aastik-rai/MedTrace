"""Provider-independent medical fact schema and evidence validation.

Stage 3 of the MedTrace evidence pipeline::

    PDF -> page text (ai.pdf_processor) -> evidence chunks (ai.chunker)
        -> fact extraction (this module) -> structured medical facts
        -> evidence references (document, page, chunk, quote)

Architectural rules:

- Facts are extracted *from document text*. Nothing in this module
  diagnoses, infers, or invents medical information.
- Every fact MUST carry evidence: ``document``, ``page_number``,
  ``chunk_id``, and a ``quote`` that is a verbatim substring of the source
  chunk (``quote in chunk_text``). Facts that cannot be supported by source
  text are rejected with :class:`FactValidationError`.
- Uncertain or historical statements keep their own status; they are never
  promoted to confirmed active diagnoses.
- No external LLM API is called here and no API key is required. The
  provider interface (:class:`FactExtractionProvider`) is what a future
  LLM adapter will implement.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import date
from typing import Any, Protocol

__all__ = [
    "SUPPORTED_DATE_PRECISIONS",
    "SUPPORTED_FACT_TYPES",
    "SUPPORTED_STATUSES",
    "Evidence",
    "FactExtractionProvider",
    "FactValidationError",
    "MedicalFact",
    "build_chunk_index",
    "extract_facts",
    "facts_to_json",
    "validate_fact",
    "validate_facts",
]

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------

SUPPORTED_FACT_TYPES: tuple[str, ...] = (
    "diagnosis",
    "medication",
    "lab_result",
    "procedure",
    "symptom",
    "allergy",
    "vital",
    "encounter",
    "condition_status",
)

SUPPORTED_STATUSES: tuple[str, ...] = (
    "active",
    "historical",
    "suspected",
    "resolved",
    "ruled_out",
    "unknown",
)

SUPPORTED_DATE_PRECISIONS: tuple[str, ...] = ("day", "month", "year", "unknown")

_DATE_PATTERNS: dict[str, re.Pattern[str]] = {
    "day": re.compile(r"^\d{4}-\d{2}-\d{2}$"),
    "month": re.compile(r"^\d{4}-\d{2}$"),
    "year": re.compile(r"^\d{4}$"),
}

EVIDENCE_FIELDS: tuple[str, ...] = ("document", "page_number", "chunk_id", "quote")
FACT_REQUIRED_FIELDS: tuple[str, ...] = ("fact_type", "name", "evidence")


class FactValidationError(ValueError):
    """Raised when a medical fact or its evidence fails validation."""


# --------------------------------------------------------------------------
# Data structures
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Evidence:
    """Traceable evidence linking a fact back to its source chunk.

    Attributes:
        document: Filename of the source document (``report.pdf``).
        page_number: 1-based page number within the document.
        chunk_id: ID of the source chunk (``report-p7-c2``).
        quote: Verbatim substring of the source chunk's text. Never a
            paraphrase: ``quote in chunk_text`` must hold.
    """

    document: str
    page_number: int
    chunk_id: str
    quote: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dictionary with a fixed field order."""
        return {
            "document": self.document,
            "page_number": self.page_number,
            "chunk_id": self.chunk_id,
            "quote": self.quote,
        }

    @classmethod
    def from_dict(cls, data: Any) -> "Evidence":
        """Build an :class:`Evidence` from a dictionary.

        Args:
            data: Dictionary with the evidence fields.

        Returns:
            The constructed evidence record.

        Raises:
            FactValidationError: If ``data`` is not a dictionary or misses
                any required field.
        """
        if not isinstance(data, dict):
            raise FactValidationError(
                f"evidence must be a dict, got {type(data).__name__}"
            )
        for field in EVIDENCE_FIELDS:
            if field not in data:
                raise FactValidationError(f"evidence is missing required field '{field}'")
        return cls(
            document=data["document"],
            page_number=data["page_number"],
            chunk_id=data["chunk_id"],
            quote=data["quote"],
        )


@dataclass(frozen=True)
class MedicalFact:
    """A single medical fact extracted from one document chunk.

    Optional fields stay ``None`` when the source text does not provide
    them; they are never fabricated.

    Attributes:
        fact_type: One of :data:`SUPPORTED_FACT_TYPES`.
        name: Primary label of the fact (e.g. a condition or test name).
        evidence: Where this fact came from (document/page/chunk/quote).
        value: Structured value for lab results, doses, measurements, etc.
        unit: Unit for ``value`` (e.g. ``%``, ``mg/dL``).
        frequency: Dosing frequency for medications (e.g. ``twice daily``).
        date: ISO date at the granularity of ``date_precision``.
        date_precision: ``day``, ``month``, ``year``, or ``unknown``.
        status: One of :data:`SUPPORTED_STATUSES`; defaults to ``unknown``
            so uncertain/historical text is never promoted to ``active``.
    """

    fact_type: str
    name: str
    evidence: Evidence
    value: str | None = None
    unit: str | None = None
    frequency: str | None = None
    date: str | None = None
    date_precision: str = "unknown"
    status: str = "unknown"

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dictionary with a fixed field order."""
        return {
            "fact_type": self.fact_type,
            "name": self.name,
            "value": self.value,
            "unit": self.unit,
            "frequency": self.frequency,
            "date": self.date,
            "date_precision": self.date_precision,
            "status": self.status,
            "evidence": self.evidence.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Any) -> "MedicalFact":
        """Build a :class:`MedicalFact` from a dictionary.

        Only required-field presence is checked here; semantic validation
        (supported fact type, quote verification, ...) happens in
        :func:`validate_fact`.

        Args:
            data: Dictionary with the fact fields.

        Returns:
            The constructed fact.

        Raises:
            FactValidationError: If ``data`` is not a dictionary, misses a
                required field, or ``evidence`` cannot be constructed.
        """
        if not isinstance(data, dict):
            raise FactValidationError(
                f"fact must be a dict, got {type(data).__name__}"
            )
        for field in FACT_REQUIRED_FIELDS:
            if field not in data:
                raise FactValidationError(f"fact is missing required field '{field}'")
        return cls(
            fact_type=data["fact_type"],
            name=data["name"],
            evidence=Evidence.from_dict(data["evidence"]),
            value=data.get("value"),
            unit=data.get("unit"),
            frequency=data.get("frequency"),
            date=data.get("date"),
            date_precision=data.get("date_precision", "unknown"),
            status=data.get("status", "unknown"),
        )


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------


def _require_non_empty_str(value: Any, label: str) -> None:
    """Raise :class:`FactValidationError` unless ``value`` is a non-empty string."""
    if not isinstance(value, str) or not value.strip():
        raise FactValidationError(f"{label} must be a non-empty string, got {value!r}")


def _validate_evidence(evidence: Evidence) -> None:
    """Validate the four mandatory evidence fields.

    Raises:
        FactValidationError: If any field is missing, empty, or malformed.
    """
    _require_non_empty_str(evidence.document, "evidence.document")
    if (
        not isinstance(evidence.page_number, int)
        or isinstance(evidence.page_number, bool)
        or evidence.page_number < 1
    ):
        raise FactValidationError(
            f"evidence.page_number must be a positive integer, got {evidence.page_number!r}"
        )
    _require_non_empty_str(evidence.chunk_id, "evidence.chunk_id")
    _require_non_empty_str(evidence.quote, "evidence.quote")


def _validate_date(date_value: Any, date_precision: Any) -> None:
    """Validate the date / date_precision pair without fabricating anything.

    Rules:
        - ``date=None`` requires precision ``unknown``.
        - A date string requires ``day``, ``month``, or ``year`` precision,
          and its format must match that precision.
        - Calendar-validity is checked for day precision.

    Raises:
        FactValidationError: On any mismatch or malformed value.
    """
    if date_precision not in SUPPORTED_DATE_PRECISIONS:
        raise FactValidationError(
            f"date_precision must be one of {SUPPORTED_DATE_PRECISIONS}, "
            f"got {date_precision!r}"
        )

    if date_value is None:
        if date_precision != "unknown":
            raise FactValidationError(
                f"date_precision '{date_precision}' was given without a date"
            )
        return

    _require_non_empty_str(date_value, "date")
    if date_precision == "unknown":
        raise FactValidationError(
            f"date '{date_value}' was given with date_precision 'unknown'; "
            "state the actual precision (day, month, year)"
        )

    pattern = _DATE_PATTERNS[date_precision]
    if not pattern.fullmatch(date_value):
        expected = {"day": "YYYY-MM-DD", "month": "YYYY-MM", "year": "YYYY"}[
            date_precision
        ]
        raise FactValidationError(
            f"date '{date_value}' does not match date_precision "
            f"'{date_precision}' (expected {expected})"
        )

    if date_precision == "day":
        try:
            date.fromisoformat(date_value)
        except ValueError as exc:
            raise FactValidationError(f"date '{date_value}' is not a valid calendar date") from exc
    elif date_precision == "month" and not 1 <= int(date_value[5:7]) <= 12:
        raise FactValidationError(f"date '{date_value}' has an invalid month")


def build_chunk_index(chunks: Any) -> dict[str, dict[str, Any]]:
    """Build a ``chunk_id -> chunk`` lookup for evidence verification.

    Args:
        chunks: Either a list of chunk records (as produced by
            :func:`ai.chunker.chunk_extraction`) or the whole dictionary
            returned by that function.

    Returns:
        A mapping from ``chunk_id`` to the full chunk record.

    Raises:
        ValueError: If ``chunks`` has the wrong shape, a chunk misses the
            ``chunk_id`` / ``page_number`` / ``text`` keys, or two chunks
            share the same ``chunk_id``.
    """
    chunk_list = _coerce_chunk_list(chunks)
    index: dict[str, dict[str, Any]] = {}
    for position, chunk in enumerate(chunk_list):
        if not isinstance(chunk, dict):
            raise ValueError(f"chunks[{position}] must be a dict, got {type(chunk).__name__}")
        missing = [key for key in ("chunk_id", "page_number", "text") if key not in chunk]
        if missing:
            raise ValueError(
                f"chunks[{position}] is missing required key(s): {', '.join(missing)}"
            )
        chunk_id = chunk["chunk_id"]
        if chunk_id in index:
            raise ValueError(f"duplicate chunk_id in source chunks: '{chunk_id}'")
        index[chunk_id] = chunk
    return index


def _coerce_chunk_list(chunks: Any) -> list[dict[str, Any]]:
    """Accept either a chunk list or a ``chunk_extraction()`` result."""
    if isinstance(chunks, list):
        return chunks
    if isinstance(chunks, dict) and isinstance(chunks.get("chunks"), list):
        return chunks["chunks"]
    raise ValueError(
        "chunks must be a list of chunk records or the dict returned by chunk_extraction()"
    )


def _verify_evidence_against_chunks(
    evidence: Evidence, source_chunks: dict[str, dict[str, Any]]
) -> None:
    """Verify a quote against the source chunk it cites.

    Checks, in order:
        1. The cited chunk exists in the index.
        2. ``quote`` is a verbatim substring of the chunk text.
        3. ``page_number`` matches the chunk's actual page.

    Raises:
        FactValidationError: If any check fails.
    """
    chunk = source_chunks.get(evidence.chunk_id)
    if chunk is None:
        raise FactValidationError(
            f"evidence.chunk_id '{evidence.chunk_id}' not found in source chunks"
        )
    if evidence.quote not in chunk["text"]:
        raise FactValidationError(
            f"evidence.quote is not a verbatim substring of chunk "
            f"'{evidence.chunk_id}'"
        )
    if evidence.page_number != chunk["page_number"]:
        raise FactValidationError(
            f"evidence.page_number ({evidence.page_number}) does not match the "
            f"page of chunk '{evidence.chunk_id}' ({chunk['page_number']})"
        )


def validate_fact(
    fact: "MedicalFact | dict[str, Any]",
    source_chunks: dict[str, dict[str, Any]] | None = None,
) -> MedicalFact:
    """Validate a medical fact and its evidence.

    Args:
        fact: A :class:`MedicalFact` or a plain dictionary of fact fields.
        source_chunks: Optional ``chunk_id -> chunk`` index (see
            :func:`build_chunk_index`). When provided, the evidence quote is
            verified verbatim against the source chunk text.

    Returns:
        The validated fact as a :class:`MedicalFact`.

    Raises:
        FactValidationError: If the fact type is unsupported, the name or
            evidence is missing/malformed, the status or date fields are
            invalid, or the quote cannot be verified against source chunks.
    """
    if isinstance(fact, dict):
        fact = MedicalFact.from_dict(fact)
    elif not isinstance(fact, MedicalFact):
        raise FactValidationError(
            f"fact must be a MedicalFact or dict, got {type(fact).__name__}"
        )

    if fact.fact_type not in SUPPORTED_FACT_TYPES:
        raise FactValidationError(
            f"unsupported fact_type '{fact.fact_type}'; "
            f"expected one of {SUPPORTED_FACT_TYPES}"
        )
    _require_non_empty_str(fact.name, "name")

    if not isinstance(fact.evidence, Evidence):
        raise FactValidationError(
            f"evidence must be an Evidence record, got {type(fact.evidence).__name__}"
        )
    _validate_evidence(fact.evidence)

    for field in ("value", "unit", "frequency"):
        field_value = getattr(fact, field)
        if field_value is not None:
            _require_non_empty_str(field_value, field)

    if fact.status not in SUPPORTED_STATUSES:
        raise FactValidationError(
            f"status must be one of {SUPPORTED_STATUSES}, got {fact.status!r}"
        )
    _validate_date(fact.date, fact.date_precision)

    if source_chunks is not None:
        _verify_evidence_against_chunks(fact.evidence, source_chunks)

    return fact


def validate_facts(
    facts: list["MedicalFact | dict[str, Any]"],
    source_chunks: dict[str, dict[str, Any]] | None = None,
) -> list[MedicalFact]:
    """Validate a list of facts in order.

    Args:
        facts: Facts as :class:`MedicalFact` objects or dictionaries.
        source_chunks: Optional chunk index for quote verification.

    Returns:
        The validated facts as :class:`MedicalFact` objects.

    Raises:
        FactValidationError: On the first invalid fact.
    """
    return [validate_fact(fact, source_chunks=source_chunks) for fact in facts]


# --------------------------------------------------------------------------
# Provider-independent extraction interface
# --------------------------------------------------------------------------


class FactExtractionProvider(Protocol):
    """Interface that future LLM (or rule-based) providers will implement.

    Implementations receive the evidence chunks (each with ``chunk_id``,
    ``page_number``, ``text``, ``start_char``, ``end_char``) and must return
    candidate facts whose ``evidence.quote`` values are copied verbatim from
    a chunk's text. Providers must extract only what the text states — never
    diagnose or invent information. Every candidate is verified by
    :func:`extract_facts` before it is returned to the caller.
    """

    def extract_facts(
        self, chunks: list[dict[str, Any]]
    ) -> list["MedicalFact | dict[str, Any]"]:
        """Return candidate facts for the given chunks."""
        ...


def extract_facts(chunks: Any, provider: FactExtractionProvider) -> list[MedicalFact]:
    """Run chunks through a fact-extraction provider and validate the results.

    This is the single entry point future LLM adapters plug into: the caller
    passes chunks from :func:`ai.chunker.chunk_extraction` and any object
    implementing :class:`FactExtractionProvider`. No API key is needed here —
    credentials belong to the provider implementation, not to this module.

    Args:
        chunks: A list of chunk records or a ``chunk_extraction()`` result.
        provider: Object implementing :class:`FactExtractionProvider`.

    Returns:
        Every candidate fact, validated against its source chunks.

    Raises:
        ValueError: If ``chunks`` has the wrong shape.
        FactValidationError: If the provider returns malformed facts,
            unsupported fact types, or quotes that cannot be verified
            against the source chunk text. Invalid facts are rejected, not
            silently accepted.
    """
    chunk_list = _coerce_chunk_list(chunks)
    source_chunks = build_chunk_index(chunk_list)

    candidates = provider.extract_facts(chunk_list)
    if candidates is None:
        return []
    if not isinstance(candidates, list):
        raise FactValidationError(
            f"provider must return a list of facts, got {type(candidates).__name__}"
        )

    facts: list[MedicalFact] = []
    for position, candidate in enumerate(candidates):
        if isinstance(candidate, MedicalFact):
            facts.append(candidate)
        elif isinstance(candidate, dict):
            facts.append(MedicalFact.from_dict(candidate))
        else:
            raise FactValidationError(
                f"provider returned invalid fact at position {position}: "
                f"expected MedicalFact or dict, got {type(candidate).__name__}"
            )
    return validate_facts(facts, source_chunks=source_chunks)


def facts_to_json(facts: list[MedicalFact]) -> str:
    """Serialize facts to deterministic, human-readable JSON.

    Args:
        facts: Validated facts.

    Returns:
        A JSON string; the same facts always produce the same output.
    """
    return json.dumps(
        [fact.to_dict() for fact in facts],
        ensure_ascii=False,
        indent=2,
    )
