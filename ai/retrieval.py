"""Stage 6: deterministic, evidence-preserving fact retrieval.

Pipeline position::

    MedicalFact[] -> PatientRecord (Stage 5) -> retrieval (this module)

Core principles:

- **Retrieval only, never creation.** This layer never constructs a
  ``MedicalFact`` and never touches evidence. It only returns references
  to facts that already passed Stage 3 validation, with ``document``,
  ``page_number``, ``chunk_id``, and ``quote`` exactly as stored.
- **Deterministic matching.** Case-insensitive substring matching over
  plain fact fields after ``strip()`` + ``casefold()``. No LLM, no
  embeddings, no synonym dictionaries (``"HTN"`` does NOT match
  ``"hypertension"``), no fuzzy matching, no medical interpretation.
- **Record ordering.** Results keep the ``PatientRecord``'s chronological
  order (known dates before unknown dates, stable ties) — no re-ranking.
- **No medical conclusions.** ``find_condition_history`` returns matching
  facts with their original statuses; it never converts them to a boolean
  and never infers anything from an empty result.
"""

from __future__ import annotations

from typing import Any

from .fact_extractor import SUPPORTED_FACT_TYPES, MedicalFact
from .patient_record import PatientRecord

__all__ = [
    "CONDITION_FACT_TYPES",
    "RetrievalError",
    "SEARCHABLE_FIELDS",
    "find_condition_history",
    "search_facts",
]

# Fact types that describe a condition (existing Stage 3 types only).
CONDITION_FACT_TYPES: tuple[str, ...] = ("diagnosis", "condition_status")

# Scalar fact fields searched by search_facts(). Dates are deliberately
# excluded: they are structured metadata, better served by date-range
# queries than substring matching.
SEARCHABLE_FIELDS: tuple[str, ...] = (
    "name",
    "value",
    "unit",
    "frequency",
    "status",
    "fact_type",
)


class RetrievalError(ValueError):
    """Raised when retrieval inputs are invalid (query, type, limit, record)."""


def _normalize(value: Any) -> str:
    """Normalize a searchable value: ``None`` -> ``""``, else str, strip, casefold."""
    if value is None:
        return ""
    if not isinstance(value, str):
        value = str(value)
    return value.strip().casefold()


def _require_record(record: Any) -> None:
    """Raise :class:`RetrievalError` unless ``record`` is a PatientRecord."""
    if not isinstance(record, PatientRecord):
        raise RetrievalError(
            f"record must be a PatientRecord, got {type(record).__name__}"
        )


def _require_query(value: Any, label: str) -> str:
    """Validate a query string and return it normalized (strip + casefold)."""
    if not isinstance(value, str) or not value.strip():
        raise RetrievalError(f"{label} must be a non-empty string, got {value!r}")
    return value.strip().casefold()


def _validate_limit(limit: Any) -> None:
    """Raise :class:`RetrievalError` unless ``limit`` is None or a positive int."""
    if limit is None:
        return
    if isinstance(limit, bool) or not isinstance(limit, int):
        raise RetrievalError(f"limit must be a positive integer, got {limit!r}")
    if limit <= 0:
        raise RetrievalError(f"limit must be a positive integer, got {limit}")


def _fact_matches(fact: MedicalFact, normalized_query: str) -> bool:
    """Case-insensitive substring match against the searchable fields."""
    return any(
        normalized_query in _normalize(getattr(fact, field))
        for field in SEARCHABLE_FIELDS
    )


def search_facts(
    record: PatientRecord,
    query: str,
    fact_type: str | None = None,
    limit: int | None = None,
) -> list[MedicalFact]:
    """Deterministically search a patient record's existing facts.

    Matching is a case-insensitive substring test (``strip()`` +
    ``casefold()``) against :data:`SEARCHABLE_FIELDS` — no synonyms, no
    fuzzy matching, no LLM. Returned facts are the record's own objects
    with evidence unchanged, in the record's chronological order.

    Args:
        record: A validated :class:`~ai.patient_record.PatientRecord`.
        query: Non-empty search text (checked after stripping whitespace).
        fact_type: Optional filter; must be one of
            :data:`ai.fact_extractor.SUPPORTED_FACT_TYPES`.
        limit: Optional positive integer cap on the number of results.

    Returns:
        Matching ``MedicalFact`` objects (originals, not copies), in
        ``PatientRecord`` order. Empty list when nothing matches — never a
        fabricated fact.

    Raises:
        RetrievalError: If ``record`` is not a ``PatientRecord``, ``query``
            is empty/whitespace/non-string, ``fact_type`` is unsupported,
            or ``limit`` is not a positive integer.
    """
    _require_record(record)
    normalized_query = _require_query(query, "query")
    _validate_limit(limit)
    if fact_type is not None and fact_type not in SUPPORTED_FACT_TYPES:
        raise RetrievalError(
            f"unsupported fact_type '{fact_type!r}'; "
            f"expected one of {SUPPORTED_FACT_TYPES}"
        )

    results: list[MedicalFact] = []
    for fact in record.facts:  # PatientRecord order (chronological)
        if fact_type is not None and fact.fact_type != fact_type:
            continue
        if _fact_matches(fact, normalized_query):
            results.append(fact)

    if limit is not None:
        results = results[:limit]
    return results


def find_condition_history(
    record: PatientRecord,
    condition_name: str,
    limit: int | None = None,
) -> list[MedicalFact]:
    """Retrieve the record's condition-related facts for a condition name.

    Deterministic helper for questions like *"Has this patient had
    hypertension before?"* — it only returns the matching existing facts
    (fact types ``diagnosis`` and ``condition_status``); it never answers in
    natural language, never converts statuses to a boolean, and never
    infers anything from an empty result.

    Matching is case-insensitive substring against the fact ``name`` only.

    Args:
        record: A validated :class:`~ai.patient_record.PatientRecord`.
        condition_name: Non-empty condition text (checked after stripping).
        limit: Optional positive integer cap on the number of results.

    Returns:
        The matching ``MedicalFact`` objects in ``PatientRecord``
        chronological order, with their original ``status`` (historical,
        active, resolved, ruled_out, ...) and evidence intact. Empty list
        if no condition facts match.

    Raises:
        RetrievalError: If ``record`` is not a ``PatientRecord``,
            ``condition_name`` is empty/whitespace/non-string, or
            ``limit`` is not a positive integer.
    """
    _require_record(record)
    target = _require_query(condition_name, "condition_name")
    _validate_limit(limit)

    results = [
        fact
        for fact in record.facts  # PatientRecord order (chronological)
        if fact.fact_type in CONDITION_FACT_TYPES
        and target in _normalize(fact.name)
    ]

    if limit is not None:
        results = results[:limit]
    return results
