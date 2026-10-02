"""Stage 5: provider-independent longitudinal patient record organization.

Pipeline position::

    MedicalFact[] (validated by Stage 3)
        -> PatientRecord (this module)
        -> chronological medical history

Core principles:

- **Never accepts unvalidated AI output.** The only accepted fact type is
  :class:`ai.fact_extractor.MedicalFact`, already validated by Stage 3.
  Raw dictionaries are rejected, not converted — callers must run
  ``extract_facts()`` first.
- **Organization, not identity resolution.** The caller explicitly supplies
  ``patient_id``. This module never tries to decide whether two documents
  belong to the same person (no name/date/symptom matching, no AI
  inference, no patient merging).
- **No medical reasoning.** Facts are sorted, grouped, and serialized —
  never inferred from, deduplicated by medical judgment, or altered.
  Evidence (document, page, chunk, quote) stays attached to every fact.
- **No I/O, no LLM, no network.** Standard library only.

Chronological ordering puts known dates first (ascending, mixed day/month/
year precision compared by their ISO prefixes) and unknown-date facts last,
in stable input order. Unknown dates are never converted into invented
dates.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Sequence

from .fact_extractor import MedicalFact

__all__ = [
    "PatientRecord",
    "PatientRecordError",
    "get_facts_by_name",
    "get_facts_by_type",
    "get_facts_for_document",
    "sort_facts_chronologically",
]


class PatientRecordError(ValueError):
    """Raised when a PatientRecord is constructed or deserialized invalidly."""


def _chronology_key(fact: MedicalFact) -> tuple[int, str]:
    """Sort key: known dates ascending first, unknown dates after.

    ISO date strings of mixed precision (``YYYY``, ``YYYY-MM``,
    ``YYYY-MM-DD``) compare correctly as plain strings because shorter
    prefixes sort before their extensions (``2019`` < ``2019-08``).
    Facts with no date are grouped last with an empty secondary key, so
    Python's stable sort preserves their input order.
    """
    if fact.date:
        return (0, fact.date)
    return (1, "")


def sort_facts_chronologically(facts: Sequence[MedicalFact]) -> list[MedicalFact]:
    """Return facts in chronological order without mutating them.

    Ordering rules:

    1. Facts with known dates come first, ascending (day/month/year
       precision all supported; ISO prefix comparison).
    2. Facts with unknown dates come after all known dates.
    3. Ties (equal dates, or consecutive unknown-date facts) keep their
       original relative order (stable sort) — the ordering is deterministic.

    The original ``date`` / ``date_precision`` of every fact is untouched;
    no date is ever invented.

    Args:
        facts: Medical facts (typically from Stage 3).

    Returns:
        A new list in chronological order; the input is not modified.
    """
    return sorted(facts, key=_chronology_key)


def _unique_documents(facts: Sequence[MedicalFact]) -> tuple[str, ...]:
    """Collect unique ``evidence.document`` values in record order."""
    seen: dict[str, None] = {}
    for fact in facts:
        document = fact.evidence.document
        if document and document not in seen:
            seen[document] = None
    return tuple(seen)


def _require_str(value: Any, label: str) -> None:
    """Raise :class:`PatientRecordError` unless ``value`` is a non-empty string."""
    if not isinstance(value, str) or not value.strip():
        raise PatientRecordError(f"{label} must be a non-empty string, got {value!r}")


@dataclass(frozen=True)
class PatientRecord:
    """An in-memory longitudinal record for one explicitly identified patient.

    The record organizes already-validated facts; it never merges patients,
    never performs medical reasoning, and never alters a fact or its
    evidence.

    Args:
        patient_id: Caller-supplied record identity (e.g.
            ``"demo-patient-001"``). Stage 5 never infers this.
        facts: Validated :class:`~ai.fact_extractor.MedicalFact` objects.
            Raw dicts are rejected — run ``extract_facts()`` first. Stored
            as a tuple in chronological order (``sort_facts_chronologically``).
        source_documents: Optional explicit list of source document names.
            Every entry must be the ``evidence.document`` of at least one
            fact. When ``None`` (default), the unique documents are derived
            from the facts in first-appearance order.

    Raises:
        PatientRecordError: If ``patient_id`` is empty, ``facts`` contains
            anything other than ``MedicalFact`` objects, or
            ``source_documents`` names a document no fact points to.
    """

    patient_id: str
    facts: tuple[MedicalFact, ...] = ()
    source_documents: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        # --- patient identity: explicit only, never inferred -------------
        _require_str(self.patient_id, "patient_id")

        # --- trust boundary: MedicalFact only, never raw AI dicts ---------
        if not isinstance(self.facts, (list, tuple)):
            raise PatientRecordError(
                f"facts must be a list or tuple of MedicalFact objects, "
                f"got {type(self.facts).__name__}"
            )
        for position, fact in enumerate(self.facts):
            if not isinstance(fact, MedicalFact):
                raise PatientRecordError(
                    f"facts[{position}] must be a validated MedicalFact, "
                    f"got {type(fact).__name__}. Raw candidate dictionaries "
                    "must first pass through extract_facts() (Stage 3)."
                )

        # Chronological timeline: stable sort, facts themselves untouched.
        ordered = tuple(sort_facts_chronologically(self.facts))
        object.__setattr__(self, "facts", ordered)

        # --- source documents: derived or validated against the facts -----
        if self.source_documents is None:
            object.__setattr__(self, "source_documents", _unique_documents(ordered))
            return

        if not isinstance(self.source_documents, (list, tuple)):
            raise PatientRecordError(
                "source_documents must be a list or tuple of document names, "
                f"got {type(self.source_documents).__name__}"
            )
        fact_documents = set(_unique_documents(ordered))
        unique: dict[str, None] = {}
        for document in self.source_documents:
            _require_str(document, "source_documents entry")
            if document not in fact_documents:
                raise PatientRecordError(
                    f"source document '{document}' is not the evidence "
                    "document of any fact in this record"
                )
            unique.setdefault(document, None)
        object.__setattr__(self, "source_documents", tuple(unique))

    # -- serialization -----------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dictionary with full evidence attached.

        Uses :meth:`MedicalFact.to_dict` — no serialization logic is
        duplicated. Every fact keeps its ``evidence`` block (document, page
        number, chunk id, quote), so the dictionary can be fully
        reconstructed with :meth:`from_dict`.
        """
        return {
            "patient_id": self.patient_id,
            "source_documents": list(self.source_documents or ()),
            "facts": [fact.to_dict() for fact in self.facts],
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialize the record to deterministic JSON text."""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

    @classmethod
    def from_dict(cls, data: Any) -> "PatientRecord":
        """Rebuild a record from :meth:`to_dict` output.

        Fact dictionaries are converted back through the existing
        :meth:`MedicalFact.from_dict` (Stage 3) — again, no duplicated
        serialization logic.

        Args:
            data: Dictionary produced by :meth:`to_dict`.

        Returns:
            An equivalent :class:`PatientRecord`.

        Raises:
            PatientRecordError: If ``data`` is not a dict, misses required
                fields, or contains malformed facts.
        """
        if not isinstance(data, dict):
            raise PatientRecordError(
                f"record must be a dict, got {type(data).__name__}"
            )
        for key in ("patient_id", "facts"):
            if key not in data:
                raise PatientRecordError(f"record is missing required field '{key}'")

        raw_facts = data["facts"]
        if not isinstance(raw_facts, (list, tuple)):
            raise PatientRecordError(
                f"record['facts'] must be a list, got {type(raw_facts).__name__}"
            )
        facts: list[MedicalFact] = []
        for position, raw in enumerate(raw_facts):
            if isinstance(raw, MedicalFact):
                facts.append(raw)
            elif isinstance(raw, dict):
                try:
                    facts.append(MedicalFact.from_dict(raw))
                except ValueError as exc:
                    raise PatientRecordError(
                        f"facts[{position}] is malformed: {exc}"
                    ) from exc
            else:
                raise PatientRecordError(
                    f"facts[{position}] must be a dict or MedicalFact, "
                    f"got {type(raw).__name__}"
                )

        source_documents = data.get("source_documents")
        if source_documents is not None and not isinstance(
            source_documents, (list, tuple)
        ):
            raise PatientRecordError(
                "record['source_documents'] must be a list, "
                f"got {type(source_documents).__name__}"
            )

        return cls(
            patient_id=data["patient_id"],
            facts=facts,
            source_documents=source_documents,
        )

    @classmethod
    def from_json(cls, text: str) -> "PatientRecord":
        """Rebuild a record from :meth:`to_json` text."""
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise PatientRecordError(f"invalid record JSON: {exc}") from exc
        return cls.from_dict(data)


# --------------------------------------------------------------------------
# Deterministic query helpers (no semantic search, no embeddings, no LLM)
# --------------------------------------------------------------------------


def get_facts_by_type(record: PatientRecord, fact_type: str) -> list[MedicalFact]:
    """Return the record's facts whose ``fact_type`` exactly equals the input.

    Matching is exact and case-sensitive against
    :data:`ai.fact_extractor.SUPPORTED_FACT_TYPES` values; unknown types
    simply match nothing. Results keep the record's chronological order.
    """
    _require_str(fact_type, "fact_type")
    return [fact for fact in record.facts if fact.fact_type == fact_type]


def get_facts_by_name(record: PatientRecord, name: str) -> list[MedicalFact]:
    """Return facts whose ``name`` matches case-insensitively (exact only).

    Matching strategy (documented, deterministic): Unicode ``casefold()``
    comparison of the whole name — ``"hypertension"`` finds
    ``"Hypertension"``. No fuzzy matching, no substring matching, no
    trimming. Results keep the record's chronological order.
    """
    _require_str(name, "name")
    target = name.casefold()
    return [fact for fact in record.facts if fact.name.casefold() == target]


def get_facts_for_document(record: PatientRecord, document: str) -> list[MedicalFact]:
    """Return facts whose ``evidence.document`` exactly equals the input.

    Exact string match; results keep the record's chronological order.
    """
    _require_str(document, "document")
    return [
        fact for fact in record.facts if fact.evidence.document == document
    ]
