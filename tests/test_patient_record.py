"""Tests for the Stage 5 longitudinal patient record builder.

Run from the repository root::

    python -m unittest discover -s tests -v

No Gemini/API calls: all fixtures are synthetic MedicalFact objects built
locally and validated through Stage 3's validate_fact(). No real patient
information is used.
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

# Make the repository root importable regardless of the runner used.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ai.fact_extractor import (  # noqa: E402
    Evidence,
    MedicalFact,
    validate_fact,
)
from ai.patient_record import (  # noqa: E402
    PatientRecord,
    PatientRecordError,
    get_facts_by_name,
    get_facts_by_type,
    get_facts_for_document,
    sort_facts_chronologically,
)

HOSPITAL = "hospital_2024.pdf"
LAB = "lab_2025.pdf"
PRESCRIPTION = "prescription_2026.pdf"


def make_fact(
    fact_type: str = "diagnosis",
    name: str = "Hypertension",
    date: str | None = None,
    date_precision: str = "unknown",
    status: str = "historical",
    document: str = HOSPITAL,
    page_number: int = 1,
    chunk_id: str | None = None,
    quote: str | None = None,
    value: str | None = None,
    unit: str | None = None,
    frequency: str | None = None,
) -> MedicalFact:
    """Build a synthetic fact validated through Stage 3 (no chunk index)."""
    if chunk_id is None:
        stem = document.removesuffix(".pdf")
        chunk_id = f"{stem}-p{page_number}-c1"
    if quote is None:
        quote = f"Synthetic {name} entry for demo records."
    fact = MedicalFact(
        fact_type=fact_type,
        name=name,
        evidence=Evidence(
            document=document,
            page_number=page_number,
            chunk_id=chunk_id,
            quote=quote,
        ),
        value=value,
        unit=unit,
        frequency=frequency,
        date=date,
        date_precision=date_precision,
        status=status,
    )
    return validate_fact(fact)


class TestPatientRecordConstruction(unittest.TestCase):
    """Tests 1-3, 22-23: identity, accepted inputs, empty records."""

    def test_patient_id_must_be_non_empty(self) -> None:
        for bad_id in ("", "   ", 123, None):
            with self.subTest(patient_id=bad_id):
                with self.assertRaises(PatientRecordError):
                    PatientRecord(patient_id=bad_id, facts=[])  # type: ignore[arg-type]

    def test_accepts_validated_medical_facts(self) -> None:
        facts = [make_fact(), make_fact(name="Amlodipine", fact_type="medication")]
        record = PatientRecord(patient_id="demo-patient-001", facts=facts)

        self.assertEqual(record.patient_id, "demo-patient-001")
        self.assertEqual(len(record.facts), 2)
        # The exact validated objects are retained (evidence untouched).
        for fact in record.facts:
            self.assertIsInstance(fact, MedicalFact)
        self.assertTrue(set(record.facts) <= set(facts))

    def test_raw_dictionaries_are_rejected(self) -> None:
        raw = {
            "fact_type": "diagnosis",
            "name": "Hypertension",
            "evidence": {
                "document": HOSPITAL,
                "page_number": 1,
                "chunk_id": "hospital_2024-p1-c1",
                "quote": "Synthetic entry.",
            },
        }
        with self.assertRaises(PatientRecordError) as ctx:
            PatientRecord(patient_id="demo-patient-001", facts=[raw])  # type: ignore[list-item]
        # The trust-boundary message points callers at Stage 3.
        self.assertIn("extract_facts", str(ctx.exception))

        # Wrong non-dict types are rejected as well.
        with self.assertRaises(PatientRecordError):
            PatientRecord(patient_id="demo-patient-001", facts=["not a fact"])  # type: ignore[list-item]

    def test_empty_fact_list_is_handled(self) -> None:
        record = PatientRecord(patient_id="demo-patient-001", facts=[])

        self.assertEqual(record.facts, ())
        self.assertEqual(record.source_documents, ())
        # Round trip works for empty records too.
        self.assertEqual(PatientRecord.from_dict(record.to_dict()), record)

    def test_multiple_documents_in_one_explicitly_owned_record(self) -> None:
        facts = [
            make_fact(document=HOSPITAL, date="2024-03-02", date_precision="day"),
            make_fact(
                fact_type="lab_result",
                name="Hemoglobin",
                document=LAB,
                date="2025-07-10",
                date_precision="day",
            ),
            make_fact(
                fact_type="medication",
                name="Amlodipine",
                document=PRESCRIPTION,
                date="2026-01-05",
                date_precision="day",
            ),
        ]
        record = PatientRecord(patient_id="demo-patient-001", facts=facts)

        self.assertEqual(
            list(record.source_documents),
            [HOSPITAL, LAB, PRESCRIPTION],
        )
        # Explicit source_documents are accepted when they correspond.
        explicit = PatientRecord(
            patient_id="demo-patient-001",
            facts=facts,
            source_documents=[LAB, HOSPITAL],
        )
        self.assertEqual(explicit.source_documents, (LAB, HOSPITAL))
        # Documents no fact points to are rejected.
        with self.assertRaises(PatientRecordError):
            PatientRecord(
                patient_id="demo-patient-001",
                facts=facts,
                source_documents=["ghost_record.pdf"],
            )


class TestChronologicalOrdering(unittest.TestCase):
    """Tests 4-10: deterministic longitudinal ordering, no invented dates."""

    def test_facts_are_sorted_chronologically(self) -> None:
        facts = [
            make_fact(date="2025-06-01", date_precision="day"),
            make_fact(date="2024-01-15", date_precision="day"),
            make_fact(date="2026-02-10", date_precision="day"),
        ]
        record = PatientRecord(patient_id="demo-patient-001", facts=facts)

        self.assertEqual(
            [fact.date for fact in record.facts],
            ["2024-01-15", "2025-06-01", "2026-02-10"],
        )

    def test_day_level_dates_sort_correctly(self) -> None:
        facts = [
            make_fact(date="2024-12-31", date_precision="day"),
            make_fact(date="2024-01-01", date_precision="day"),
            make_fact(date="2024-07-15", date_precision="day"),
        ]
        sorted_dates = [f.date for f in sort_facts_chronologically(facts)]
        self.assertEqual(
            sorted_dates, ["2024-01-01", "2024-07-15", "2024-12-31"]
        )

    def test_month_level_dates_sort_correctly(self) -> None:
        facts = [
            make_fact(date="2024-11", date_precision="month"),
            make_fact(date="2023-12", date_precision="month"),
            make_fact(date="2024-03", date_precision="month"),
        ]
        sorted_dates = [f.date for f in sort_facts_chronologically(facts)]
        self.assertEqual(sorted_dates, ["2023-12", "2024-03", "2024-11"])

    def test_year_level_dates_sort_correctly(self) -> None:
        facts = [
            make_fact(date="2024", date_precision="year"),
            make_fact(date="2022", date_precision="year"),
            make_fact(date="2023", date_precision="year"),
        ]
        sorted_dates = [f.date for f in sort_facts_chronologically(facts)]
        self.assertEqual(sorted_dates, ["2022", "2023", "2024"])

        # Mixed precision compares by ISO prefix (year before its months).
        mixed = [
            make_fact(date="2021-05-05", date_precision="day"),
            make_fact(date="2020", date_precision="year"),
            make_fact(date="2020-12", date_precision="month"),
        ]
        self.assertEqual(
            [f.date for f in sort_facts_chronologically(mixed)],
            ["2020", "2020-12", "2021-05-05"],
        )

    def test_unknown_date_facts_remain_in_the_record(self) -> None:
        facts = [
            make_fact(date=None),  # unknown
            make_fact(date="2024-05-05", date_precision="day"),
            make_fact(date=None, name="Amlodipine", fact_type="medication"),
        ]
        record = PatientRecord(patient_id="demo-patient-001", facts=facts)

        # Nothing is dropped: all three facts remain...
        self.assertEqual(len(record.facts), 3)
        # ...and unknown-date facts come after known dates, input order kept.
        self.assertEqual(record.facts[0].date, "2024-05-05")
        self.assertIsNone(record.facts[1].date)
        self.assertIsNone(record.facts[2].date)
        self.assertEqual(record.facts[1].name, "Hypertension")
        self.assertEqual(record.facts[2].name, "Amlodipine")

    def test_unknown_dates_are_never_invented(self) -> None:
        fact = make_fact(date=None, date_precision="unknown")
        record = PatientRecord(patient_id="demo-patient-001", facts=[fact])

        self.assertIsNone(record.facts[0].date)
        self.assertEqual(record.facts[0].date_precision, "unknown")
        serialized = record.to_json()
        self.assertNotIn("1900-01-01", serialized)  # no fake sentinel dates
        self.assertIn('"date": null', serialized)
        self.assertNotIn("date\": \"20", serialized.split('"facts"')[1].replace(f'"{fact.date}"', ""))

    def test_original_medical_fact_dates_are_unchanged(self) -> None:
        fact = make_fact(date="2024-05-06", date_precision="day")
        original_date, original_precision = fact.date, fact.date_precision

        sorted_list = sort_facts_chronologically([fact])
        record = PatientRecord(patient_id="demo-patient-001", facts=[fact])

        self.assertEqual(fact.date, original_date)
        self.assertEqual(fact.date_precision, original_precision)
        self.assertIs(sorted_list[0], fact)
        self.assertIs(record.facts[0], fact)  # same object, not a copy


class TestEvidencePreservation(unittest.TestCase):
    """Tests 11-14: evidence stays attached to every fact, verbatim."""

    def setUp(self) -> None:
        self.quote = (
            "Synthetic note: blood pressure was recorded during the demo visit."
        )
        self.fact = make_fact(
            document=HOSPITAL,
            page_number=3,
            chunk_id="hospital_2024-p3-c2",
            quote=self.quote,
            date="2024-03-02",
            date_precision="day",
        )
        self.record = PatientRecord(
            patient_id="demo-patient-001", facts=[self.fact]
        )
        self.stored = self.record.facts[0]

    def test_evidence_document_is_preserved(self) -> None:
        self.assertEqual(self.stored.evidence.document, HOSPITAL)
        self.assertIs(self.stored.evidence, self.fact.evidence)

    def test_page_numbers_are_preserved(self) -> None:
        self.assertEqual(self.stored.evidence.page_number, 3)

    def test_chunk_ids_are_preserved(self) -> None:
        self.assertEqual(self.stored.evidence.chunk_id, "hospital_2024-p3-c2")

    def test_evidence_quotes_are_preserved_exactly(self) -> None:
        self.assertEqual(self.stored.evidence.quote, self.quote)
        # Serialization keeps the quote verbatim too.
        self.assertEqual(
            self.record.to_dict()["facts"][0]["evidence"]["quote"], self.quote
        )


class TestSourceDocuments(unittest.TestCase):
    """Test 15: unique, deterministic source document list."""

    def test_source_documents_are_deduplicated_deterministically(self) -> None:
        facts = [
            make_fact(document=HOSPITAL, date="2024-01-10", date_precision="day"),
            make_fact(
                fact_type="lab_result",
                name="Hemoglobin",
                document=LAB,
                date="2025-02-10",
                date_precision="day",
            ),
            make_fact(  # same document again, must not duplicate
                fact_type="vital",
                name="Blood pressure",
                document=HOSPITAL,
                date="2024-06-10",
                date_precision="day",
            ),
        ]
        record = PatientRecord(patient_id="demo-patient-001", facts=facts)

        self.assertEqual(
            list(record.source_documents), [HOSPITAL, LAB]
        )
        # Deterministic: same input -> same output.
        again = PatientRecord(patient_id="demo-patient-001", facts=facts)
        self.assertEqual(record.source_documents, again.source_documents)
        # Derived documents always come from evidence, never invented.
        self.assertEqual(
            set(record.source_documents),
            {f.evidence.document for f in record.facts},
        )


class TestQueryHelpers(unittest.TestCase):
    """Tests 16-18: deterministic, exact helpers (no semantic search)."""

    def setUp(self) -> None:
        self.record = PatientRecord(
            patient_id="demo-patient-001",
            facts=[
                make_fact(
                    fact_type="diagnosis",
                    name="Hypertension",
                    date="2024-01-15",
                    date_precision="day",
                ),
                make_fact(
                    fact_type="medication",
                    name="Amlodipine",
                    document=PRESCRIPTION,
                    date="2026-01-05",
                    date_precision="day",
                ),
                make_fact(
                    fact_type="lab_result",
                    name="Hemoglobin",
                    document=LAB,
                    date="2025-07-10",
                    date_precision="day",
                ),
                make_fact(
                    fact_type="diagnosis",
                    name="Hypertension",
                    document=LAB,
                    date="2025-03-03",
                    date_precision="day",
                ),
            ],
        )

    def test_get_facts_by_type_is_exact(self) -> None:
        diagnoses = get_facts_by_type(self.record, "diagnosis")
        self.assertEqual(len(diagnoses), 2)
        self.assertTrue(all(f.fact_type == "diagnosis" for f in diagnoses))
        # Exact match only: unknown types and case variants match nothing.
        self.assertEqual(get_facts_by_type(self.record, "diagnosis "), [])
        self.assertEqual(get_facts_by_type(self.record, "Diagnosis"), [])
        self.assertEqual(get_facts_by_type(self.record, "disease"), [])

    def test_get_facts_by_name_is_case_insensitive(self) -> None:
        self.assertEqual(
            [f.name for f in get_facts_by_name(self.record, "hypertension")],
            ["Hypertension", "Hypertension"],
        )
        self.assertEqual(
            [f.fact_type for f in get_facts_by_name(self.record, "AMLODIPINE")],
            ["medication"],
        )
        # Exact (case-folded) matching only: no fuzzy/substring behavior.
        self.assertEqual(get_facts_by_name(self.record, "hyperten"), [])
        self.assertEqual(get_facts_by_name(self.record, "hypertension "), [])

    def test_get_facts_for_document(self) -> None:
        lab_facts = get_facts_for_document(self.record, LAB)
        self.assertEqual(len(lab_facts), 2)
        self.assertTrue(
            all(f.evidence.document == LAB for f in lab_facts)
        )
        self.assertEqual(get_facts_for_document(self.record, "missing.pdf"), [])
        # Results follow the record's chronological order.
        self.assertEqual(
            [f.date for f in lab_facts], ["2025-03-03", "2025-07-10"]
        )


class TestNoAggressiveDeduplication(unittest.TestCase):
    """Test 19: similar facts are never silently removed."""

    def test_similar_and_duplicate_facts_are_all_retained(self) -> None:
        facts = [
            make_fact(
                fact_type="vital",
                name="Blood pressure",
                quote="Synthetic: blood pressure was 138/88 mmHg.",
                date="2024-04-04",
                date_precision="day",
                status="unknown",
            ),
            make_fact(
                fact_type="vital",
                name="Blood pressure",
                quote="Synthetic: blood pressure was 145/92 mmHg.",
                date="2025-05-05",
                date_precision="day",
                status="unknown",
            ),
            make_fact(status="historical", date="2024-01-01", date_precision="day"),
            make_fact(status="active", date="2025-01-01", date_precision="day"),
        ]
        # Two byte-identical facts: still not removed.
        identical = make_fact(status="historical", date="2023-09-09", date_precision="day")
        facts.append(identical)
        facts.append(make_fact(status="historical", date="2023-09-09", date_precision="day"))

        record = PatientRecord(patient_id="demo-patient-001", facts=facts)
        self.assertEqual(len(record.facts), 6)
        self.assertEqual(len([f for f in record.facts if f.name == "Blood pressure"]), 2)
        self.assertEqual(len([f for f in record.facts if f.status == "active"]), 1)
        # Historical vs active hypertension both remain (no medical merging).
        self.assertEqual(len([f for f in record.facts if f.status == "historical"]), 3)
        # Both byte-identical duplicates survive.
        self.assertEqual(len([f for f in record.facts if f == identical]), 2)


class TestSerialization(unittest.TestCase):
    """Tests 20-21: JSON round trip with full evidence."""

    def setUp(self) -> None:
        self.record = PatientRecord(
            patient_id="demo-patient-001",
            facts=[
                make_fact(
                    date="2024-01-15",
                    date_precision="day",
                    page_number=3,
                    chunk_id="hospital_2024-p3-c2",
                    quote="Synthetic diagnosis quote for the demo record.",
                ),
                make_fact(
                    fact_type="lab_result",
                    name="Hemoglobin",
                    value="13.8",
                    unit="g/dL",
                    document=LAB,
                    date="2025-07-10",
                    date_precision="day",
                ),
                make_fact(date=None),
            ],
        )

    def test_serialization_preserves_all_fact_and_evidence_fields(self) -> None:
        data = self.record.to_dict()

        self.assertEqual(data["patient_id"], "demo-patient-001")
        self.assertEqual(data["source_documents"], [HOSPITAL, LAB])
        self.assertEqual(len(data["facts"]), 3)

        expected = [fact.to_dict() for fact in self.record.facts]
        self.assertEqual(data["facts"], expected)

        evidence = data["facts"][0]["evidence"]
        self.assertEqual(
            set(evidence.keys()),
            {"document", "page_number", "chunk_id", "quote"},
        )
        self.assertEqual(evidence["page_number"], 3)
        self.assertEqual(evidence["chunk_id"], "hospital_2024-p3-c2")
        # The JSON is actually serializable.
        reloaded = json.loads(self.record.to_json())
        self.assertEqual(reloaded, data)

    def test_deserialization_recreates_equivalent_medical_facts(self) -> None:
        data = self.record.to_dict()
        rebuilt = PatientRecord.from_dict(data)

        self.assertEqual(rebuilt, self.record)
        for original, restored in zip(self.record.facts, rebuilt.facts):
            self.assertIsInstance(restored, MedicalFact)
            self.assertEqual(restored, original)
            self.assertEqual(restored.evidence, original.evidence)

        # JSON text round trip as well.
        self.assertEqual(PatientRecord.from_json(self.record.to_json()), self.record)

    def test_from_dict_rejects_malformed_input(self) -> None:
        with self.assertRaises(PatientRecordError):
            PatientRecord.from_dict("not a dict")
        with self.assertRaises(PatientRecordError):
            PatientRecord.from_dict({"facts": []})  # missing patient_id
        with self.assertRaises(PatientRecordError):
            PatientRecord.from_dict({"patient_id": "demo-patient-001"})  # missing facts


if __name__ == "__main__":
    unittest.main()
