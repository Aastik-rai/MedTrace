"""Tests for the Stage 6 deterministic retrieval layer.

Run from the repository root::

    python -m unittest discover -s tests -v

No LLM/Gemini/network calls. All fixtures are synthetic MedicalFact
objects validated through Stage 3; no real patient information is used.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

# Make the repository root importable regardless of the runner used.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ai.fact_extractor import Evidence, MedicalFact, validate_fact  # noqa: E402
from ai.patient_record import PatientRecord  # noqa: E402
from ai.retrieval import (  # noqa: E402
    CONDITION_FACT_TYPES,
    RetrievalError,
    find_condition_history,
    search_facts,
)

HOSPITAL = "hospital_2024.pdf"
LAB = "lab_2025.pdf"
PRESCRIPTION = "prescription_2026.pdf"


def make_fact(
    fact_type: str = "diagnosis",
    name: str = "Hypertension",
    date: str | None = None,
    date_precision: str = "unknown",
    status: str = "unknown",
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


def make_record(facts: list[MedicalFact]) -> PatientRecord:
    """Wrap synthetic facts in an explicitly identified patient record."""
    return PatientRecord(patient_id="demo-patient-001", facts=facts)


def build_demo_facts() -> list[MedicalFact]:
    """Synthetic facts used by most tests (input order is shuffled)."""
    return [
        # Condition-related facts (name: Hypertension), chronological:
        make_fact(
            fact_type="condition_status",
            status="ruled_out",
            date="2024-01-05",
            date_precision="day",
            page_number=1,
            quote="Synthetic: hypertension was ruled out on review.",
        ),
        make_fact(
            status="historical",
            date="2024-03-10",
            date_precision="day",
            page_number=2,
            chunk_id="hospital_2024-p2-c1",
            quote="Synthetic: patient has a history of hypertension.",
        ),
        make_fact(
            status="active",
            date="2024-11-20",
            date_precision="day",
            page_number=4,
            quote="Synthetic: hypertension remains active.",
        ),
        make_fact(
            fact_type="condition_status",
            status="resolved",
            document=LAB,
            date="2025-06-01",
            date_precision="day",
            quote="Synthetic: hypertension recorded as resolved.",
        ),
        # Non-condition facts:
        make_fact(
            fact_type="vital",
            name="Blood pressure",
            value="138/88",
            unit="mmHg",
            date="2024-04-04",
            date_precision="day",
            quote="Synthetic: blood pressure was 138/88 mmHg.",
        ),
        make_fact(
            fact_type="lab_result",
            name="Hemoglobin",
            value="13.8",
            unit="g/dL",
            document=LAB,
            date="2025-07-10",
            date_precision="day",
            quote="Synthetic: hemoglobin was 13.8 g/dL.",
        ),
        make_fact(
            fact_type="procedure",
            name="Stress test",
            document=LAB,
            date="2025-09-09",
            date_precision="day",
            quote="Synthetic: a stress test was performed.",
        ),
        make_fact(
            fact_type="medication",
            name="Amlodipine",
            value="5 mg",
            frequency="once daily",
            status="active",
            document=PRESCRIPTION,
            date="2026-01-05",
            date_precision="day",
            quote="Synthetic: amlodipine 5 mg once daily was prescribed.",
        ),
    ]


class TestSearchFactsMatching(unittest.TestCase):
    """Tests 1-4: exact, case-insensitive, substring, field coverage."""

    def setUp(self) -> None:
        self.record = make_record(build_demo_facts())

    def test_search_finds_an_exact_fact_name(self) -> None:
        results = search_facts(self.record, "Hemoglobin")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].name, "Hemoglobin")

    def test_search_is_case_insensitive(self) -> None:
        for query in ("hypertension", "HYPERTENSION", "hypeRtension"):
            with self.subTest(query=query):
                results = search_facts(self.record, query)
                self.assertEqual(len(results), 4)
                self.assertTrue(all(r.name == "Hypertension" for r in results))

    def test_search_supports_deterministic_substring_matching(self) -> None:
        press = search_facts(self.record, "press")
        self.assertEqual(len(press), 1)
        self.assertEqual(press[0].name, "Blood pressure")

        globin = search_facts(self.record, "globin")
        self.assertEqual(len(globin), 1)
        self.assertEqual(globin[0].name, "Hemoglobin")

        # Deterministic: same query, same result every time.
        self.assertEqual(
            [f.name for f in press], [f.name for f in search_facts(self.record, "press")]
        )

    def test_search_matches_value_unit_status_and_fact_type_fields(self) -> None:
        # value
        by_value = search_facts(self.record, "138/88")
        self.assertEqual([f.name for f in by_value], ["Blood pressure"])

        # unit
        by_unit = search_facts(self.record, "MMHG")
        self.assertEqual([f.name for f in by_unit], ["Blood pressure"])

        # status (only the one historical fact; others are explicit)
        by_status = search_facts(self.record, "historical")
        self.assertEqual(len(by_status), 1)
        self.assertEqual(by_status[0].status, "historical")

        # fact_type
        by_type = search_facts(self.record, "lab_result")
        self.assertEqual([f.name for f in by_type], ["Hemoglobin"])

        # frequency
        by_frequency = search_facts(self.record, "once daily")
        self.assertEqual([f.name for f in by_frequency], ["Amlodipine"])


class TestSearchFactsFilters(unittest.TestCase):
    """Tests 5-11: fact_type filter, validation, limit handling."""

    def setUp(self) -> None:
        self.record = make_record(build_demo_facts())

    def test_fact_type_filter_works(self) -> None:
        all_hypertension = search_facts(self.record, "hypertension")
        self.assertEqual(len(all_hypertension), 4)

        only_diagnosis = search_facts(
            self.record, "hypertension", fact_type="diagnosis"
        )
        self.assertEqual(len(only_diagnosis), 2)
        self.assertTrue(all(f.fact_type == "diagnosis" for f in only_diagnosis))

        # Empty intersection still returns [], never other types.
        self.assertEqual(
            search_facts(self.record, "hypertension", fact_type="lab_result"), []
        )

    def test_invalid_fact_type_raises(self) -> None:
        for bad_type in ("disease", "Diagnosis", "", "diagnoses"):
            with self.subTest(fact_type=bad_type):
                with self.assertRaises(RetrievalError):
                    search_facts(self.record, "hypertension", fact_type=bad_type)
        with self.assertRaises(RetrievalError):
            search_facts(self.record, "hypertension", fact_type=123)  # type: ignore[arg-type]

    def test_empty_query_is_rejected(self) -> None:
        with self.assertRaises(RetrievalError):
            search_facts(self.record, "")

    def test_whitespace_only_query_is_rejected(self) -> None:
        for query in ("   ", "\t\n", " \r "):
            with self.subTest(query=repr(query)):
                with self.assertRaises(RetrievalError):
                    search_facts(self.record, query)
        # Non-strings are rejected too.
        with self.assertRaises(RetrievalError):
            search_facts(self.record, None)  # type: ignore[arg-type]

    def test_limit_none_returns_all_matching_facts(self) -> None:
        results = search_facts(self.record, "hypertension", limit=None)
        self.assertEqual(len(results), 4)
        self.assertEqual(results, search_facts(self.record, "hypertension"))

    def test_positive_limit_truncates_results(self) -> None:
        everything = search_facts(self.record, "hypertension")
        truncated = search_facts(self.record, "hypertension", limit=2)

        self.assertEqual(len(truncated), 2)
        # Truncation keeps the earliest results (record order), no re-ranking.
        self.assertEqual(truncated, everything[:2])
        # limit larger than the match count returns all matches.
        self.assertEqual(
            len(search_facts(self.record, "hypertension", limit=100)), 4
        )

    def test_invalid_limit_is_rejected(self) -> None:
        for bad_limit in (0, -1, -100, "5", 1.5, [2], True):
            with self.subTest(limit=repr(bad_limit)):
                with self.assertRaises(RetrievalError):
                    search_facts(self.record, "hypertension", limit=bad_limit)  # type: ignore[arg-type]

    def test_malformed_record_is_rejected(self) -> None:
        with self.assertRaises(RetrievalError):
            search_facts({"patient_id": "x", "facts": []}, "query")  # type: ignore[arg-type]
        with self.assertRaises(RetrievalError):
            find_condition_history(None, "hypertension")  # type: ignore[arg-type]


class TestNoMatches(unittest.TestCase):
    """Test 12 & 20: empty results, never fabricated facts or inferences."""

    def test_no_matches_returns_empty_list(self) -> None:
        record = make_record(build_demo_facts())
        self.assertEqual(search_facts(record, "zzz-no-such-text"), [])

    def test_absent_condition_produces_no_facts_and_no_inference(self) -> None:
        record = make_record(build_demo_facts())
        before = list(record.facts)

        self.assertEqual(find_condition_history(record, "diabetes"), [])
        self.assertEqual(search_facts(record, "diabetes"), [])

        # Nothing was added, removed, or altered in the record.
        self.assertEqual(list(record.facts), before)
        self.assertEqual(len(record.facts), len(before))


class TestEvidencePreservation(unittest.TestCase):
    """Tests 13-15: originals returned, evidence byte-for-byte unchanged."""

    def setUp(self) -> None:
        self.record = make_record(build_demo_facts())
        self.results = search_facts(self.record, "hypertension")

    def test_returned_facts_are_the_original_medical_fact_data(self) -> None:
        stored_ids = {id(fact) for fact in self.record.facts}
        for fact in self.results:
            self.assertIn(id(fact), stored_ids)  # identity, not a copy
            self.assertIsInstance(fact, MedicalFact)

        # Same objects the record stores (chronological order included).
        self.assertIs(self.results[1], self.record.facts[1])

    def test_evidence_is_preserved_exactly(self) -> None:
        for fact in self.results:
            self.assertIsInstance(fact.evidence, Evidence)
        # A full fact round-trips identically through retrieval.
        stored = next(
            f for f in self.record.facts if f.evidence.chunk_id == "hospital_2024-p2-c1"
        )
        retrieved = next(
            f for f in self.results if f.evidence.chunk_id == "hospital_2024-p2-c1"
        )
        self.assertIs(retrieved, stored)
        self.assertEqual(retrieved.to_dict(), stored.to_dict())

    def test_document_page_chunk_quote_remain_unchanged(self) -> None:
        target = next(
            f for f in self.results if f.evidence.chunk_id == "hospital_2024-p2-c1"
        )
        evidence = target.evidence
        self.assertEqual(evidence.document, HOSPITAL)
        self.assertEqual(evidence.page_number, 2)
        self.assertEqual(evidence.chunk_id, "hospital_2024-p2-c1")
        self.assertEqual(
            evidence.quote, "Synthetic: patient has a history of hypertension."
        )


class TestFindConditionHistory(unittest.TestCase):
    """Tests 16-19: condition facts only, statuses preserved, no inference."""

    def setUp(self) -> None:
        self.record = make_record(build_demo_facts())
        self.history = find_condition_history(self.record, "hypertension")

    def test_finds_diagnosis_facts(self) -> None:
        diagnoses = [f for f in self.history if f.fact_type == "diagnosis"]
        self.assertEqual(len(diagnoses), 2)
        self.assertEqual(CONDITION_FACT_TYPES, ("diagnosis", "condition_status"))

    def test_finds_condition_status_facts(self) -> None:
        statuses = [f for f in self.history if f.fact_type == "condition_status"]
        self.assertEqual(len(statuses), 2)

    def test_does_not_return_unrelated_fact_types(self) -> None:
        self.assertEqual(len(self.history), 4)  # only the 4 condition facts
        unrelated_types = {"medication", "lab_result", "procedure", "vital"}
        self.assertFalse(
            any(f.fact_type in unrelated_types for f in self.history)
        )
        self.assertEqual(
            [f.name for f in find_condition_history(self.record, "amlodipine")], []
        )

    def test_preserves_statuses_without_boolean_conversion(self) -> None:
        # Exact statuses survive retrieval, in chronological record order.
        self.assertEqual(
            [(f.date, f.status) for f in self.history],
            [
                ("2024-01-05", "ruled_out"),
                ("2024-03-10", "historical"),
                ("2024-11-20", "active"),
                ("2025-06-01", "resolved"),
            ],
        )
        # Matching is case-insensitive and substring-based on the name.
        self.assertEqual(
            len(find_condition_history(self.record, "HYPERTENSION")), 4
        )
        self.assertEqual(len(find_condition_history(self.record, "hyper")), 4)

    def test_limit_applies_to_condition_history(self) -> None:
        self.assertEqual(len(find_condition_history(self.record, "hypertension", limit=2)), 2)
        with self.assertRaises(RetrievalError):
            find_condition_history(self.record, "hypertension", limit=0)
        with self.assertRaises(RetrievalError):
            find_condition_history(self.record, "  ")


class TestOrderingAndDuplicates(unittest.TestCase):
    """Tests 21-22: no dedup, PatientRecord order preserved."""

    def test_duplicate_facts_are_not_aggressively_deduplicated(self) -> None:
        identical = make_fact(
            status="historical",
            date="2024-03-10",
            date_precision="day",
            quote="Synthetic duplicate entry.",
        )
        record = make_record(
            [identical, make_fact(
                status="historical",
                date="2024-03-10",
                date_precision="day",
                quote="Synthetic duplicate entry.",
            )]
        )
        results = search_facts(record, "hypertension")

        self.assertEqual(len(results), 2)
        self.assertEqual(results[0], results[1])  # both byte-identical, both kept
        self.assertEqual(
            len(find_condition_history(record, "hypertension")), 2
        )

    def test_patient_record_ordering_is_preserved(self) -> None:
        record = make_record(build_demo_facts())

        # Record stores facts chronologically regardless of input order...
        self.assertEqual(
            [f.date for f in record.facts],
            [
                "2024-01-05",
                "2024-03-10",
                "2024-04-04",
                "2024-11-20",
                "2025-06-01",
                "2025-07-10",
                "2025-09-09",
                "2026-01-05",
            ],
        )
        # ...and retrieval returns results in exactly that order (no re-rank).
        results = search_facts(record, "hypertension")
        record_order = [id(f) for f in record.facts if id(f) in {id(r) for r in results}]
        self.assertEqual([id(r) for r in results], record_order)
        self.assertEqual(
            [f.date for f in results],
            ["2024-01-05", "2024-03-10", "2024-11-20", "2025-06-01"],
        )


if __name__ == "__main__":
    unittest.main()
