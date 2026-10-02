"""Tests for the MedTrace medical fact schema and evidence validation.

Run from the repository root::

    python -m unittest discover -s tests -v
    # or
    pytest tests/test_fact_extractor.py

All fixtures are neutral structural samples. No real patient data and no
medical claims are used.
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
from ai.fact_extractor import (  # noqa: E402
    SUPPORTED_DATE_PRECISIONS,
    SUPPORTED_FACT_TYPES,
    SUPPORTED_STATUSES,
    Evidence,
    FactValidationError,
    MedicalFact,
    build_chunk_index,
    extract_facts,
    facts_to_json,
    validate_fact,
    validate_facts,
)

PAGE_TEXT = (
    "Sample condition was documented on 2024-01-01. "
    "Sample medication was prescribed. "
    "Sample test result was recorded as 42 units."
)
VALID_QUOTE = "Sample condition was documented on 2024-01-01."


def make_chunks() -> dict:
    """Build a chunk_extraction() result containing one chunk."""
    extraction = {
        "document": "sample_report.pdf",
        "source_path": "C:/data/sample_report.pdf",
        "total_pages": 1,
        "pages": [{"page_number": 1, "text": PAGE_TEXT}],
    }
    return chunk_extraction(extraction, chunk_size=1000, overlap=200)


def make_evidence(**overrides) -> Evidence:
    """Build a valid evidence record pointing at the sample chunk."""
    fields = {
        "document": "sample_report.pdf",
        "page_number": 1,
        "chunk_id": "sample_report-p1-c1",
        "quote": VALID_QUOTE,
    }
    fields.update(overrides)
    return Evidence(**fields)


def make_fact(**overrides) -> MedicalFact:
    """Build a valid fact with defaults that can be overridden per test."""
    fields = {
        "fact_type": "diagnosis",
        "name": "Sample condition",
        "evidence": make_evidence(),
    }
    fields.update(overrides)
    return MedicalFact(**fields)


class TestFactAndEvidenceCreation(unittest.TestCase):
    """Tests 1-2: creating facts and evidence records."""

    def test_fact_creation(self) -> None:
        fact = make_fact()

        self.assertEqual(fact.fact_type, "diagnosis")
        self.assertEqual(fact.name, "Sample condition")
        # Optional fields default to None / unknown instead of being invented.
        self.assertIsNone(fact.value)
        self.assertIsNone(fact.unit)
        self.assertIsNone(fact.frequency)
        self.assertIsNone(fact.date)
        self.assertEqual(fact.date_precision, "unknown")
        self.assertEqual(fact.status, "unknown")

        # Validation accepts a fully specified fact too.
        rich = make_fact(
            value="42",
            unit="units",
            frequency="once daily",
            date="2024-01-01",
            date_precision="day",
            status="active",
        )
        self.assertEqual(validate_fact(rich), rich)

    def test_evidence_creation(self) -> None:
        evidence = make_evidence()

        self.assertEqual(evidence.document, "sample_report.pdf")
        self.assertEqual(evidence.page_number, 1)
        self.assertEqual(evidence.chunk_id, "sample_report-p1-c1")
        self.assertEqual(evidence.quote, VALID_QUOTE)
        self.assertEqual(
            evidence.to_dict(),
            {
                "document": "sample_report.pdf",
                "page_number": 1,
                "chunk_id": "sample_report-p1-c1",
                "quote": VALID_QUOTE,
            },
        )


class TestSupportedTypesAndStatuses(unittest.TestCase):
    """Tests 3, 4, 14: fact types and status values."""

    def test_supported_fact_types(self) -> None:
        self.assertEqual(
            SUPPORTED_FACT_TYPES,
            (
                "diagnosis",
                "medication",
                "lab_result",
                "procedure",
                "symptom",
                "allergy",
                "vital",
                "encounter",
                "condition_status",
            ),
        )
        for fact_type in SUPPORTED_FACT_TYPES:
            fact = make_fact(fact_type=fact_type)
            self.assertEqual(validate_fact(fact), fact)

    def test_invalid_fact_type(self) -> None:
        for bad_type in ("symptom_status", "", "Diagnosis", "disease"):
            with self.assertRaises(FactValidationError):
                validate_fact(make_fact(fact_type=bad_type))
        with self.assertRaises(FactValidationError):
            validate_fact(make_fact(fact_type=123))  # type: ignore[arg-type]

    def test_status_validation(self) -> None:
        self.assertEqual(
            SUPPORTED_STATUSES,
            ("active", "historical", "suspected", "resolved", "ruled_out", "unknown"),
        )
        for status in SUPPORTED_STATUSES:
            fact = make_fact(status=status)
            self.assertEqual(validate_fact(fact), fact)

        for bad_status in ("maybe", "ACTIVE", "confirmed", ""):
            with self.assertRaises(FactValidationError):
                validate_fact(make_fact(status=bad_status))


class TestMissingFields(unittest.TestCase):
    """Tests 5-11: required fields must be present and non-empty."""

    def test_missing_fact_name(self) -> None:
        with self.assertRaises(FactValidationError):
            MedicalFact.from_dict({"fact_type": "diagnosis", "evidence": make_evidence().to_dict()})
        for bad_name in ("", "   "):
            with self.assertRaises(FactValidationError):
                validate_fact(make_fact(name=bad_name))

    def test_missing_evidence(self) -> None:
        with self.assertRaises(FactValidationError):
            MedicalFact.from_dict({"fact_type": "diagnosis", "name": "Sample condition"})

    def test_missing_document(self) -> None:
        evidence = make_evidence().to_dict()
        del evidence["document"]
        with self.assertRaises(FactValidationError):
            validate_fact(
                {"fact_type": "diagnosis", "name": "Sample condition", "evidence": evidence}
            )

    def test_missing_page_number(self) -> None:
        evidence = make_evidence().to_dict()
        del evidence["page_number"]
        with self.assertRaises(FactValidationError):
            validate_fact(
                {"fact_type": "diagnosis", "name": "Sample condition", "evidence": evidence}
            )
        # Wrong types are rejected too.
        with self.assertRaises(FactValidationError):
            validate_fact(make_fact(evidence=make_evidence(page_number="7")))

    def test_missing_chunk_id(self) -> None:
        evidence = make_evidence().to_dict()
        del evidence["chunk_id"]
        with self.assertRaises(FactValidationError):
            validate_fact(
                {"fact_type": "diagnosis", "name": "Sample condition", "evidence": evidence}
            )

    def test_missing_quote(self) -> None:
        evidence = make_evidence().to_dict()
        del evidence["quote"]
        with self.assertRaises(FactValidationError):
            validate_fact(
                {"fact_type": "diagnosis", "name": "Sample condition", "evidence": evidence}
            )

    def test_empty_quote(self) -> None:
        for bad_quote in ("", "   "):
            with self.assertRaises(FactValidationError):
                validate_fact(make_fact(evidence=make_evidence(quote=bad_quote)))


class TestEvidenceVerification(unittest.TestCase):
    """Tests 12-13: quotes must be verbatim substrings of source chunks."""

    def test_evidence_quote_verification(self) -> None:
        index = build_chunk_index(make_chunks())

        fact = make_fact()
        validated = validate_fact(fact, source_chunks=index)
        self.assertEqual(validated, fact)

        # A longer verbatim quote from deeper in the chunk also verifies.
        longer = make_fact(
            evidence=make_evidence(quote="Sample medication was prescribed.")
        )
        validate_fact(longer, source_chunks=index)

    def test_invalid_evidence_quote(self) -> None:
        index = build_chunk_index(make_chunks())

        # Paraphrased quote: not present in the source chunk.
        paraphrased = make_fact(
            evidence=make_evidence(quote="The sample condition was recorded on 2024-01-01.")
        )
        with self.assertRaises(FactValidationError):
            validate_fact(paraphrased, source_chunks=index)

        # Quote invented from thin air.
        invented = make_fact(evidence=make_evidence(quote="Never stated in the document."))
        with self.assertRaises(FactValidationError):
            validate_fact(invented, source_chunks=index)

        # Unknown chunk id.
        unknown_chunk = make_fact(evidence=make_evidence(chunk_id="sample_report-p9-c9"))
        with self.assertRaises(FactValidationError):
            validate_fact(unknown_chunk, source_chunks=index)

        # Page number that contradicts the cited chunk.
        wrong_page = make_fact(evidence=make_evidence(page_number=2))
        with self.assertRaises(FactValidationError):
            validate_fact(wrong_page, source_chunks=index)


class TestDatesAndValues(unittest.TestCase):
    """Tests 15-17: date precision, lab values, medication fields."""

    def test_date_and_date_precision_validation(self) -> None:
        self.assertEqual(
            SUPPORTED_DATE_PRECISIONS, ("day", "month", "year", "unknown")
        )

        valid_pairs = [
            ("2024-01-01", "day"),
            ("2024-01", "month"),
            ("2024", "year"),
            (None, "unknown"),
        ]
        for date_value, precision in valid_pairs:
            fact = make_fact(date=date_value, date_precision=precision)
            self.assertEqual(validate_fact(fact), fact)

        invalid_pairs = [
            (None, "day"),                # precision without a date
            ("2024-01-01", "unknown"),    # date without precision
            ("2024-1-1", "day"),          # not zero-padded
            ("2024-13-01", "day"),        # impossible month
            ("2024-02-30", "day"),        # impossible calendar day
            ("2024-01-01", "month"),      # precision/date mismatch
            ("2024-01-01", "hour"),       # unsupported precision
            ("yesterday", "day"),         # not an ISO date
        ]
        for date_value, precision in invalid_pairs:
            with self.assertRaises(FactValidationError):
                validate_fact(make_fact(date=date_value, date_precision=precision))

    def test_lab_style_values(self) -> None:
        fact = make_fact(
            fact_type="lab_result",
            name="Sample test result",
            value="42",
            unit="units",
            date="2024-01-01",
            date_precision="day",
        )

        validated = validate_fact(fact)
        as_dict = validated.to_dict()
        self.assertEqual(as_dict["fact_type"], "lab_result")
        self.assertEqual(as_dict["value"], "42")
        self.assertEqual(as_dict["unit"], "units")

    def test_medication_style_fields(self) -> None:
        fact = make_fact(
            fact_type="medication",
            name="Sample medication",
            value="500 mg",
            frequency="twice daily",
        )

        validated = validate_fact(fact)
        as_dict = validated.to_dict()
        self.assertEqual(as_dict["value"], "500 mg")
        self.assertEqual(as_dict["frequency"], "twice daily")
        # Irrelevant fields stay null instead of being forced onto the fact.
        self.assertIsNone(as_dict["unit"])

        with self.assertRaises(FactValidationError):
            validate_fact(make_fact(fact_type="medication", name="Sample", value=""))


class TestMultipleFactsAndSerialization(unittest.TestCase):
    """Tests 18-19 and the provider pipeline."""

    def test_multiple_facts(self) -> None:
        facts = [
            make_fact(fact_type="diagnosis", name="Sample condition"),
            make_fact(fact_type="medication", name="Sample medication"),
            make_fact(fact_type="lab_result", name="Sample test result"),
        ]

        validated = validate_facts(facts)
        self.assertEqual(len(validated), 3)
        self.assertEqual(validated, facts)  # order preserved

        self.assertEqual(validate_facts([]), [])

    def test_deterministic_serialization(self) -> None:
        facts = [
            make_fact(fact_type="diagnosis", name="Sample condition", date="2024-01-01", date_precision="day"),
            make_fact(fact_type="lab_result", name="Sample test result", value="42", unit="units"),
        ]

        first_json = facts_to_json(facts)
        second_json = facts_to_json(list(facts))
        self.assertEqual(first_json, second_json)

        # Round trip: dict -> fact -> dict is lossless.
        rebuilt = MedicalFact.from_dict(facts[0].to_dict())
        self.assertEqual(rebuilt, facts[0])
        self.assertEqual(rebuilt.to_dict(), facts[0].to_dict())

    def test_extract_facts_runs_provider_and_verifies_evidence(self) -> None:
        """A provider plug-in returns candidates; all are verified verbatim."""

        class FakeProvider:
            def extract_facts(self, chunks):
                self.seen_chunks = chunks
                return [
                    {
                        "fact_type": "diagnosis",
                        "name": "Sample condition",
                        "evidence": {
                            "document": "sample_report.pdf",
                            "page_number": 1,
                            "chunk_id": chunks[0]["chunk_id"],
                            "quote": "Sample condition was documented on 2024-01-01.",
                        },
                    }
                ]

        provider = FakeProvider()
        facts = extract_facts(make_chunks(), provider)

        self.assertEqual(len(facts), 1)
        self.assertIsInstance(facts[0], MedicalFact)
        self.assertEqual(facts[0].fact_type, "diagnosis")
        # The provider received the actual chunk records as input.
        self.assertEqual(provider.seen_chunks[0]["chunk_id"], "sample_report-p1-c1")

    def test_extract_facts_rejects_unverifiable_quotes(self) -> None:
        """A provider that paraphrases evidence is rejected, not accepted."""

        class BadProvider:
            def extract_facts(self, chunks):
                return [
                    {
                        "fact_type": "diagnosis",
                        "name": "Sample condition",
                        "evidence": {
                            "document": "sample_report.pdf",
                            "page_number": 1,
                            "chunk_id": chunks[0]["chunk_id"],
                            "quote": "A paraphrased claim with no source.",
                        },
                    }
                ]

        with self.assertRaises(FactValidationError):
            extract_facts(make_chunks(), BadProvider())


if __name__ == "__main__":
    unittest.main()
