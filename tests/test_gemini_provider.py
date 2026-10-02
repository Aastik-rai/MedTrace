"""Tests for the Stage 4 Gemini provider adapter.

Run from the repository root::

    python -m unittest discover -s tests -v

These tests NEVER call the real Gemini API and require no API key: the two
SDK seams (``_get_client`` / ``_build_config``) and the network boundary
(``_call_model``) are mocked with unittest.mock.

All fixtures are synthetic structural samples — no real patient data.
"""

from __future__ import annotations

import json
import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

# Make the repository root importable regardless of the runner used.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ai.chunker import chunk_extraction  # noqa: E402
from ai.fact_extractor import FactValidationError, MedicalFact, extract_facts  # noqa: E402
from ai.providers import (  # noqa: E402
    DEFAULT_MODEL,
    GeminiFactExtractionProvider,
    GeminiProviderError,
)
from ai.providers.gemini_provider import (  # noqa: E402
    build_extraction_prompt,
    parse_model_response,
)

# Synthetic structural fixtures only (no real patient records).
PAGE_TEXT = (
    "Patient has a history of hypertension. "
    "Blood pressure was 138/88 mmHg. "
    "Patient was prescribed amoxicillin 500 mg."
)
QUOTE = "Patient has a history of hypertension."
CHUNK_ID = "sample_report-p1-c1"
DOCUMENT = "sample_report.pdf"


def make_chunks() -> list[dict]:
    """Build one evidence chunk containing PAGE_TEXT (via the real chunker)."""
    extraction = {
        "document": DOCUMENT,
        "source_path": f"C:/data/{DOCUMENT}",
        "total_pages": 1,
        "pages": [{"page_number": 1, "text": PAGE_TEXT}],
    }
    return chunk_extraction(extraction, chunk_size=1000, overlap=200)["chunks"]


def make_response(**fact_overrides) -> str:
    """Build a Gemini-style structured JSON response as a string."""
    evidence = {
        "document": DOCUMENT,
        "page_number": 1,
        "chunk_id": CHUNK_ID,
        "quote": QUOTE,
    }
    evidence.update(fact_overrides.pop("evidence", {}))
    fact = {
        "fact_type": "diagnosis",
        "name": "Hypertension",
        "value": None,
        "unit": None,
        "frequency": None,
        "date": None,
        "date_precision": "unknown",
        "status": "historical",
        "evidence": evidence,
    }
    fact.update(fact_overrides)
    return json.dumps({"facts": [fact]})


def make_provider(**kwargs) -> GeminiFactExtractionProvider:
    """Provider with an explicit key and a deterministic model name."""
    kwargs.setdefault("api_key", "test-key-not-real")
    kwargs.setdefault("model", "test-model")
    return GeminiFactExtractionProvider(**kwargs)


class TestProviderConfiguration(unittest.TestCase):
    """Tests 1-3: initialization, API key handling, model configuration."""

    def test_init_with_explicit_api_key(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True):
            provider = GeminiFactExtractionProvider(api_key="test-key-not-real")

        # Lazy client: nothing SDK-related happens at construction time.
        self.assertIsNone(provider._client)
        self.assertNotIn("test-key-not-real", repr(provider))

    def test_missing_api_key_produces_clear_error(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(GeminiProviderError) as ctx:
                GeminiFactExtractionProvider()
        self.assertEqual(str(ctx.exception), "GEMINI_API_KEY is not set")

    def test_model_configuration(self) -> None:
        # Default model.
        with mock.patch.dict(os.environ, {"GEMINI_API_KEY": "k"}, clear=True):
            self.assertEqual(GeminiFactExtractionProvider().model, DEFAULT_MODEL)
            self.assertEqual(DEFAULT_MODEL, "gemini-2.5-flash")

        # Model from environment.
        env = {"GEMINI_API_KEY": "k", "GEMINI_MODEL": "gemini-env-model"}
        with mock.patch.dict(os.environ, env, clear=True):
            self.assertEqual(GeminiFactExtractionProvider().model, "gemini-env-model")

        # Explicit argument beats the environment.
        with mock.patch.dict(os.environ, env, clear=True):
            provider = GeminiFactExtractionProvider(model="gemini-arg-model")
            self.assertEqual(provider.model, "gemini-arg-model")


class TestResponseParsing(unittest.TestCase):
    """Tests 4-6: converting Gemini-style responses into candidates."""

    def test_valid_response_converts_to_candidate_facts(self) -> None:
        candidates = parse_model_response(make_response())

        self.assertEqual(len(candidates), 1)
        self.assertIsInstance(candidates[0], dict)
        self.assertEqual(candidates[0]["fact_type"], "diagnosis")
        self.assertEqual(candidates[0]["evidence"]["chunk_id"], CHUNK_ID)

    def test_empty_extraction_returns_zero_facts(self) -> None:
        provider = make_provider()
        with mock.patch.object(provider, "_call_model", return_value='{"facts": []}'):
            self.assertEqual(provider.extract_facts(make_chunks()), [])
            # Also through the Stage 3 pipeline: zero candidates -> zero facts.
            self.assertEqual(extract_facts(make_chunks(), provider), [])

    def test_malformed_response_raises_provider_error(self) -> None:
        malformed = [
            "this is not json",                    # invalid JSON
            '{"nope": []}',                        # missing facts key
            '{"facts": "not-a-list"}',             # facts not a list
            '{"facts": [42]}',                     # fact not an object
            '["not", "an", "object"]',             # top level not an object
        ]
        for raw in malformed:
            with self.subTest(raw=raw):
                with self.assertRaises(GeminiProviderError):
                    parse_model_response(raw)


class TestCandidateFidelity(unittest.TestCase):
    """Tests 8 and 11: candidates keep Gemini's fields byte-for-byte."""

    def test_candidate_evidence_fields_preserved_exactly(self) -> None:
        distinctive_quote = "BP was 138/88 mmHg.  Patient advised: 'REST'."  # exact casing/punctuation
        response = make_response(
            evidence={
                "quote": distinctive_quote,
                "page_number": 1,
                "chunk_id": CHUNK_ID,
                "document": DOCUMENT,
            }
        )
        provider = make_provider()
        with mock.patch.object(provider, "_call_model", return_value=response):
            candidates = provider.extract_facts(make_chunks())

        # The provider must not rewrite, normalize, or sanitize anything.
        expected = json.loads(response)["facts"][0]
        self.assertEqual(candidates[0], expected)
        self.assertEqual(candidates[0]["evidence"]["quote"], distinctive_quote)

    def test_page_chunk_document_metadata_is_preserved(self) -> None:
        provider = make_provider(document=DOCUMENT)
        with mock.patch.object(provider, "_call_model", return_value=make_response()):
            facts = extract_facts(make_chunks(), provider)

        self.assertEqual(len(facts), 1)
        evidence = facts[0].evidence
        self.assertIsInstance(facts[0], MedicalFact)
        self.assertEqual(evidence.document, DOCUMENT)
        self.assertEqual(evidence.page_number, 1)
        self.assertEqual(evidence.chunk_id, CHUNK_ID)
        self.assertEqual(evidence.quote, QUOTE)


class TestValidationIntegration(unittest.TestCase):
    """Tests 9-10: the EXISTING Stage 3 layer judges every candidate."""

    def test_hallucinated_quote_is_rejected(self) -> None:
        # Quote not present in the chunk text -> Stage 3 must reject it.
        response = make_response(
            evidence={"quote": "Patient has a history of diabetes."}
        )
        provider = make_provider()
        with mock.patch.object(provider, "_call_model", return_value=response):
            with self.assertRaises(FactValidationError):
                extract_facts(make_chunks(), provider)

    def test_wrong_page_number_is_rejected(self) -> None:
        # Verbatim quote but wrong page -> rejected by existing validation.
        response = make_response(evidence={"page_number": 9})
        provider = make_provider()
        with mock.patch.object(provider, "_call_model", return_value=response):
            with self.assertRaises(FactValidationError):
                extract_facts(make_chunks(), provider)

    def test_nonexistent_chunk_id_is_rejected(self) -> None:
        response = make_response(evidence={"chunk_id": "other_report-p9-c9"})
        provider = make_provider()
        with mock.patch.object(provider, "_call_model", return_value=response):
            with self.assertRaises(FactValidationError):
                extract_facts(make_chunks(), provider)

    def test_valid_exact_quote_is_accepted(self) -> None:
        provider = make_provider()
        with mock.patch.object(provider, "_call_model", return_value=make_response()):
            facts = extract_facts(make_chunks(), provider)

        self.assertEqual(len(facts), 1)
        self.assertEqual(facts[0].evidence.quote, QUOTE)
        self.assertIn(QUOTE, PAGE_TEXT)  # the fixture really is verbatim


class TestApiFailures(unittest.TestCase):
    """Tests 7: SDK/API failures surface as GeminiProviderError."""

    def test_api_failure_raises_provider_error(self) -> None:
        provider = make_provider()
        mock_client = mock.MagicMock()
        mock_client.models.generate_content.side_effect = RuntimeError("quota exceeded")

        with mock.patch.object(provider, "_get_client", return_value=mock_client):
            with mock.patch.object(provider, "_build_config", return_value="cfg"):
                with self.assertRaises(GeminiProviderError) as ctx:
                    provider.extract_facts(make_chunks())

        self.assertIn("quota exceeded", str(ctx.exception))

    def test_empty_model_response_raises_provider_error(self) -> None:
        provider = make_provider()
        mock_client = mock.MagicMock()
        mock_client.models.generate_content.return_value = SimpleNamespace(text="  ")

        with mock.patch.object(provider, "_get_client", return_value=mock_client):
            with mock.patch.object(provider, "_build_config", return_value="cfg"):
                with self.assertRaises(GeminiProviderError) as ctx:
                    provider.extract_facts(make_chunks())

        self.assertIn("empty response", str(ctx.exception))


class TestNoNetworkDuringTests(unittest.TestCase):
    """Test 12: unit tests must never create a client or hit the API."""

    def test_no_api_request_is_made(self) -> None:
        provider = make_provider()
        with mock.patch.object(
            provider, "_call_model", return_value=make_response()
        ) as call_model:
            with mock.patch.object(provider, "_get_client") as get_client:
                facts = extract_facts(make_chunks(), provider)

        get_client.assert_not_called()
        call_model.assert_called_once()
        self.assertIsNone(provider._client)
        self.assertEqual(len(facts), 1)


class TestExtractionPrompt(unittest.TestCase):
    """The prompt carries the extraction-only and verbatim-evidence rules."""

    def test_prompt_contains_evidence_rules(self) -> None:
        chunk = make_chunks()[0]
        prompt = build_extraction_prompt(chunk, DOCUMENT)

        # Chunk context is included.
        self.assertIn(DOCUMENT, prompt)
        self.assertIn(CHUNK_ID, prompt)
        self.assertIn(PAGE_TEXT, prompt)

        # Extraction-not-diagnosis and evidence rules are stated.
        self.assertIn("NOT diagnosis", prompt)
        self.assertIn("Never invent a diagnosis", prompt)
        self.assertIn("exact substring of the supplied chunk text", prompt)
        self.assertIn("do not paraphrase", prompt)
        self.assertIn('{"facts": []}', prompt)

        # Supported value sets are listed.
        self.assertIn("lab_result", prompt)
        self.assertIn("ruled_out", prompt)
        self.assertIn("date_precision", prompt)


if __name__ == "__main__":
    unittest.main()
