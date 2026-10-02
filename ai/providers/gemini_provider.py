"""Gemini adapter for MedTrace medical fact extraction (Stage 4).

Pipeline position::

    evidence chunks -> GeminiFactExtractionProvider -> candidate dicts
                    -> ai.fact_extractor.extract_facts() -> trusted MedicalFact

This file is an *adapter*, intentionally separate from the core fact schema
in :mod:`ai.fact_extractor`.

Security / evidence boundary:

- The API key comes from the environment (``GEMINI_API_KEY``). It is never
  hardcoded, never printed, and never stored in source.
- **Gemini output is NEVER trusted.** Everything Gemini returns is only a
  *candidate*; the Stage 3 validation layer verifies every quote, chunk id,
  and page number against the original chunks before anything becomes a
  trusted :class:`~ai.fact_extractor.MedicalFact`.
- Structured JSON output (explicit schema) is used instead of free-form text.
- For the hackathon demo use synthetic medical records only — never real
  patient data.

No LLM call happens unless a caller explicitly runs the provider; importing
this module requires only the standard library (``google-genai`` is imported
lazily when a request is actually made).
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

from ..fact_extractor import (
    SUPPORTED_DATE_PRECISIONS,
    SUPPORTED_FACT_TYPES,
    SUPPORTED_STATUSES,
)

__all__ = [
    "DEFAULT_MODEL",
    "ENV_API_KEY",
    "ENV_MODEL",
    "FACT_RESPONSE_SCHEMA",
    "GeminiFactExtractionProvider",
    "GeminiProviderError",
    "build_extraction_prompt",
    "parse_model_response",
]

DEFAULT_MODEL = "gemini-2.5-flash"
ENV_API_KEY = "GEMINI_API_KEY"
ENV_MODEL = "GEMINI_MODEL"

# Chunk ids look like "<document stem>-p<page>-c<n>"; used only as a
# fallback when the caller did not pass an explicit document name.
_CHUNK_ID_RE = re.compile(r"-p\d+-c\d+$")


class GeminiProviderError(Exception):
    """Raised for provider-level failures (key, SDK, API, malformed output).

    Evidence problems are *not* raised here — those are rejected by
    :mod:`ai.fact_extractor` after candidates are returned.
    """


# --------------------------------------------------------------------------
# Structured output schema
# --------------------------------------------------------------------------

FACT_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "OBJECT",
    "properties": {
        "facts": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "fact_type": {
                        "type": "STRING",
                        "enum": list(SUPPORTED_FACT_TYPES),
                    },
                    "name": {"type": "STRING"},
                    "value": {"type": "STRING", "nullable": True},
                    "unit": {"type": "STRING", "nullable": True},
                    "frequency": {"type": "STRING", "nullable": True},
                    "date": {"type": "STRING", "nullable": True},
                    "date_precision": {
                        "type": "STRING",
                        "enum": list(SUPPORTED_DATE_PRECISIONS),
                    },
                    "status": {
                        "type": "STRING",
                        "enum": list(SUPPORTED_STATUSES),
                    },
                    "evidence": {
                        "type": "OBJECT",
                        "properties": {
                            "document": {"type": "STRING"},
                            "page_number": {"type": "INTEGER"},
                            "chunk_id": {"type": "STRING"},
                            "quote": {"type": "STRING"},
                        },
                        "required": ["document", "page_number", "chunk_id", "quote"],
                    },
                },
                # Same required fields as ai.fact_extractor.FACT_REQUIRED_FIELDS.
                "required": ["fact_type", "name", "evidence"],
            },
        }
    },
    "required": ["facts"],
}


# --------------------------------------------------------------------------
# Prompt
# --------------------------------------------------------------------------


def build_extraction_prompt(chunk: dict[str, Any], document: str) -> str:
    """Build the extraction prompt for a single evidence chunk.

    The prompt tells Gemini to *extract*, never to diagnose, and to copy
    evidence verbatim with exact identifiers. This is instruction only —
    actual enforcement happens in :mod:`ai.fact_extractor`.

    Args:
        chunk: One chunk record from :func:`ai.chunker.chunk_extraction`.
        document: Source document filename the chunk belongs to.

    Returns:
        The prompt string.
    """
    fact_types = ", ".join(SUPPORTED_FACT_TYPES)
    statuses = ", ".join(SUPPORTED_STATUSES)
    precisions = ", ".join(SUPPORTED_DATE_PRECISIONS)

    return f"""You are extracting structured medical facts from a medical-document chunk.

Your job is extraction, NOT diagnosis.

Document:
{document}

Page:
{chunk.get("page_number", "")}

Chunk ID:
{chunk.get("chunk_id", "")}

Text:
{chunk.get("text", "")}

RULES
1. Only extract information explicitly present in the supplied text above.
2. Never invent a diagnosis.
3. Never infer a medical condition merely because a medication, symptom, test,
   or procedure might commonly be associated with it.
4. Never infer a date that is not explicitly supported by the text.
5. Never infer status unless supported by the text; use "unknown" otherwise.
6. Every fact MUST have evidence.
7. evidence.quote MUST be an exact substring of the supplied chunk text:
   do not paraphrase, do not summarize, do not modify punctuation, do not
   normalize capitalization, and copy the quote exactly.
8. Do not combine text from different chunks into one quote.
9. evidence.chunk_id MUST be exactly "{chunk.get("chunk_id", "")}".
10. evidence.page_number MUST be exactly {chunk.get("page_number", "")}.
11. evidence.document MUST be exactly "{document}".
12. Do not manufacture evidence identifiers.

Supported fact types: {fact_types}
Supported statuses: {statuses}
Supported date_precision values: {precisions}
  (day = YYYY-MM-DD, month = YYYY-MM, year = YYYY, unknown = date is null)

If there are no supported medical facts, return:
{{"facts": []}}

Focus on medically meaningful facts that fit the supported fact types.
Do not extract irrelevant administrative information unless it represents a
supported encounter or medical fact.

Remember: even if a fact seems obvious, if the text does not state it, do
not extract it. Gemini output is verified afterwards against the source
chunk; any invented evidence will be rejected."""


# --------------------------------------------------------------------------
# Response parsing (pure, no I/O)
# --------------------------------------------------------------------------


def parse_model_response(raw: str | dict[str, Any]) -> list[dict[str, Any]]:
    """Convert a Gemini structured-output response into candidate facts.

    Args:
        raw: JSON text (or an already-decoded dict) shaped like
            ``{"facts": [...]}``.

    Returns:
        The list of candidate fact dictionaries (unvalidated by design —
        validation belongs to :func:`ai.fact_extractor.extract_facts`).

    Raises:
        GeminiProviderError: If the response is not JSON, is not an object,
            lacks a ``facts`` list, or contains a non-object fact.
    """
    if isinstance(raw, dict):
        data: Any = raw
    elif isinstance(raw, str):
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise GeminiProviderError(f"Gemini returned invalid JSON: {exc}") from exc
    else:
        raise GeminiProviderError(
            f"expected a JSON string or dict from Gemini, got {type(raw).__name__}"
        )

    if not isinstance(data, dict):
        raise GeminiProviderError(
            f"expected a JSON object with a 'facts' list, got {type(data).__name__}"
        )

    facts = data.get("facts")
    if not isinstance(facts, list):
        raise GeminiProviderError("Gemini response is missing a 'facts' list")

    for position, fact in enumerate(facts):
        if not isinstance(fact, dict):
            raise GeminiProviderError(
                f"fact at position {position} is not a JSON object, "
                f"got {type(fact).__name__}"
            )
    return facts


# --------------------------------------------------------------------------
# Provider
# --------------------------------------------------------------------------


class GeminiFactExtractionProvider:
    """FactExtractionProvider adapter for Google Gemini (``google-genai``).

    One Gemini request is made per chunk (no batching in Stage 4). The
    provider returns *candidate* dictionaries exactly as Gemini produced
    them; trust is established later by
    :func:`ai.fact_extractor.extract_facts`.

    Args:
        api_key: Explicit API key. If ``None``, ``GEMINI_API_KEY`` is read
            from the environment. Never hardcode a key.
        model: Model name. If ``None``, ``GEMINI_MODEL`` is read from the
            environment, falling back to ``gemini-2.5-flash``.
        document: Source document filename, included in the prompt so Gemini
            can fill ``evidence.document`` exactly. If ``None``, the document
            stem is derived from each chunk id (pass this explicitly for
            exact filenames).
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        document: str | None = None,
    ) -> None:
        # SECURITY: the key lives only on this instance; it is never logged,
        # never formatted into prompts, and never written to disk.
        key = api_key if api_key is not None else os.environ.get(ENV_API_KEY)
        if not key:
            raise GeminiProviderError(f"{ENV_API_KEY} is not set")

        self._api_key = key
        self._model = model or os.environ.get(ENV_MODEL) or DEFAULT_MODEL
        self._document = document
        self._client: Any = None  # created lazily; no SDK needed until a request

    @property
    def model(self) -> str:
        """The model name used for requests."""
        return self._model

    # -- SDK seams (patched in unit tests; the only SDK touchpoints) --------

    def _get_client(self) -> Any:
        """Create and cache the ``google.genai`` client.

        Raises:
            GeminiProviderError: If ``google-genai`` is not installed or the
                client cannot be constructed.
        """
        if self._client is None:
            try:
                from google import genai
            except ImportError as exc:
                raise GeminiProviderError(
                    "google-genai is not installed. Install it with: "
                    "pip install -r ai/requirements.txt"
                ) from exc
            try:
                self._client = genai.Client(api_key=self._api_key)
            except Exception as exc:
                raise GeminiProviderError(
                    f"failed to create the Gemini client: {exc}"
                ) from exc
        return self._client

    def _build_config(self) -> Any:
        """Build the SDK config requesting structured JSON output.

        Raises:
            GeminiProviderError: If ``google-genai`` is not installed.
        """
        try:
            from google.genai import types
        except ImportError as exc:
            raise GeminiProviderError(
                "google-genai is not installed. Install it with: "
                "pip install -r ai/requirements.txt"
            ) from exc
        # Explicit schema -> Gemini must return matching JSON, not prose.
        return types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=FACT_RESPONSE_SCHEMA,
        )

    def _call_model(self, chunk: dict[str, Any]) -> str:
        """Send one chunk to Gemini and return the raw JSON text.

        This is the single network boundary of the adapter; unit tests
        patch this method (or the two SDK seams above) so no test ever
        performs a real API request.

        Raises:
            GeminiProviderError: On any SDK/API failure or empty response.
        """
        prompt = build_extraction_prompt(
            chunk, self._document or self._document_from_chunk(chunk)
        )
        client = self._get_client()
        config = self._build_config()
        try:
            response = client.models.generate_content(
                model=self._model,
                contents=prompt,
                config=config,
            )
        except GeminiProviderError:
            raise
        except Exception as exc:
            chunk_id = chunk.get("chunk_id", "?")
            raise GeminiProviderError(
                f"Gemini API request failed for chunk '{chunk_id}': {exc}"
            ) from exc

        text = getattr(response, "text", None)
        if not text or not str(text).strip():
            chunk_id = chunk.get("chunk_id", "?")
            raise GeminiProviderError(
                f"Gemini returned an empty response for chunk '{chunk_id}'"
            )
        return str(text)

    # -- Helpers ------------------------------------------------------------

    @staticmethod
    def _document_from_chunk(chunk: dict[str, Any]) -> str:
        """Derive a document name from a chunk id (``report-p1-c1`` -> ``report``)."""
        chunk_id = str(chunk.get("chunk_id", ""))
        return _CHUNK_ID_RE.sub("", chunk_id) or "unknown_document"

    # -- FactExtractionProvider protocol ------------------------------------

    def extract_facts(self, chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Return *candidate* facts for the given chunks (one request each).

        The returned dictionaries are whatever Gemini produced. They are
        deliberately NOT validated here — the caller must run them through
        :func:`ai.fact_extractor.extract_facts`, which rejects invented
        quotes, wrong page numbers, and nonexistent chunk ids.

        Args:
            chunks: Evidence chunk records.

        Returns:
            Candidate fact dictionaries.

        Raises:
            GeminiProviderError: On missing key, SDK/API failure, or
                malformed model output.
        """
        candidates: list[dict[str, Any]] = []
        for chunk in chunks:
            raw = self._call_model(chunk)
            candidates.extend(parse_model_response(raw))
        return candidates
