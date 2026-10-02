# Gemini Fact Extraction Provider (Stage 4)

## 1. What the provider does

`ai/providers/gemini_provider.py` implements `GeminiFactExtractionProvider`,
a `FactExtractionProvider` adapter that sends **one evidence chunk per
Gemini request** using Google's current `google-genai` SDK (not the
deprecated `google-generativeai`) and returns **candidate** fact
dictionaries in structured JSON output — never free-form text.

Default model: `gemini-2.5-flash`.

## 2. Where it fits in the pipeline

```
PDF → ai/pdf_processor.py → page text
    → ai/chunker.py        → evidence chunks
    → GeminiFactExtractionProvider (Stage 4) → candidate dicts
    → ai/fact_extractor.py (Stage 3)         → trusted MedicalFact objects
```

The adapter sits *outside* the core schema: the core package (`ai`) stays
usable with zero LLM dependencies; the provider is imported separately via
`from ai.providers import GeminiFactExtractionProvider`.

## 3. Installing dependencies

```powershell
.venv\Scripts\python -m pip install -r ai/requirements.txt
```

`ai/requirements.txt` contains `PyMuPDF` and `google-genai`. The SDK is
imported lazily — unit tests and the rest of the pipeline run without it.

## 4. Setting the API key

Copy `.env.example` to `.env` and fill in your key, or set it in your shell:

```powershell
$env:GEMINI_API_KEY = "your_key_here"
```

- The key is read only from the environment — never hardcoded, never
  printed, never written to disk by this code.
- `.env` is git-ignored (`.env.example` is tracked and contains no
  credentials).
- If the key is missing, the provider raises
  `GeminiProviderError("GEMINI_API_KEY is not set")`.

## 5. Selecting the model

Priority: explicit `GeminiFactExtractionProvider(model=...)` → `GEMINI_MODEL`
environment variable → default `gemini-2.5-flash`.

## 6. Why Gemini output is not trusted

The model is prompted to extract (never diagnose), to copy quotes verbatim,
and to reuse the exact `document` / `page_number` / `chunk_id` — **but the
prompt is only instruction, not enforcement.** LLMs hallucinate. The
provider therefore passes Gemini's output through untouched as candidate
dictionaries and relies entirely on the existing Stage 3 validation layer.

## 7. How `fact_extractor.py` validates evidence

Every candidate runs through `extract_facts(chunks, provider)`, which:

1. Converts candidates via `MedicalFact.from_dict()` (required fields only).
2. Rejects unsupported `fact_type` / `status` / `date_precision` values.
3. Verifies `quote in chunk_text` **verbatim** — paraphrases fail.
4. Verifies the cited `chunk_id` exists and `page_number` matches that
   chunk's real page.

Failures raise `FactValidationError`; nothing invalid becomes a
`MedicalFact`. Tests confirm hallucinated quotes, wrong pages, and
nonexistent chunk ids are all rejected.

## 8. Running tests

```powershell
.venv\Scripts\python -m unittest discover -s tests -v
```

The 16 Stage 4 tests mock the SDK seams (`_get_client`, `_build_config`,
`_call_model`) — **no API key is required and no network request is made**.

Manual run (needs a real key, synthetic PDF only):

```powershell
.venv\Scripts\python scripts/test_gemini_provider.py path\to\synthetic_report.pdf
```

## 9. Use synthetic data for the demo

For the MedTrace hackathon demo, use **synthetic medical records only**.
Never process real patient records without proper governance; never commit
real health data or API keys.

## 10. Scope

This is a hackathon-stage adapter, not a medically safe or production-ready
system. No batching, RAG, vector search, OCR, or chatbot logic is included —
Stage 4 is intentionally only the provider + evidence boundary.
