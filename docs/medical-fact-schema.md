# MedTrace Medical Fact Schema

Stage 3 of the pipeline: **structured medical facts with mandatory evidence**.

```
PDF → page text → evidence chunks → fact extraction → medical facts → evidence references
     (pdf_processor)  (chunker)      (fact_extractor)
```

## Why evidence traceability is mandatory

MedTrace exists to connect every medical fact back to its original source:

> **medical fact → source document → exact page → supporting text**

A fact without verifiable evidence is indistinguishable from a hallucination.
Therefore every fact carries an `evidence` block, and the extraction layer
**rejects** (never silently accepts) any fact whose quote cannot be found
verbatim in the cited source chunk. The extractor never diagnoses, infers, or
invents medical information — it only reflects what the document says.

## Supported fact types

`diagnosis`, `medication`, `lab_result`, `procedure`, `symptom`, `allergy`,
`vital`, `encounter`, `condition_status`

Defined in `ai/fact_extractor.py` as `SUPPORTED_FACT_TYPES`.

## Fact structure

```json
{
  "fact_type": "diagnosis",
  "name": "Sample condition",
  "value": null,
  "unit": null,
  "frequency": null,
  "date": "2024-01-01",
  "date_precision": "day",
  "status": "active",
  "evidence": {
    "document": "sample_report.pdf",
    "page_number": 7,
    "chunk_id": "sample_report-p7-c2",
    "quote": "Sample condition was documented on 2024-01-01."
  }
}
```

| Field | Required | Notes |
|---|---|---|
| `fact_type` | yes | Must be in `SUPPORTED_FACT_TYPES` |
| `name` | yes | Non-empty label of the fact |
| `evidence` | yes | See below — all four sub-fields required |
| `value` | no | Lab values, doses, measurements (string) |
| `unit` | no | Unit for `value` (e.g. `%`, `mg/dL`) |
| `frequency` | no | Medication frequency (e.g. `twice daily`) |
| `date` | no | ISO date at the stated precision — never fabricated |
| `date_precision` | no | `day`, `month`, `year`, `unknown` (default `unknown`) |
| `status` | no | See below (default `unknown`) |

Optional fields stay `null` when the source text does not provide them.

## Evidence requirements

Every fact **must** include:

- `document` — filename of the source document
- `page_number` — 1-based page within that document
- `chunk_id` — the exact chunk the fact was read from
- `quote` — **verbatim substring** of that chunk's text

Invariant checked by `validate_fact()` when source chunks are available:

```
quote in chunk_text          # never a paraphrase
evidence.page_number == chunk.page_number
evidence.chunk_id ∈ source chunks
```

The quote is never normalized, rewritten, or summarized. If a fact cannot be
supported by source text, it is not returned — validation raises
`FactValidationError`.

## Status values

`active`, `historical`, `suspected`, `resolved`, `ruled_out`, `unknown`

The schema distinguishes statements such as *"has X"*, *"may have X"*,
*"history of X"*, and *"X ruled out"*. Uncertain or historical statements are
**never** converted into confirmed active diagnoses, and when the source gives
no signal the status defaults to `unknown` rather than being invented.

## Date handling

| Source says | `date` | `date_precision` |
|---|---|---|
| `2019-08-12` | `"2019-08-12"` | `"day"` |
| `August 2019` | `"2019-08"` | `"month"` |
| `2019` | `"2019"` | `"year"` |
| no date stated | `null` | `"unknown"` |

Validation rules:

- A date requires a matching precision (`day` → `YYYY-MM-DD`, `month` →
  `YYYY-MM`, `year` → `YYYY`), including real calendar-day checks.
- A precision of `unknown` requires `date: null`, and vice versa.
- Missing or partial dates are left as-is — never guessed.

## Provider interface (no LLM wired yet)

`ai/fact_extractor.py` defines `FactExtractionProvider`, a typing `Protocol`:

```python
class FactExtractionProvider(Protocol):
    def extract_facts(self, chunks: list[dict]) -> list[MedicalFact | dict]: ...
```

`extract_facts(chunks, provider)` is the single entry point: it passes the
chunk records to the provider, then validates every candidate fact against the
source chunks. A future LLM adapter (OpenAI, Gemini, Claude, local model)
implements this protocol and owns its own credentials — this module requires
**no API keys** and makes **no network calls**.

## Usage

```python
from ai import extract_pdf_text, chunk_extraction, extract_facts

extraction = extract_pdf_text("record.pdf")
chunks = chunk_extraction(extraction)
facts = extract_facts(chunks, my_provider)   # all quotes verified
```
