# Deterministic Fact Retrieval (Stage 6)

## Purpose

Stage 6 answers *"which existing facts are relevant?"* — nothing more. It is
a small, provider-independent layer over the `PatientRecord` (Stage 5) that
returns already-validated `MedicalFact` objects for questions like *"facts
mentioning blood pressure"* or *"has this patient had hypertension before?"*.

## Relationship to PatientRecord

`PatientRecord` is the **source of truth**. Retrieval reads `record.facts`
only: it never writes, never re-sorts, never creates a `MedicalFact`, and
never touches evidence. The pipeline stays:

```
PDF → pdf_processor → chunker → Gemini → fact_extractor (Stage 3 validation)
    → patient_record (Stage 5) → retrieval (Stage 6)
```

## Deterministic retrieval

Matching = `strip()` + `casefold()` + **substring** test over plain fields.
Same input, same output, every time. No ranking: results are returned in
the record's established chronological order (known dates before unknown
dates, stable ties).

`search_facts(record, query, fact_type=None, limit=None)`:

- **Searched fields** (`SEARCHABLE_FIELDS`): `name`, `value`, `unit`,
  `frequency`, `status`, `fact_type`. Dates are excluded — they are
  structured metadata, better served by date-range queries later.
- **`fact_type`** filters against `SUPPORTED_FACT_TYPES` (Stage 3); an
  unsupported value raises `RetrievalError`.
- **`limit`** must be a positive integer (`None` = all matches).
- Empty / whitespace-only queries raise `RetrievalError`.
- No match → `[]` (never a fabricated fact).

```python
search_facts(record, "138/88")                    # matches the value field
search_facts(record, "hypertension", fact_type="diagnosis")
search_facts(record, "hemoglobin", limit=5)
```

## find_condition_history

`find_condition_history(record, condition_name, limit=None)` retrieves the
condition-related facts (existing types `diagnosis` + `condition_status`
only) whose **name** contains the query, case-insensitively. It is the
building block for *"Has this patient had hypertension before?"* — but it
returns facts, not a natural-language answer, and never converts statuses
to a boolean.

```python
find_condition_history(record, "hypertension")
# -> diagnosis/historical, diagnosis/active, condition_status/resolved, ...
```

## Evidence preservation

Every returned object is the record's **original** fact (identity, not a
copy). `document`, `page_number`, `chunk_id`, and `quote` are exactly as
stored — retrieval performs no evidence rewriting, quote generation,
summarization, or inference. Only Stage 3 ever validates evidence; Stage 6
can only return facts that already passed it.

## Why no LLM

Retrieval must be reproducible, testable offline, and free of hallucination
risk at the final mile. An LLM here could paraphrase a quote, drop a fact,
or invent one — exactly what the trust boundary exists to prevent.

## Why synonyms and semantic search are deferred

No synonym dictionary (`"HTN"` does **not** match `"hypertension"`), no
fuzzy matching, no embeddings, no vector store. Predictable retrieval now;
semantic expansion is a deliberate future stage where it can be evaluated,
not smuggled in.

## Limitations (explicit)

- **Retrieval does not diagnose** and makes no medical conclusions.
- Absence of a condition from results does **not** mean the patient never
  had it — it means no stored fact matched this deterministic query.
- Statuses are passed through untouched (`historical` stays `historical`);
  nothing is inferred about whether a condition is currently true.
