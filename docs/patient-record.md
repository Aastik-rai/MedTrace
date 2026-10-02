# PatientRecord — Longitudinal Record (Stage 5)

## 1. What `PatientRecord` does

`ai/patient_record.py` is an in-memory domain model that organizes
already-validated medical facts into a **longitudinal patient record**:
a chronological timeline of facts with their evidence attached, plus the
unique source documents they came from.

```python
from ai import PatientRecord

record = PatientRecord(
    patient_id="demo-patient-001",   # supplied by the caller
    facts=validated_facts,           # MedicalFact[] from extract_facts()
)

for fact in record.facts:            # chronological order
    print(fact.date, fact.name, fact.evidence.quote)
```

## 2. Why it is separate from Gemini

Stage 5 performs no LLM calls and imports no SDKs — standard library only.
It sits strictly *after* Stage 3 validation in the pipeline:

```
Gemini candidates → Stage 3 validate → MedicalFact[] → Stage 5 PatientRecord
```

The record layer is a pure data/organization concern, so it stays
provider-independent (Gemini today, another model tomorrow) and testable
without credentials or network access.

## 3. Why only validated `MedicalFact` objects are accepted

The constructor accepts **only** `MedicalFact` instances. A raw dict (AI
output that skipped validation) raises `PatientRecordError` with a message
pointing at `extract_facts()`. This reinforces the trust boundary: unvalidated
AI output can never reach the record layer. `PatientRecord` does *not*
re-run Stage 3 validation — that would duplicate the validation system;
`MedicalFact` objects are assumed to have passed it already.

Patient identity follows the same rule: the caller explicitly supplies
`patient_id`. Stage 5 never attempts identity resolution — no merging by
name, dates, symptoms, or AI inference.

## 4. How chronological ordering works

`sort_facts_chronologically(facts)` (used automatically by the constructor)
orders by a deterministic key:

1. Facts with known dates first, ascending.
2. Facts with unknown dates after all known dates.
3. Ties keep their original relative order (Python's stable sort).

Day/month/year precisions compare correctly as ISO prefix strings:
`"2020" < "2020-12" < "2021-05-05"`.

## 5. How unknown dates are handled

Unknown-date facts **stay in the record** and sort last. Their `date`
remains `null` and `date_precision` remains `"unknown"`. Stage 5 never:

- invents a date (no `1900-01-01` sentinels),
- fills dates from the current date,
- modifies a fact's original `date` / `date_precision` (facts are the same
  frozen objects that Stage 3 produced).

## 6. How source documents are collected

Derived from `fact.evidence.document` — never invented filenames — in
first-appearance order of the chronologically sorted facts, deduplicated
deterministically:

```json
["hospital_2024.pdf", "lab_2025.pdf"]
```

An explicit `source_documents` list may be passed instead; every entry must
be the evidence document of at least one fact or construction fails.

## 7. How evidence remains attached to every fact

The record stores the complete `MedicalFact` objects themselves (same
objects, not copies). Nothing is stripped or altered, so doctors can
navigate:

```
PatientRecord → MedicalFact → Evidence → document + page + chunk + exact quote
```

`to_dict()` serializes every fact with its full `evidence` block via the
existing `MedicalFact.to_dict()`.

## 8. How deterministic query helpers work

| Helper | Matching |
|---|---|
| `get_facts_by_type(record, fact_type)` | exact, case-sensitive |
| `get_facts_by_name(record, name)` | exact after Unicode `casefold()` — `"hypertension"` finds `"Hypertension"` |
| `get_facts_for_document(record, document)` | exact on `evidence.document` |

All return lists in the record's chronological order. No fuzzy matching,
no semantic search, no embeddings, no LLM.

## 9. Serialization format

```json
{
  "patient_id": "demo-patient-001",
  "source_documents": ["hospital_2024.pdf", "lab_2025.pdf"],
  "facts": [
    {
      "fact_type": "diagnosis",
      "name": "Hypertension",
      "value": null,
      "unit": null,
      "frequency": null,
      "date": "2024-01-15",
      "date_precision": "day",
      "status": "historical",
      "evidence": {
        "document": "hospital_2024.pdf",
        "page_number": 3,
        "chunk_id": "hospital_2024-p3-c2",
        "quote": "..."
      }
    }
  ]
}
```

`to_dict()` / `to_json()` and `from_dict()` / `from_json()` round-trip
losslessly, reusing `MedicalFact.to_dict()` / `MedicalFact.from_dict()` —
no duplicated serialization logic.

## 10. What Stage 5 intentionally does NOT do

- **No medical reasoning.** It never decides whether a disease is active,
  cured, normal/abnormal, or whether two facts represent the same condition.
- **No truth assessment beyond Stage 3.** It relies on Stage 3's evidence
  validation; it does not re-judge fact truth.
- **No patient merging.** The caller owns `patient_id`; identity is never
  inferred.
- **No deduplication.** Similar or even byte-identical facts are all
  preserved — a future stage may define a medically meaningful policy.
- **No database, RAG, vector search, chatbot, OCR, or LLM calls** — those
  belong to later stages.
