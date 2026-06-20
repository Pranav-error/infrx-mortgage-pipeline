# DocCompiler-Lite + PTT
**InfrX 2026 Hackathon — Problem Statement B**
*Team Noobda · REVA University, Bengaluru*

> "Compile the document, then reason on it."

---

## What we're building

A system that takes a large, unstructured multi-page mortgage loan PDF (100–2,000 pages) and gives it structure in two ways:

1. **Logical Pagination** — split the blob into individual document instances with exact start/end pages, including telling apart multiple instances of the same document type back-to-back (e.g. 3× Form 1040, 9× paystubs)
2. **Table Recovery** — reconstruct tables that span page boundaries using Probabilistic Table Threading (PTT), a 5-signal Bayesian belief graph that decides whether two adjacent table fragments belong to the same logical table

---

## Architecture

```
Raw PDF (scan / native / photo)
        │
        ▼
┌─────────────────────────────────────────────┐
│           DOCUMENT COMPILER                 │
│  extract.py                                 │
│  • pdfplumber  → native pages (free, fast)  │
│  • Claude Haiku VLM → scanned pages only    │
│  Output: PageRecord + FragmentRecord per pg │
└────────────────────┬────────────────────────┘
                     │
                     ▼
        DIR — Document Intermediate Representation
        fragment_id · page · bbox · headers · rows
        column_fingerprint · value_types · last_row
                     │
          ┌──────────┴──────────┐
          ▼                     ▼
┌──────────────────┐   ┌─────────────────────────┐
│   classify.py    │   │       stitch.py          │
│                  │   │  PTT Belief Graph        │
│ 27 doc types     │   │  5-signal Bayesian fusion│
│ heuristic first  │   │  per adjacent frag pair  │
│ Haiku parallel   │   │                          │
│ carry-forward    │   │  P > 0.9 → auto-merge    │
│ 84% accuracy     │   │  P 0.7–0.9 → LLM arbiter │
└──────────────────┘   │  P < 0.3 → reject        │
                       └─────────────────────────┘
                                    │
                                    ▼
                      Grounded Output
                      threaded tables · cells · bbox · confidence
```

---

## Repository structure

```
src/
├── extract.py          # Step 2 — page-level extraction (pdfplumber + VLM fallback)
├── classify.py         # Step 3 — document-type classification (27 types, parallel async)
├── eval_classify.py    # Evaluation script — accuracy vs labels.json ground truth
├── run_extraction.py   # CLI runner — runs extract.py on a dataset package
└── test_vlm_sample.py  # VLM sanity test

DataSet/
├── pkg_000000/         # 200-page sample loan file + labels.json ground truth
├── pkg_000001/         # ...
└── pkg_000005/         # 6 packages total

Idea.md                 # Full architecture doc, build plan, demo script
Problem_statement.md    # PS B brief from Infrrd
requirements.txt        # pip dependencies
```

---

## Module details

### `src/extract.py`
Page-level extraction — Step 2 of the pipeline.

- **Native PDF pages** → `pdfplumber` (free, deterministic, zero AI cost)
- **Scanned/photo pages** → Claude Haiku VLM (selective fallback only)

**Output per page:**
```python
PageRecord(page_index, text, has_text_layer, page_height, page_width)
FragmentRecord(fragment_id, page_index, bbox, headers, rows, last_row, page_height, page_width, source)
```

**Run:**
```bash
python3 src/run_extraction.py --pkg "DataSet /pkg_000000"
# add --no-vlm to skip scanned pages (no API key needed)
```

---

### `src/classify.py`
Document-type classification — Step 3 of the pipeline.

**27 document types** matching `labels.json` `doc_type` / `doc_type_label_id` exactly:

| label_id | key | section |
|---|---|---|
| 0 | urla_1003 | application |
| 1 | form_1008 | application |
| 2 | loan_estimate | disclosures |
| 3 | closing_disclosure | disclosures |
| 4 | paystub | income |
| 5 | w2 | income |
| 6 | voe | income |
| 7 | form_1040 | income |
| 8 | schedule_1 | income |
| 9 | schedule_c | income |
| 10 | bank_stmt_checking | assets |
| 11 | bank_stmt_combo | assets |
| 12 | brokerage_stmt | assets |
| 13 | check_image | assets |
| 14 | deposit_receipt | assets |
| 15 | credit_report | credit |
| 16 | du_findings | underwriting |
| 17 | lpa_feedback | underwriting |
| 18 | purchase_contract | property |
| 19 | purchase_addendum | property |
| 20 | options_addendum | property |
| 21 | email_correspondence | misc |
| 22 | letter_of_explanation | misc |
| 23 | gift_letter | misc |
| 24 | insurance_declaration | property |
| 25 | loan_summary | underwriting |
| 26 | filler | misc |

**Classification strategy (3-tier):**

```
Page text
   │
   ├─ continuation page? (transaction rows, no header)
   │    → carry_forward from previous confident type  [0 cost]
   │
   ├─ heuristic keyword confidence ≥ 0.6?
   │    → return immediately                          [0 cost]
   │
   └─ ambiguous → Claude Haiku (parallel async)       [~$0.0003/page]
```

**Output per page:**
```python
{
  "page_index": 0,
  "doc_type": "bank_stmt_checking",
  "doc_type_label_id": 10,
  "confidence": 0.94,
  "method": "heuristic" | "llm" | "carry_forward"
}
```

**Performance (6 packages, 285 digital pages):**

| Package | Pages | Accuracy | Time |
|---|---|---|---|
| pkg_000000 | 52 | 92.3% | 1.4s |
| pkg_000001 | 30 | 73.3% | 3.2s |
| pkg_000002 | 27 | 81.5% | 8.1s |
| pkg_000003 | 65 | 80.0% | 2.1s |
| pkg_000004 | 36 | 83.3% | 2.2s |
| pkg_000005 | 75 | 88.0% | 5.4s |
| **Overall** | **285** | **84.2%** | **~24s total** |

Note: scanned pages (73% of dataset) get text from `extract.py`'s VLM pass first — classifier never calls a model twice on the same page.

**Usage:**
```python
from src.extract import extract_pdf
from src.classify import classify_pages

pages, fragments = extract_pdf("DataSet /pkg_000000/package.pdf")
results = classify_pages(pages)
# → [{"page_index": 0, "doc_type": "bank_stmt_checking", "doc_type_label_id": 10, ...}]
```

---

### `src/eval_classify.py`
Evaluation script — runs classifier against labels.json ground truth.

```bash
python3 src/eval_classify.py                          # all 6 packages
python3 src/eval_classify.py --pkg "DataSet /pkg_000000"  # single package
```

Outputs per-doc-type accuracy, confusion matrix, and LLM call count.

---

## PTT — Probabilistic Table Threading

The core of the system. For every pair of adjacent table fragments, PTT computes 5 independent signals and fuses them via Bayesian belief fusion to decide: **same logical table or different table?**

| Signal | What it measures |
|---|---|
| Header similarity | Cosine over header-token embeddings — survives reworded headers |
| Column-width fingerprint | Normalised column boundaries (0–1) — survives font/header drift |
| Value-type continuity | date/currency/text consistent across boundary? Mismatch = strong negative |
| Spatial flow direction | Does A end near page bottom and B start near page top? |
| Subtotal pattern | Subtotal row on last line = table end (negative signal) |

**Decision thresholds:**
- `P > 0.9` → auto-merge (~85% of pairs, no LLM)
- `P 0.7–0.9` → LLM arbiter (~15%, structured judgment)
- `P < 0.3` → reject edge (different table)

---

## Efficiency story

| Stage | AI used? | Detail |
|---|---|---|
| Native PDF extraction | No | pdfplumber — free |
| Scanned page extraction | Yes (VLM) | Claude Haiku, only pages with no text layer |
| Document classification | Conditional | Heuristic first; Haiku only for ambiguous pages (parallel) |
| PTT 5-signal scoring | No | Pure deterministic math |
| LLM arbiter | ~15% of boundary pairs | Only uncertain edges |

**At 2,000-page scale:** majority of pages never touch a model. Classification of ~500 digital pages runs in ~15–20s with parallel async.

---

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# also: brew install poppler  (for pdf2image on macOS)

export ANTHROPIC_API_KEY=sk-ant-...

# Run extraction on a package
python3 src/run_extraction.py --pkg "DataSet /pkg_000000"

# Evaluate classifier
python3 src/eval_classify.py
```

---

## What's not built yet (honest limits)

- `stitch.py` — PTT belief graph (in progress)
- Semantic compression (10k → 1.5k tokens boilerplate stripping)
- Async VLM extraction (currently sequential — ~7min for 148 scanned pages)
- `column_fingerprint` and `value_types` fields in FragmentRecord (needed by PTT signals 2 & 3)
- Document-instance segmentation (splitting back-to-back same-type docs into instances)
