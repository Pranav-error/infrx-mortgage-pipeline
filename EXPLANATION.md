# DocCompiler — End-to-End Explanation
**Team Noobda · InfrX 2026 · Problem Statement B**

---

## One-Line Pitch

> Treat document stitching as **coreference resolution**, not segmentation — ask "do these pages share the same identity?" instead of "where does this document end?"

---

## The Problem

A mortgage loan file is a **single shuffled PDF** with 100–2,000+ pages containing 10–30 different document types: bank statements, tax forms, pay stubs, loan applications, credit reports, etc. Pages arrive in **random order**, with **no labels**, **no printed page numbers**, and **no table of contents**.

**Our job:** Take this chaos and produce a structured index — what document type is each page, where each document instance starts and ends, and which table fragments across page breaks belong to the same logical table.

---

## Architecture Overview (5 Stages)

```
┌──────────────────────────────────────────────────────────────────┐
│                         INPUT: Shuffled PDF                       │
└───────────────────────────────┬──────────────────────────────────┘
                                │
                                ▼
┌──────────────────────────────────────────────────────────────────┐
│  STAGE 1 — EXTRACTION  (extract.py)                              │
│  pdfplumber (digital) + GPT-4o-mini VLM (scanned, async ×10)    │
│  Output: PageRecord { page_index, text, tables[], fragments[] }  │
└───────────────────────────────┬──────────────────────────────────┘
                                │
                                ▼
┌──────────────────────────────────────────────────────────────────┐
│  STAGE 2 — CLASSIFICATION  (classify.py)                         │
│  FrugalGPT cascade: table-headers → structural → keywords → LLM │
│  Output: { page_index, doc_type, confidence, method } per page   │
└───────────────────────────────┬──────────────────────────────────┘
                                │
                                ▼
┌──────────────────────────────────────────────────────────────────┐
│  STAGE 3 — SEGMENTATION  (segment.py)                            │
│  Page Stream Segmentation: 5-rule pairwise boundary detection    │
│  + global coreference merge for long-range splits                │
│  Output: DocInstance { doc_type, start_page, end_page, attr }    │
└───────────────────────────────┬──────────────────────────────────┘
                                │
                                ▼
┌──────────────────────────────────────────────────────────────────┐
│  STAGE 4 — STITCHING  (stitch.py)                                │
│  Probabilistic Table Threading: 5-signal Naive Bayes fusion      │
│  + LLM arbiter for uncertain zone (P = 0.7–0.9)                 │
│  Output: Thread groups — which table fragments → same table      │
└───────────────────────────────┬──────────────────────────────────┘
                                │
                                ▼
┌──────────────────────────────────────────────────────────────────┐
│  STAGE 5 — RENDER  (render.py)                                   │
│  Assemble labels.json-compatible output with all metadata        │
│  Output: { documents[], pages[], tables[] }                      │
└──────────────────────────────────────────────────────────────────┘
```

---

## Stage 1: Extraction (`extract.py`)

### What it does
Takes a raw PDF and extracts text + table structure from every page.

### How it works

**Digital pages** (have an embedded text layer):
- `pdfplumber` extracts raw text and detects table bounding boxes
- Table cells are extracted with their `(x0, y0, x1, y1)` coordinates — needed later for spatial signals in stitching
- Column headers are captured as ordered lists — needed for fingerprint matching

**Scanned pages** (photographs/images, no text layer):
- Rendered to JPEG at 100 DPI, resized to 512px max width (quality 70)
- Blank pages detected via numpy pixel density check and skipped
- **Tesseract triage**: parallel OCR (8 threads) classifies pages as text-only vs table-bearing
- Text-only pages → lightweight OCR prompt (1500 max tokens)
- Table pages → full extraction prompt with JSON structure (4096 max tokens)
- Sent to **GPT-4o-mini VLM** (Vision Language Model) as base64 images
- **Async with semaphore**: 50 concurrent VLM calls

### Why two passes?
Digital extraction is free and instant. VLM costs money ($0.0003/page) and takes time. We only invoke VLM on pages where pdfplumber finds no text — the `_has_text_layer()` check gates this.

### Output per page
```python
PageRecord {
    page_index: 47,          # physical position in PDF
    text: "Chase Bank...",    # extracted text (digital or VLM)
    fragments: [              # table fragments with geometry
        {
            headers: ["Date", "Description", "Amount", "Balance"],
            rows: [["02/01", "ACH DEPOSIT", "+$1,200", "$5,400"], ...],
            bbox: [72, 200, 540, 680],   # (x0, top, x1, bottom) in PDF points
            page_height: 792,
            column_fingerprint: "0.13|0.35|0.72|0.89",  # normalized x-positions
        }
    ]
}
```

---

## Stage 2: Classification (`classify.py`)

### What it does
Assigns one of **27 document types** to every page. Types include:
- **Application:** `urla_1003`, `form_1008`
- **Disclosures:** `loan_estimate`, `closing_disclosure`
- **Income:** `paystub`, `w2`, `voe`, `form_1040`, `schedule_1`, `schedule_c`
- **Assets:** `bank_stmt_checking`, `bank_stmt_combo`, `brokerage_stmt`, `check_image`, `deposit_receipt`
- **Credit:** `credit_report`
- **Underwriting:** `du_findings`, `lpa_feedback`, `loan_summary`
- **Property:** `purchase_contract`, `purchase_addendum`, `options_addendum`, `insurance_declaration`
- **Misc:** `email_correspondence`, `letter_of_explanation`, `gift_letter`, `filler`

### How it works — FrugalGPT Cascade

The classifier uses a **4-level cascade** — each level is more expensive but more powerful. Most pages resolve at the cheap levels; only the ambiguous ones escalate to LLM.

```
Page text
    │
    ▼
Level 1: TABLE HEADER FINGERPRINT (free, instant)
    "Date | Description | Amount | Balance" → bank_stmt_checking (conf: 0.95)
    "Gross Pay | Federal Tax | Net Pay"     → paystub (conf: 0.95)
    17 predefined column-header patterns.
    If match → DONE, skip all other levels.
    │
    ▼ (no header match)
Level 2: STRUCTURAL COMBO DETECTION (free, instant)
    Detects multi-account summary tables by SHAPE, not bank name:
    Pattern: [account type] + [masked account ****XXXX or ...XXXX] + [two dollar columns]
    2+ such rows → bank_stmt_combo (conf: 0.92)
    Format-agnostic — works on Capital One, Chase, Wells Fargo, any bank.
    │
    ▼ (not combo)
Level 3: KEYWORD HEURISTIC (free, instant)
    ~200 regex patterns across 26 doc types. Each page scored against all types.
    Specificity override: bank_stmt_combo beats bank_stmt_checking when both fire.
    If confidence ≥ 0.60 → DONE
    │
    ▼ (low confidence)
Level 4: CARRY-FORWARD (free, instant)
    If page starts with "DATE DESCRIPTION AMOUNT BALANCE" (table continuation)
    AND previous page had a high-confidence type → inherit that type.
    Works because multi-page bank statements have identical headers on every page.
    │
    ▼ (still unresolved)
Level 5: LLM (GPT-4o-mini / GPT-4o-mini, ~$0.0003/page)
    Two-prompt strategy:
    - Page has ANY mortgage keyword → closed-set prompt (must pick one of 27 types)
    - Page has ZERO mortgage keywords → open-set prompt (can return "unknown")
    PARALLEL ASYNC: all LLM-bound pages fire simultaneously → 50 pages ≈ same latency as 1.
```

### Why this order matters (FrugalGPT pattern)
- Level 1 (headers) is the most precise signal — column structure is universal across banks
- Level 2 (structural) catches combo statements before keywords misclassify them
- Level 3 (keywords) catches most remaining types — ~200 patterns across 26 types
- Level 4 (carry-forward) handles continuation pages that have no identifying headers
- Level 5 (LLM) handles everything else — generalizes to unseen formats

**Result:** Only ~20–35% of pages hit the LLM. The rest are classified for free in microseconds.

### Academic reference
**FrugalGPT** (Chen et al., 2023) — cascading from cheap models to expensive ones, stopping as soon as confidence is sufficient.

### Accuracy: **91.0%** across 40 packages (1,845 digital pages)
- 100% on: `bank_stmt_checking`, `closing_disclosure`, `paystub`, `form_1008`, `du_findings`, `loan_summary`, `brokerage_stmt`, `purchase_contract`, `form_1040`, `w2`
- 92% on `bank_stmt_combo` (was 0% before structural detection fix)

---

## Stage 3: Segmentation (`segment.py`)

### What it does
Groups consecutive pages into **document instances** — detecting where one document ends and the next begins.

### Key reframe: Coreference, not segmentation
Instead of asking "where does this document end?", we ask **"do page N and page N+1 belong to the same document instance?"** This is a pairwise coreference question that can be answered with concrete signals.

### The 5 Boundary Rules (checked for every adjacent page pair)

| Priority | Rule | Signal | Example |
|----------|------|--------|---------|
| 1 | **Doc type changed** | `classify` says page A = `bank_stmt_checking`, page B = `paystub` | Always a boundary — unconditional |
| 2 | **Distinguishing attr changed** | Same type but different instance attribute | `Feb 2024` → `Mar 2024` = new statement |
| 3 | **Fixed-length type exhausted** | Known single-page types exceeded their page count | W2 = 1 page, paystub = 1 page → second page of same type = new instance |
| 4 | **Balance break** | `ending_balance(page_A) ≠ beginning_balance(page_B)` | Ending: $11,200 → Beginning: $8,340 → different bank account |
| 5 | **Institution name changed** | Different bank name in page header | "Chase" → "Wells Fargo" = new document |

### How balance-break works (format-agnostic, no keywords)

Every US bank statement is **legally required** to show beginning and ending balances. We scan every page for both patterns:

```
Page 1: "Beginning Balance: $12,450.00"  ← found opening
        (no ending balance)              ← not the last page

Page 2: (no beginning balance)
        (no ending balance)              ← middle page

Page 3: (no beginning balance)
        "Ending Balance: $11,200.00"     ← found closing
```

The boundary check fires only at the **natural joint** between two statements:

```python
ending_balance(page_A)    = $11,200.00
beginning_balance(page_B) = $11,200.00   ← SAME → same statement, no boundary
                          = $8,340.00    ← DIFFERENT → new statement, boundary fires
```

This works on **any bank, any format** — the words "Beginning Balance" and "Ending Balance" always appear because of the legal requirement.

### Distinguishing attributes

For same-type pages, we extract attributes that make each instance unique:

| Doc type | Attribute extracted |
|----------|-------------------|
| `bank_stmt_checking` | Statement period ("February 2024"), account number ("****1234"), bank name |
| `paystub` | Pay period end date |
| `form_1040` | Tax year (2022, 2023) |
| `w2` | Tax year |

If page A has `Feb 2024` and page B has `Mar 2024` → different instances even though both are `bank_stmt_checking`.

### Second Pass: Global Coreference Merge

**Problem:** The first pass is left-to-right — it can only merge **adjacent** pages. If page 1 and page 2000 belong to the same bank statement (same account, same period) but are separated by 1998 other pages, they come out as two separate instances.

**Fix:** `merge_coreference_instances()` — groups all instances by `(doc_type, distinguishing_attr)`. Any two instances with the same key → same logical document → merged.

```
Before merge:
  bank_stmt#1  p1–p3    [****1234, Feb 2024]   ← fragment 1
  (1998 other pages in between)
  bank_stmt#5  p2001–p2003  [****1234, Feb 2024]   ← fragment 2

After merge:
  bank_stmt#1  p1–p2003  6 pages  [****1234, Feb 2024]  ← reunited
```

### Academic reference
- **Page Stream Segmentation (PSS)** — arXiv 2408.11981
- **DocSplit error taxonomy** — arXiv 2602.15958

---

## Stage 4: Stitching (`stitch.py`)

### What it does
Threads table fragments that span page breaks back into logical tables. A 14-page bank statement might have the transaction table split across 12 page breaks — stitching reconnects them.

### How it works — Probabilistic Table Threading (PTT)

For each pair of **adjacent table fragments** within the same document instance, compute 5 independent signals, then fuse them with **Naive Bayes** to get a posterior probability P(same_table | signals).

### The 5 Signals

| # | Signal | What it measures | Example |
|---|--------|-----------------|---------|
| 1 | **Header similarity** | Token overlap between column headers | "Date \| Description \| Amount \| Balance" vs "Date \| Description \| Amount \| Balance" → 1.0 |
| 2 | **Column fingerprint** | Normalized x-position match of columns | Both have columns at 13%, 35%, 72%, 89% → high match |
| 3 | **Value-type continuity** | Same data types in same columns | Both have [date, text, currency, currency] → match |
| 4 | **Spatial flow** | Fragment A ends near page bottom + Fragment B starts near page top | A ends at 85% height + B starts at 10% → strong continuation signal |
| 5 | **Subtotal pattern** (negative) | "Total" or "Ending Balance" in last row of A | If present → table likely ENDS here, NOT a continuation |

### Naive Bayes Fusion

Each signal produces a **log-likelihood ratio**:

```
LLR_i = log P(signal_i | same_table) - log P(signal_i | diff_table)
```

We model each signal as a Gaussian with empirically calibrated parameters:

```
Signal         μ_same   μ_diff   σ      Weight   Separation
header         0.815    0.407    0.34   1.2      0.408
fingerprint    0.932    0.702    0.20   1.1      0.230
value_type     0.790    0.456    0.27   1.0      0.334
spatial        0.863    0.255    0.31   1.8      0.608  ← BEST signal
subtotal       0.050    0.600    0.15   1.5      (inverted)
```

Fusion formula (Naive Bayes — signals assumed independent):

```
log_odds_posterior = log_odds_prior + Σ (weight_i × LLR_i)
P_posterior = sigmoid(log_odds_posterior)
```

### Decision thresholds

```
P > 0.90  → AUTO-MERGE     (~85% of pairs — zero LLM cost)
P 0.70–0.90 → LLM ARBITER  (~15% — structured judgment call)
P 0.30–0.70 → FLAG         (manual review needed)
P < 0.30  → REJECT         (definitely different tables)
```

### LLM Arbiter (for the uncertain zone)

When the posterior lands in 0.70–0.90, the signals disagree — maybe headers match but spatial flow doesn't, or vice versa. The LLM arbiter:
1. Injects BOTH fragments as context (headers + last 3 rows of A + first 3 rows of B)
2. Shows all 5 signal values
3. Asks: "Same logical table? YES/NO with confidence"
4. Overrides the Bayesian decision if the LLM is confident

This is the **RAG-like pattern** in our pipeline — retrieving the actual fragment content before asking the model to judge.

### Why Naive Bayes, not a neural classifier?

- **Interpretable**: You can see exactly which signal drove the merge/reject decision
- **Calibratable**: We tuned parameters on 87 real fragment pairs from the dataset
- **No training data needed**: Gaussian parameters estimated from summary statistics
- **Fast**: O(1) per pair — no model inference for 85% of decisions

---

## Stage 5: Render (`render.py`)

### What it does
Assembles all pipeline outputs into a single structured JSON matching the `labels.json` schema:

```json
{
  "documents": [
    {
      "doc_instance_id": "bank_stmt_checking#1",
      "doc_type": "bank_stmt_checking",
      "start_page": 1,
      "end_page": 14,
      "page_count": 14,
      "distinguishing_attr": "February 2024, ****1234"
    },
    ...
  ],
  "pages": [
    { "page_index": 0, "doc_type": "w2", "confidence": 0.95, "method": "heuristic" },
    ...
  ],
  "tables": [
    { "table_id": "T1", "thread_group": 0, "pages": [1,2,3,...,14] },
    ...
  ]
}
```

---

## Cost & Efficiency (`cascade.py`)

The **CascadeController** tracks every decision in real time:

| Metric | Value |
|--------|-------|
| Classification accuracy (40 pkgs, 1,845 pages) | **91.0%** |
| LLM call rate (classify) | ~20–35% of pages |
| PTT auto-merge rate | ~85% of fragment pairs |
| PTT LLM arbiter rate | ~15% |
| Spatial signal separation | 0.608 (best single signal) |
| Cost per page (GPT-4o-mini) | ~$0.0002 |
| VLM extraction (1577 pages, 50 concurrent) | ~3–4 min |
| Classification time (2000 pages) | ~15–20s |
| Segmentation time | <0.05s (pure regex, no LLM) |
| **Full pipeline (2049 pages)** | **~6–8 min** |

---

## How Pagination Works (No Printed Page Numbers)

We **never** use printed page numbers. The PDF has a physical page order (0, 1, 2, ...) from the file structure. We assign logical position within each document instance:

```
page_in_doc = page_index - instance.start_page + 1
```

Physical page 47 in a bank statement starting at physical page 45 → `page_in_doc = 3`.

---

## How Scanned Pages Work

```
Scanned page image
      │
      ▼
extract.py: GPT-4o-mini VLM (async, 10 workers)
      │  → raw text + table headers + table rows
      ▼
classify.py ← sees IDENTICAL pipeline to digital pages
```

`classify.py` never knows whether a page was scanned or digital — it just sees text. With 50 async VLM workers and Tesseract triage, 1500+ scanned pages process in ~6–8 min.

---

## How Unknown/Unseen Documents Are Handled

**Two-prompt strategy:**

1. Page has ANY mortgage keyword signal → **closed-set prompt** (must pick one of 27 types)
2. Page has ZERO mortgage signals → **open-set prompt** → can return `"unknown"`

A magazine article with no financial keywords → classified as `unknown`, not forced into a wrong category.

---

## Algorithms Reference

| Algorithm | Where | Academic Name / Reference |
|-----------|-------|--------------------------|
| Confidence-thresholded cascade | classify.py | FrugalGPT (Chen et al. 2023) |
| Pairwise boundary detection | segment.py | Page Stream Segmentation (PSS) — arXiv 2408.11981 |
| Global coreference merge | segment.py | DocSplit error taxonomy — arXiv 2602.15958 |
| Naive Bayes belief fusion | stitch.py | Probabilistic Table Threading (PTT) |
| Async semaphore concurrency | classify.py, extract.py | Standard asyncio pattern |
| Two-prompt open/closed set | classify.py | Domain-gating strategy |
| Table-header fingerprinting | classify.py | Format-agnostic structural classification |
| Gaussian log-likelihood ratios | stitch.py | Standard Bayesian inference |

---

## Project Structure

```
src/
├── extraction/
│   ├── extract.py              # Two-pass PDF extraction (pdfplumber + VLM)
│   └── run_extraction.py       # CLI wrapper
├── classification/
│   ├── classify.py             # 4-level cascade classifier (27 types)
│   └── eval_classify.py        # Accuracy evaluation vs labels.json
├── segmentation/
│   └── segment.py              # PSS boundary detection + coreference merge
├── stitching/
│   └── stitch.py               # Naive Bayes PTT + LLM arbiter
├── pipeline/
│   ├── run_pipeline.py         # End-to-end orchestrator
│   └── cascade.py              # Cost & escalation tracker
└── output/
    └── render.py               # Final JSON output assembler
```

Total: ~4,800 lines of Python + ~920 lines of documentation.

---

## End-to-End Walkthrough Example

**Input:** `pkg_000005/package.pdf` — 170 pages, completely shuffled, no labels.

**Step 1 — Extract:** pdfplumber extracts text from 75 digital pages. 95 scanned pages queued for VLM.

**Step 2 — Classify (2.3s):**
- 8 pages resolved by table-header fingerprint (free)
- 50 pages resolved by carry-forward (free)
- 17 pages sent to LLM (23% escalation rate)
- Result: every page tagged with one of 27 types

**Step 3 — Segment (0.04s):**
- Left-to-right boundary scan: type changes, attribute changes, balance breaks
- Coreference merge: reunite fragments split by intervening documents
- Result: **11 document instances** detected

**Step 4 — Stitch:**
- Within the 52-page bank statement (p0–p110): 12 table fragments across page breaks
- Naive Bayes fusion on each adjacent pair → 85% auto-merge, 15% LLM arbiter
- Result: 12 fragments threaded into 1 logical transaction table

**Step 5 — Render:** Final structured JSON with documents[], pages[], tables[].

**Total wall time:** ~5s for digital pages, ~25s if scanned pages included.
