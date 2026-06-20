# InfrX Pipeline — Algorithm Reference

**Problem:** A 2,000-page mortgage PDF file where pages are in the wrong order, mixed document types, scanned and digital pages combined. Output must be a fully structured `labels.json`-compatible document.

**Solution:** 5-stage pipeline — Extract → Classify → Segment → Stitch → Render

---

## Pipeline Overview

```
Raw PDF (jumbled, multi-doc, mixed scan/digital)
        │
        ▼
┌──────────────────┐
│  1. EXTRACT      │  pdfplumber (digital) + async GPT-4o-mini VLM (scanned)
│                  │  Output: pages[], tables[] (fragments), raw text per page
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  2. CLASSIFY     │  Per-page doc_type assignment
│                  │  3-level cascade: table header → keyword → LLM
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  3. SEGMENT      │  PSS (Page Stream Segmentation)
│                  │  Adjacent-pair boundary detection → DocInstance list
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  4. STITCH       │  PTT (Probabilistic Table Threading)
│                  │  Group by DocInstance → sort → Naive Bayes fusion
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  5. RENDER       │  Merge all stages → labels.json output + demo view
└──────────────────┘
```

---

## Stage 1 — Extraction

**File:** `src/extraction/extract.py`

### What it does
Two-pass extraction on the raw PDF.

**Pass 1 — pdfplumber (serial, free)**
- Runs on every page to detect whether it has a text layer
- Digital pages: extract text + table cells with exact bboxes
- Scanned pages: queued for Pass 2

**Pass 2 — GPT-4o-mini VLM (async parallel, with Tesseract triage)**
- Step A: Batch-render scanned pages to JPEG (512px max width, quality 70) in batches of 50 via poppler
- Step B: Blank page detection — numpy pixel density check skips near-white pages
- Step C: Tesseract triage — parallel OCR (ThreadPoolExecutor(8)) classifies each page as text-only vs table-bearing
- Step D: Fire VLM API calls concurrently (50 workers) via `AsyncOpenAI` + `asyncio.gather` with semaphore
  - Text-only pages → lightweight OCR prompt (max 1500 tokens)
  - Table pages → full extraction prompt with JSON structure (max 4096 tokens)
- VLM returns `page_text` + table structure per page

### Output per page
```json
{
  "page_index": 5,
  "render_mode": "digital | scanned",
  "text": "...",
  "width": 612, "height": 792,
  "has_table": true,
  "fragment_ids": ["frag_5_0"]
}
```

### Output per fragment (table on a page)
```json
{
  "fragment_id": "frag_5_0",
  "page_index": 5,
  "headers": ["Date", "Description", "Deposits", "Balance"],
  "rows": [["01/15", "PAYROLL", "4500.00", "16950.00"]],
  "column_fingerprint": [0.0, 0.2, 0.64, 0.82],
  "value_types": ["date", "text", "currency", "currency"],
  "bbox": [72, 100, 540, 700],
  "cells": [ ... ]
}
```

### Time complexity
| Component | Complexity | Wall time (2000 pages) |
|-----------|-----------|------------------------|
| pdfplumber Pass 1 | O(n) serial | ~50–60 s |
| Batch render (50/batch, 512px JPEG) | O(n) | ~120–130 s |
| Tesseract triage (8 threads) | O(n/8) | ~70 s |
| VLM Pass 2 (50 concurrent, OCR-only split) | O(n/50) | ~120–180 s |
| **Total** | **O(n)** | **~6–8 min** |

### Speed optimizations (implemented)
- **Batch rendering**: 50 pages per batch to avoid OOM, JPEG at 512px max width / quality 70
- **Blank page detection**: Numpy pixel density check — skips near-white pages
- **Tesseract triage**: Parallel OCR (8 threads) classifies pages into text-only vs table-bearing
- **Dual VLM prompts**: Text-only pages use lightweight OCR prompt (1500 tokens); table pages use full extraction (4096 tokens)
- **50 concurrent VLM workers**: Saturates API rate limits
- **GPT-4o-mini backend**: Lower cost per token than Claude Haiku for vision tasks

### What could still be optimised
- **No caching:** Re-running on the same PDF re-extracts everything. A page-level hash cache would make reruns near-instant.
- **VLM truncation:** Very dense pages (>50 rows) can still overflow max_tokens.

---

## Stage 2 — Classification

**File:** `src/classification/classify.py`

### What it does
Assigns a `doc_type` label to every page. Uses a 3-level confidence-thresholded cascade (FrugalGPT pattern).

**Level 1 — Table-header fingerprint (free)**
- Checks if the page's table headers match a known fingerprint (`_HEADER_FINGERPRINTS` dict)
- E.g., `["Date", "Description", "Withdrawals", "Deposits", "Balance"]` → `bank_stmt_checking`
- Fires first, most reliable for structured table pages

**Level 2 — Keyword heuristic (free)**
- Regex patterns per doc type (`DOC_SIGNALS` dict — 26 types, 3–8 patterns each)
- Score = fraction of patterns that match
- If score ≥ 0.60 → assign and skip LLM

**Level 3 — Claude Haiku LLM (costs money)**
- Only for pages below 0.60 heuristic confidence
- All LLM calls run in parallel via `AsyncAnthropic` + semaphore
- Prompt includes first 1200 chars of page text
- Returns `{doc_type, confidence, reasoning}`

**Special case — carry-forward**
- Pages starting with a transaction table header pattern (continuation pages) inherit the previous confident page's doc_type
- No LLM call needed

### Output per page
```json
{
  "page_index": 5,
  "doc_type": "bank_stmt_checking",
  "doc_type_label_id": 10,
  "confidence": 0.86,
  "method": "heuristic | table_header | carry_forward | llm"
}
```

### Time complexity
| Method | Cost | Typical % of pages |
|--------|------|--------------------|
| table_header | O(1) per page | ~30% |
| heuristic | O(patterns) per page | ~55% |
| carry_forward | O(1) per page | ~10% |
| LLM (parallel) | O(1) wall time, $$$ | ~5% |
| **Total** | **O(n)** | — |

> With 2000 pages and 5% LLM rate: **~100 Haiku calls** running in parallel → ~3–5 s wall time, ~$0.03 cost.

### What needs optimising
- **Carry-forward miss rate:** Continuation pages with non-standard table headers fall through to LLM. Could add more regex patterns to `_CONTINUATION_RE`.
- **26 doc types:** Pattern list was built for mortgage files. For other industries, DOC_SIGNALS needs extension. Currently general enough to work on any text-heavy PDF.
- **Confidence calibration:** The 0.60 threshold was set empirically on pkg_000000. A validation run on more packages may reveal it needs tuning per doc_type.

---

## Stage 3 — Segmentation

**File:** `src/segmentation/segment.py`

### What it does
Converts the flat per-page classification list into a list of **DocInstances** — each representing one logical document (e.g., "Chase Bank Statement January 2024").

**Key insight:** Framed as **coreference**, not segmentation.
Ask: "Do page N and page N+1 belong to the same document instance?"
Not: "Where does this document end?"

**Boundary detection — 5 signals (in priority order)**

| Signal | Rule | Example |
|--------|------|---------|
| `doc_type_changed` | Type A ≠ Type B | bank_stmt → w2 |
| `attr_changed` | Same type but different instance identifier | Jan 2024 → Feb 2024 |
| `fixed_length` | Known 1-page types (w2, paystub) | w2 page 2 = new instance |
| `balance_break` | Ending balance ≠ beginning balance | Different accounts concatenated |
| `new_doc_header` | Bank institution name changes on page B | Chase → Wells Fargo |

**Distinguishing attributes extracted by doc_type:**
- `bank_stmt_checking` → statement period (regex) + account number
- `paystub` → pay period end date
- `form_1040` → tax year

### Output (DocInstance)
```python
DocInstance(
    doc_instance_id    = "bank_stmt_checking#3",
    doc_type           = "bank_stmt_checking",
    doc_type_label_id  = 10,
    start_page         = 22,
    end_page           = 25,
    page_count         = 4,
    instance_ordinal   = 3,
    distinguishing_attr= "February 2024",
)
```

### Time complexity
- **O(n)** — single left-to-right pass over sorted page list
- For each adjacent pair: regex match + dict lookup
- No all-pairs comparison

### What needs optimising
- **Balance-break false positives:** If a page has a running total mid-statement (not end-of-period), the balance signal can fire incorrectly. Would need to distinguish "running balance" vs "closing balance" rows.
- **Attribute extraction gaps:** Only 4 doc types have regex attr extractors. Adding patterns for `loan_estimate` (loan amount + lender) and `closing_disclosure` (closing date) would reduce instance over-splitting.
- **Multi-page gaps:** If a jumbled PDF inserts a filler page between two W2 pages of the same instance, the boundary fires and creates 2 instances instead of 1. No current fix without knowing the true page count.

---

## Stage 4 — Stitching (Table Threading)

**File:** `src/stitching/stitch.py`

### What it does
Reconnects table fragments that were split across pages into **logical table threads** — the complete table spanning multiple pages.

**Key insight:** Framed as **coreference**, not segmentation.
Ask: "Do these two fragments share the same logical identity?"
Not: "Where does this table end?"

### The O(n) ordering trick
Naive approach: compare every fragment to every other fragment → O(n²).
Our approach:

```
All fragments
     │
     ▼
Group by DocInstance  ← from Stage 3
     │
     ▼
Sort by page_index within each group
     │
     ▼
Compare adjacent fragments only  ← O(n) total
     │
     ▼
thread_fragments() per group
```

Since fragments of the same logical table must come from the same document instance AND appear on consecutive pages, we never need to compare across instances or non-adjacent pages.

### Naive Bayes fusion — 5 signals

For each adjacent fragment pair, compute posterior P(same_table | signals) via Naive Bayes:

```
P(same | h, c, v, s, sub)  ∝  P(prior) × ∏ P(signal_i | same) / P(signal_i | diff)
```

| Signal | What it measures | Weight if match |
|--------|-----------------|-----------------|
| `header_similarity` | Token overlap between column headers | High |
| `column_fingerprint` | Normalised x-positions of column boundaries | High |
| `value_type_continuity` | Same data types per column (date/currency/text) | Medium |
| `spatial_flow` | Fragment B starts near top of page (continuation) | Medium |
| `subtotal_score` | Fragment A's last row looks like a subtotal | Low (negative) |

### Decision thresholds
```
P > 0.90  →  auto-merge   (no LLM, ~85% of pairs)
P 0.70–0.90 →  LLM arbiter  (Claude Haiku, ~15% of pairs)
P 0.30–0.70 →  flag + review
P < 0.30  →  reject edge
```

### Time complexity
- **O(n)** — each fragment compared to at most one adjacent candidate per DocInstance
- LLM arbiter calls run synchronously (sync Anthropic client)
- For 2000-page file with ~300 fragments: ~299 pair evaluations

### What needs optimising
- **LLM arbiter is synchronous:** If many pairs land in the 0.70–0.90 range, arbiter calls block sequentially. Should be converted to async (same pattern as classify.py).
- **Synthetic column fingerprint:** VLM pages use evenly-spaced fingerprints (`[0.0, 0.2, 0.4, 0.6, 0.8]`). This makes fingerprint signal neutral rather than informative. VLM could be prompted to return column x-positions, or OCR+layout analysis could extract them.
- **Subtotal detection:** Current regex catches "Total", "Grand Total", "Subtotal". Some bank statements use custom labels ("Closing Balance", "Balance Forward") — these are missed and the subtotal signal fires incorrectly.

---

## Stage 5 — Render

**File:** `src/output/render.py`

### What it does
- Assembles all stage outputs into the final `labels.json`-compatible dict
- Fills in all `null` fields on pages and fragments that were deferred to later stages
- Renumbers `row_idx` monotonically across stitched fragments
- Prints the before/after demo view
- Exposes `score_vs_labels()` for ground truth evaluation

### Time complexity
- **O(n + f + t)** where n=pages, f=fragments, t=threads
- Pure in-memory assembly, no I/O

---

## End-to-End Timing (measured)

### Dummy test (20 pages, all digital, no VLM)
| Stage | Time | Notes |
|-------|------|-------|
| Classify | 43.8 ms | 0 LLM calls, all heuristic |
| Segment | 0.6 ms | 15 instances detected |
| Stitch | 0.0 ms | 7 threads |
| Render | 0.1 ms | |
| **Total** | **44.6 ms** | |

### Real run (pkg_000027, 200 pages, mixed)
| Stage | Time | Notes |
|-------|------|-------|
| Extract Pass 1 | ~5 s | pdfplumber on 200 pages |
| Extract Pass 2 | ~60 s | scanned pages, 50 concurrent VLM, Tesseract triage |
| Classify | ~2–4 s | ~5% LLM escalation |
| Segment | <1 s | |
| Stitch | <1 s | |
| Render | <1 s | |
| **Total** | **~70 s** | bottleneck: VLM extraction |

### Measured (2049 pages, 77% scanned)
| Stage | Time | Notes |
|-------|------|-------|
| Extract Pass 1 | ~57 s | 471 digital + 1578 scanned |
| Batch render | ~128 s | 50 pages/batch, 512px JPEG |
| Tesseract triage | ~71 s | 622 table pages, 955 OCR-only |
| VLM Pass 2 | ~120–180 s | 50 concurrent workers |
| Classify | ~15 s | |
| Segment + Stitch + Render | <2 s | |
| **Total** | **~6–8 min** | |

---

## What Needs Optimising (Priority Order)

### ~~1. VLM extraction concurrency~~ ✅ DONE
Raised to 50 concurrent workers. Added Tesseract triage + OCR-only lightweight prompt. Image resize to 512px, JPEG quality 70.

### 2. LLM stitch arbiter is synchronous (MEDIUM impact)
`llm_arbiter()` in `stitch.py` uses the sync `Anthropic` client.
On a heavily fragmented file, 50+ arbiter calls run one by one.
**Fix:** Refactor `thread_fragments()` to async, mirror classify.py's pattern.

### 3. No extraction cache (MEDIUM impact)
Every pipeline run re-extracts every page from scratch.
**Fix:** SHA256(page_index + pdf_mtime) → cache fragment to disk. Skip VLM if cache hit.

### 4. Balance-break false positives (LOW impact, HIGH correctness)
The balance-break signal over-splits bank statements in jumbled PDFs.
**Fix:** Only fire balance-break if the ending balance row is explicitly labelled "Closing Balance" / "Ending Balance".

### 5. Synthetic column fingerprints for VLM pages (LOW impact)
VLM pages all get `[0.0, 0.2, 0.4, ...]` fingerprints → fingerprint signal is always neutral.
**Fix:** Prompt VLM to return column x-positions as percentages.

---

## Data Flow Summary

```
PDF file
  │
  ├─ pdfplumber ──────────────────────────────────────────────────────────┐
  │   digital pages: text + cells + bbox                                  │
  │                                                                        │
  └─ AsyncAnthropic VLM ──────────────────────────────────────────────────┤
      scanned pages: text + table structure                                │
                                                                           ▼
                                                              extraction dict
                                                  {pages[], tables[], charts[]}
                                                                           │
                                              classify_pages() ────────────┤
                                              adds: doc_type per page       │
                                                                           │
                                              segment_documents() ──────────┤
                                              adds: doc_instance_id         │
                                                    boundary flags          │
                                                    page_in_doc             │
                                                                           │
                                              thread_fragments() ────────────┤
                                              per DocInstance group:        │
                                              adds: table_id               │
                                                    page_span              │
                                                    header_repeats         │
                                                                           │
                                              render_output() ──────────────┘
                                              final labels.json dict
                                              {documents[], pages[], tables[]}
```

---

## Key Algorithms Referenced

| Algorithm | Where used | Academic reference |
|-----------|-----------|-------------------|
| FrugalGPT cascade | Classification | Chen et al. 2023, arXiv 2305.05176 |
| PSS boundary detection | Segmentation | arXiv 2408.11981, arXiv 2602.15958 |
| Naive Bayes fusion | Stitching (PTT) | Standard probabilistic inference |
| Coreference framing | Segment + Stitch | Re-framing segmentation as identity matching |
| AsyncIO semaphore | Extract + Classify | Standard concurrency pattern |
