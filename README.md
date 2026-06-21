# InfrX Mortgage Pipeline

---

## 🥈 2nd Place — InfrX 2026 Hackathon · ₹75,000 Prize

> **Team Noobda · REVA University, Bengaluru**
> Problem Statement B — out of all competing teams nationwide.

---

A mortgage loan package is a 100–2,000 page blob: bank statements, pay stubs, tax returns, contracts — all merged into one PDF with zero structure. This pipeline reads that blob and reconstructs it: which pages belong to which document, where each document starts and ends, and which table rows on page 47 are actually the continuation of the table that started on page 44.

---

## Results at a glance

| Metric | Value |
|---|---|
| Hackathon placement | **2nd place — InfrX 2026 · ₹75,000 prize** |
| Classification accuracy | **91.0%** (40 packages · 1,845 pages) |
| Test set | 17 PDFs · 2,295 pages · 742 documents detected |
| Document types supported | 27 mortgage types |
| Tables extracted (single package) | up to 66 logical tables |
| Speed — VLM + Tesseract mode | ~3 min / 100-page scanned PDF |
| Speed — Tesseract-only mode | ~6.5 pages / sec · zero API cost |

---

## End-to-end pipeline overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         INPUT                                           │
│              package.pdf  (100 – 2,000 pages)                           │
│         Native digital  /  Scanned  /  Mixed                            │
└────────────────────────────┬────────────────────────────────────────────┘
                             │
                             ▼
              ┌──────────────────────────┐
              │   STAGE 1 — EXTRACTION   │   extract.py
              │                          │
              │  Per page, decide:       │
              │  ┌─────────────────────┐ │
              │  │ text layer ≥ 30 ch? │ │
              │  └────┬──────────┬─────┘ │
              │     YES          NO      │
              │      ▼           ▼       │
              │  pdfplumber    GPT-4o-   │
              │  (free, fast)  mini VLM  │
              │                (scanned) │
              └────────────┬─────────────┘
                           │  PageRecord  ×  N pages
                           │  FragmentRecord  ×  M table fragments
                           ▼
              ┌──────────────────────────┐
              │ STAGE 2 — CLASSIFICATION │   classify.py
              │                          │
              │  4-tier cascade per page │
              │  (see detail below)      │
              └────────────┬─────────────┘
                           │  { page_index, doc_type,
                           │    confidence, method }  ×  N
                           ▼
              ┌──────────────────────────┐
              │  STAGE 3 — SEGMENTATION  │   segment.py
              │                          │
              │  6 boundary rules find   │
              │  where each document     │
              │  starts and ends         │
              └────────────┬─────────────┘
                           │  DocInstance[]
                           │  (start_page, end_page, doc_type, attr)
                           ▼
              ┌──────────────────────────┐
              │    STAGE 4 — STITCHING   │   stitch.py
              │    (PTT — Probabilistic  │
              │     Table Threading)     │
              │                          │
              │  5-signal Naive Bayes    │
              │  fuses adjacent table    │
              │  fragments into logical  │
              │  tables                  │
              └────────────┬─────────────┘
                           │  LogicalTable[]
                           │  (cells, page_span, headers)
                           ▼
              ┌──────────────────────────┐
              │    STAGE 5 — RENDER      │   render.py
              │                          │
              │  Assemble final JSON +   │
              │  accuracy report vs      │
              │  labels.json ground truth│
              └────────────┬─────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         OUTPUT                                          │
│                    pipeline_output.json                                 │
│   documents[]  ·  tables[]  ·  pages[]  ·  cascade_stats{}             │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Stage 1 — Extraction

**Goal:** turn each PDF page into structured text + table fragments, choosing the cheapest tool that works.

```
                       PDF page
                          │
              ┌───────────▼───────────┐
              │  pdfplumber text extract
              │  len(text) ≥ 30 chars?│
              └───────────────────────┘
                    │           │
                   YES          NO
                    │           │
                    ▼           ▼
             ┌──────────┐  ┌───────────────────────────────┐
             │pdfplumber│  │  Scanned-page path             │
             │          │  │                                │
             │ • text   │  │  1. Render page at 100 DPI     │
             │ • tables │  │  2. Resize to max 512px wide   │
             │ • bbox   │  │  3. JPEG compress (quality 70) │
             │ • col    │  │  4. Send to GPT-4o-mini VLM    │
             │   finger │  │     (50 concurrent workers)    │
             │   print  │  │  5. Parse JSON response:       │
             │          │  │     text + tables + bbox       │
             └──────────┘  └───────────────────────────────┘
                    │                   │
                    └─────────┬─────────┘
                              ▼
                    PageRecord (per page)
                    ┌──────────────────────────────┐
                    │ page_index                   │
                    │ text          (OCR / native) │
                    │ has_text_layer               │
                    │ page_width, page_height       │
                    └──────────────────────────────┘

                    FragmentRecord (per table on page)
                    ┌──────────────────────────────┐
                    │ fragment_id                  │
                    │ page_index                   │
                    │ headers       ["Date", ...]  │
                    │ rows          [[...], ...]   │
                    │ value_types   ["date", ...]  │
                    │ column_fingerprint [0.12,...] │
                    │ bbox          [x0,y0,x1,y1]  │
                    │ last_row      last data row  │
                    └──────────────────────────────┘
```

**Why two paths?** pdfplumber is deterministic and free. VLM costs ~$0.002/page. Scanned pages (photo/fax) have no embedded text so pdfplumber returns nothing — that's the only time VLM is used. On a typical 150-page package, ~30–40% of pages are scanned; the rest are processed for free.

**Tesseract-only mode (`--no-vlm`):** replaces the VLM path with local Tesseract OCR. No API key, no cost, ~6.5 pages/sec. Accuracy is lower on low-quality scans.

---

## Stage 2 — Classification

**Goal:** label every page with one of 27 mortgage document types.

The classifier is a 4-tier cascade — it stops at the first tier that is confident enough, so most pages never reach the LLM.

```
                        page text
                            │
          ┌─────────────────▼──────────────────┐
          │  TIER 1 — Blank / near-blank?       │
          │  len(text.strip()) < 20             │──► filler  (conf=0.85)
          └─────────────────┬──────────────────┘
                            │ non-blank
                            ▼
          ┌──────────────────────────────────────┐
          │  TIER 2 — Continuation page?         │
          │                                      │
          │  _is_continuation(text):             │
          │   • first line matches table header  │
          │   • 5+ data rows                     │
          │   • 2+ rows with $X.XX or $X,XXX     │
          │                                      │
          │  carry_streak < MAX_CARRY (8)        │──► inherit prev type
          └─────────────────┬────────────────────┘   (conf=0.50)
                            │ not a continuation
                            ▼
          ┌──────────────────────────────────────┐
          │  TIER 3A — Table-header fingerprint  │
          │  (22 exact column-set signatures)    │
          │                                      │
          │  e.g.                                │
          │  {date, descr, withdrawals,          │
          │   deposits, balance}                 │──► bank_stmt_checking (0.98)
          │                                      │
          │  {symbol, shares, price, value}      │──► brokerage_stmt     (0.98)
          │                                      │
          │  {earnings, hours, rate,             │
          │   amount, ytd}                       │──► paystub            (0.97)
          │                                      │
          │  ... 19 more fingerprints            │
          └─────────────────┬────────────────────┘
                            │ no fingerprint match
                            ▼
          ┌──────────────────────────────────────┐
          │  TIER 3B — Keyword heuristic scorer  │
          │                                      │
          │  For each of 27 doc types:           │
          │    score = regex_hits / total_patterns│
          │                                      │
          │  score ≥ 0.60  ──────────────────────│──► best-scoring type
          │  (HEURISTIC_SHORTCUT_THRESHOLD)      │    (no LLM)
          │                                      │
          │  Specificity override:               │
          │  bank_stmt_combo beats               │
          │  bank_stmt_checking if both ≥ 0.50   │
          └─────────────────┬────────────────────┘
                            │ still ambiguous
                            ▼
          ┌──────────────────────────────────────┐
          │  TIER 4 — GPT-4o-mini (async)        │
          │                                      │
          │  has ANY mortgage signal?            │
          │    YES → closed-set prompt           │
          │            (must pick from 27 types) │
          │    NO  → open-set prompt             │
          │            (can return "unknown")    │
          │                                      │
          │  Up to 20 concurrent calls           │
          │  Rate-limit retry: 2^attempt backoff │
          │  ~$0.0003 / page                     │
          └──────────────────────────────────────┘

Output per page:
{
  page_index:        int,
  doc_type:          "bank_stmt_checking",
  doc_type_label_id: 10,
  confidence:        0.94,
  method:            "heuristic" | "table_header" | "carry_forward" | "llm"
}
```

**27 supported document types:**

| ID | Key | Category |
|---|---|---|
| 0 | urla_1003 | Application |
| 1 | form_1008 | Application |
| 2 | loan_estimate | Disclosures |
| 3 | closing_disclosure | Disclosures |
| 4 | paystub | Income |
| 5 | w2 | Income |
| 6 | voe | Income |
| 7 | form_1040 | Income |
| 8 | schedule_1 | Income |
| 9 | schedule_c | Income |
| 10 | bank_stmt_checking | Assets |
| 11 | bank_stmt_combo | Assets |
| 12 | brokerage_stmt | Assets |
| 13 | check_image | Assets |
| 14 | deposit_receipt | Assets |
| 15 | credit_report | Credit |
| 16 | du_findings | Underwriting |
| 17 | lpa_feedback | Underwriting |
| 18 | purchase_contract | Property |
| 19 | purchase_addendum | Property |
| 20 | options_addendum | Property |
| 21 | email_correspondence | Misc |
| 22 | letter_of_explanation | Misc |
| 23 | gift_letter | Misc |
| 24 | insurance_declaration | Property |
| 25 | loan_summary | Underwriting |
| 26 | filler | Misc |

---

## Stage 3 — Segmentation

**Goal:** given a sequence of per-page type labels, find the exact page boundaries between documents.

A single left-to-right pass checks 6 rules at each page transition. Any rule firing = new document boundary.

```
     page N          page N+1         Boundary rule
  ─────────────   ─────────────    ──────────────────────────────────

  bank_stmt  →   paystub           RULE 1: doc_type changed
                                           (always a boundary)

  bank_stmt  →   bank_stmt         RULE 2: distinguishing_attr changed
  [Chase,         [Wells Fargo,             Extract attr via regex:
   Feb 2024]       Mar 2024]                bank_stmt → statement period,
                                            account number, bank name
                                            paystub   → pay period end date
                                            w2        → tax year
                                            chapter   → chapter number

  paystub    →   paystub           RULE 3: known fixed length reached
  (page 2 of 1)                            w2=1p, paystub=1p, 1040=2p

  bank_stmt  →   bank_stmt         RULE 4: balance break
  ending bal       beginning bal            ending_balance(N) ≠
  $48,277.02       $12,500.00               beginning_balance(N+1)

  bank_stmt  →   bank_stmt         RULE 5: new institution header
  "Chase Bank"    "Wells Fargo"             (only for bank/brokerage types)

  narrative  →   narrative         RULE 6: chapter header detected
  Chapter 1       Chapter 2                new "Chapter N" heading = new doc


Post-processing: coreference merge
  bank_stmt [Feb 2024, ****1234] pages 1–3
  ...other docs in between...
  bank_stmt [Feb 2024, ****1234] pages 200–205
  ──────────────────────────────────────────────►  merged as one instance


Output — DocInstance:
{
  doc_instance_id:   "bank_stmt_checking#2",
  doc_type:          "bank_stmt_checking",
  doc_type_label_id: 10,
  start_page:        14,
  end_page:          27,
  page_count:        14,
  instance_ordinal:  2,
  distinguishing_attr: "Feb 2024"    ← null if undetermined
}
```

---

## Stage 4 — PTT (Probabilistic Table Threading)

**The core innovation.** A bank statement's transaction table gets cut at every page boundary. Stage 4 reconnects those fragments into one logical table.

Two adjacent fragments might belong to the same table or to different ones — hard rules can't always tell. So we use **Naive Bayes** to fuse five signals into one probability.

### The five signals

```
Fragment A (page N)                    Fragment B (page N+1)
╔═══════════════════════════════╗      ╔═══════════════════════════════╗
║ Date  Desc   Withdraw Deposit ║      ║ Date  Desc   Withdraw Deposit ║
║ ──── ──────  ──────── ─────── ║      ║ ──── ──────  ──────── ─────── ║
║ 2/14  RENT    1200.00         ║      ║ 2/22  PAYROLL          2957.02║
║ 2/15  AMAZON    49.99         ║      ║ 2/23  NETFLIX   15.99         ║
║ 2/16  GAS       45.00         ║  →?  ║ 2/24  GROCERY   127.45        ║
║ ...                           ║      ║ ...                           ║
║ [ends near page bottom]       ║      ║ [starts near page top]        ║
╚═══════════════════════════════╝      ╚═══════════════════════════════╝
         │                                        │
         └──────────── 5 signals ─────────────────┘

SIGNAL 1 — spatial (weight 1.8, best discriminator)
  Does fragment A end near the bottom of its page
  AND fragment B start near the top of its page?
  Score: a_bottom > 70% of page_height  →  +0.5
         b_top    < 30% of page_height  →  +0.5
  P(score|same)=N(0.863,0.31)  P(score|diff)=N(0.255,0.31)

SIGNAL 2 — subtotal (weight 1.5, negative signal)
  Did fragment A end with a TOTAL / SUBTOTAL / GRAND TOTAL row?
  If yes → table ended → they are NOT the same table
  Score: last row matches "total|subtotal|grand total|balance forward"  →  1.0
         second-to-last row matches                                     →  0.7
         no match                                                       →  0.0
  P(score|same)=N(0.05,0.15)   P(score|diff)=N(0.60,0.15)
  (inverted: HIGH score = DIFFERENT tables)

SIGNAL 3 — header (weight 1.2)
  Are the column headers the same?
  Token overlap per column (SequenceMatcher) × column-count bonus
  P(score|same)=N(0.815,0.34)  P(score|diff)=N(0.407,0.34)

SIGNAL 4 — fingerprint (weight 1.1)
  Do column X-positions match?
  Normalized column boundaries (0–1 of page width), mean absolute deviation
  Lower weight because same-width pages across different tables look similar
  P(score|same)=N(0.932,0.20)  P(score|diff)=N(0.702,0.20)

SIGNAL 5 — value_type (weight 1.0, baseline)
  Do column data types match across the break?
  Per-column: date | currency | integer | percent | text
  Fraction of columns with matching type
  P(score|same)=N(0.790,0.27)  P(score|diff)=N(0.456,0.27)
```

### The math

```
prior = 0.60          (60% chance adjacent fragments on consecutive pages
                       are the same table — conservative estimate)

log_odds = log(prior / (1 - prior))

for each signal i:
    llr_i = log P(score_i | same) - log P(score_i | different)
            (Gaussian log-likelihood ratio, constant terms cancel)
    log_odds += weight_i × llr_i

P(same_table) = sigmoid(log_odds)
              = 1 / (1 + exp(-log_odds))
```

### Decision thresholds

```
P(same_table)
    │
    ├── ≥ 0.90  ──►  MERGE        auto-merge, no LLM  (~85% of pairs)
    │
    ├── 0.70–0.90 ►  LLM ARBITER  escalate to GPT-4o-mini  (~15%)
    │                Show headers + last/first rows + 5 signal scores
    │                Ask: "same logical table? true/false"
    │
    ├── 0.30–0.70 ►  FLAG         include but mark uncertain
    │
    └── < 0.30   ──►  REJECT       definitely different table → stop


Hard gate: fragments must be on consecutive pages (page_diff == 1).
           Non-consecutive → REJECT regardless of scores.
```

### Threading algorithm

```
Fragments on pages: [4] [5] [6] [7] [8]
                     A   B   C   D   E

Evaluate pair (A, B):  P=0.96  →  MERGE      thread=[A,B]
Evaluate pair (B, C):  P=0.94  →  MERGE      thread=[A,B,C]
Evaluate pair (C, D):  P=0.88  →  LLM ARBITER
  LLM says: same_table=true    →  MERGE      thread=[A,B,C,D]
Evaluate pair (D, E):  P=0.22  →  REJECT     stop

Result: one logical table spanning pages 4–7 (4 pages, 3 edges)
        fragment E starts a new logical table
```

---

## Stage 5 — Render

Assembles all stage outputs into the final JSON, assigns `table_id` and `doc_instance_id` to every fragment, computes cascade stats (LLM call counts, estimated cost), and optionally compares against `labels.json` ground truth.

---

## Data flow: what moves between stages

```
raw PDF
   │
   │  extract_pdf(pdf_path, use_vlm=True)
   ▼
pages[]          list of PageRecord
                 { page_index, text, has_text_layer, width, height }

tables[]         list of FragmentRecord
                 { fragment_id, page_index, headers, rows,
                   value_types, column_fingerprint, bbox }
   │
   │  classify_pages(pages)
   ▼
classifications[]
                 { page_index, doc_type, doc_type_label_id,
                   confidence, method }
   │
   │  segment_documents(classifications, page_texts)
   ▼
doc_instances[]  list of DocInstance
                 { doc_instance_id, doc_type, start_page, end_page,
                   page_count, distinguishing_attr }
   │
   │  thread_fragments(fragments grouped by doc_instance)
   ▼
threads[]        list of LogicalTable
                 { thread_id, fragments[], page_start, page_end,
                   edges[{ score, decision, signals{} }], flagged }
   │
   │  render_output(all above)
   ▼
pipeline_output.json
   {
     documents[],   ← DocInstance list
     tables[],      ← LogicalTable list with full cells
     pages[],       ← per-page metadata + cell data
     cascade_stats  ← LLM calls, cost, escalation %
   }
```

---

## Web UI

A Next.js app that wraps the pipeline with a live streaming interface.

```
Browser                              Next.js server               Python pipeline
   │                                       │                            │
   │  POST /api/process                    │                            │
   │  (FormData: pdf + mode)               │                            │
   ├──────────────────────────────────────►│                            │
   │                                       │  spawn python3             │
   │                                       │  run_pipeline.py           │
   │                                       ├───────────────────────────►│
   │                                       │                            │
   │◄── SSE stream ────────────────────────┤◄── stdout lines ───────────┤
   │  { type:"log",    message:"..." }     │                            │
   │  { type:"log",    message:"..." }     │                            │
   │  { type:"log",    message:"..." }     │                            │
   │                                       │                            │
   │                                       │  pipeline finishes         │
   │◄──────────────────────────────────────┤◄── exit 0 ─────────────────┤
   │  { type:"done",                       │  read pipeline_output.json │
   │    result:{                           │  send slim slice:          │
   │      documents[],                     │  documents + tables        │
   │      tables[],                        │  (skip 12MB raw JSON)      │
   │      totalPages }}                    │                            │

                 ┌────────────────────────────────────────┐
                 │              DONE STATE                │
                 │                                        │
                 │  ┌──────────────┐  ┌────────────────┐ │
                 │  │  DocSidebar  │  │   PdfViewer    │ │
                 │  │              │  │                │ │
                 │  │ • 27 docs    │  │  PDF.js        │ │
                 │  │ • type badge │  │  inline viewer │ │
                 │  │ • page range │  │                │ │
                 │  │ • click →    │  │ ← jump to page │ │
                 │  │   jump page  │  │                │ │
                 │  └──────────────┘  └────────────────┘ │
                 └────────────────────────────────────────┘

Mode toggle (idle screen):
  ┌────────────────────────────┬──────────────────────────┐
  │  VLM + Tesseract           │  Tesseract — Fast        │
  │  GPT-4o-mini + Tesseract   │  Local OCR only          │
  │  Best accuracy             │  Free · no API key       │
  │  ~3 min / 100 scanned pgs  │  ~6.5 pages/sec          │
  └────────────────────────────┴──────────────────────────┘
```

---

## Output JSON structure

```json
{
  "schema_version": "1.0.0",
  "total_pages": 148,

  "documents": [
    {
      "doc_instance_id":   "bank_stmt_checking#2",
      "doc_type":          "bank_stmt_checking",
      "doc_type_label_id": 10,
      "start_page":        14,
      "end_page":          27,
      "page_count":        14,
      "instance_ordinal":  2,
      "distinguishing_attr": "Feb 2024"
    }
  ],

  "tables": [
    {
      "table_id":     "table_0003",
      "doc_instance_id": "bank_stmt_checking#2",
      "doctype":      "bank_stmt_checking",
      "page_span":    { "start_page": 14, "end_page": 27 },
      "row_count_logical": 268,
      "n_fragments":  14,
      "columns":      [{ "col_idx": 0 }, { "col_idx": 1 }, ...],
      "cells": [
        { "page_index": 14, "row_idx": -1, "col_idx": 0,
          "is_header": true, "text": "Date", "bbox": [...] },
        { "page_index": 14, "row_idx":  0, "col_idx": 0,
          "is_header": false, "text": "02/01", "bbox": [...] }
      ]
    }
  ],

  "pages": [
    {
      "page_index":      14,
      "doc_type":        "bank_stmt_checking",
      "doc_instance_id": "bank_stmt_checking#2",
      "boundary":        "start",
      "has_table":       true,
      "table_ids":       ["table_0003"],
      "render_mode":     "digital",
      "text":            "CHASE BANK\nStatement Period: Feb 1–29, 2024\n..."
    }
  ],

  "cascade_stats": {
    "classification": {
      "total_pages": 148,
      "llm_calls": 8,
      "escalation_pct": "5.4%",
      "estimated_cost_usd": 0.00240
    },
    "stitching": {
      "total_pairs": 12,
      "llm_calls": 2,
      "escalation_pct": "16.7%",
      "estimated_cost_usd": 0.00060
    }
  }
}
```

---

## Cost & speed profile

| Stage | AI used | When | Cost |
|---|---|---|---|
| pdfplumber extraction | No | All digital pages | Free |
| Tesseract triage | No | All scanned pages (pre-filter) | Free |
| GPT-4o-mini VLM extraction | Yes | Scanned pages only | ~$0.002/page |
| Classification heuristic | No | Most pages (91% of tier-3 pages) | Free |
| Classification LLM | Yes | Ambiguous pages only (~5–8%) | ~$0.0003/page |
| PTT 5-signal fusion | No | All adjacent fragment pairs | Free |
| PTT LLM arbiter | Yes | Uncertain edges (~15% of pairs) | ~$0.0002/call |

**At 150 pages (typical package):** total AI cost ≈ $0.05–0.15 depending on scan ratio.

**Speed optimisations applied:**

| Optimisation | Effect |
|---|---|
| Tesseract pre-triage | Text-only scanned pages skip full VLM; get lightweight OCR prompt (1,500 tokens vs 4,096) |
| Image resize to 512px wide | ~4× fewer vision tokens vs full resolution |
| JPEG quality 70 | Smaller payload, faster upload |
| Batch rendering (50 pages/batch) | Avoids OOM on 2,000-page PDFs |
| 50 concurrent VLM workers | Saturates API rate limits |
| Carry-forward (capped at 8) | Continuation pages skip classifier entirely |
| Blank page detection | Pixel density check, skips near-white pages |
| Parallel Tesseract (8 threads) | Overlaps OCR with other work |

---

## Repository structure

```
src/
├── extraction/
│   ├── extract.py           Two-pass extraction (pdfplumber + GPT-4o-mini VLM)
│   └── run_extraction.py    CLI wrapper
├── classification/
│   ├── classify.py          4-tier cascade classifier (27 types, async)
│   └── eval_classify.py     Accuracy eval vs labels.json ground truth
├── segmentation/
│   └── segment.py           PSS boundary detection + coreference merge
├── stitching/
│   └── stitch.py            Naive Bayes PTT + LLM arbiter
├── pipeline/
│   └── run_pipeline.py      End-to-end orchestrator
└── output/
    └── render.py            Final JSON assembler

web/
├── app/
│   ├── page.tsx             Main UI (upload → live logs → split view)
│   └── api/
│       ├── process/         SSE endpoint — streams pipeline output
│       └── pdf/[id]/        Serves PDF to the inline viewer
└── components/
    ├── PdfViewer.tsx         PDF.js viewer with page navigation
    ├── DocSidebar.tsx        Document list with type colours
    └── UploadZone.tsx        Drag-and-drop upload

DataSet /
└── pkg_000000/ … pkg_000039/   40 packages — labels.json ground truth

pagination-test/
└── doc_000.pdf … doc_016.pdf   17 test PDFs (2,295 pages)

results/
├── summary.json                Aggregated: 17 PDFs, 742 docs
├── README.md                   How to verify results
├── Pipeline-Architecture.pdf   Architecture slide deck
└── doc_000/ … doc_016/
    └── pipeline_output.json

Algorithm.md          Deep-dive algorithm reference
requirements.txt      Python dependencies
run.sh                Convenience runner
```

---

## Setup

```bash
# 1. Python environment
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# macOS dependencies
brew install poppler tesseract

# 2. API key (only needed for VLM mode)
export OPENAI_API_KEY=sk-...

# 3. Run pipeline on a test PDF
python3 src/pipeline/run_pipeline.py \
  --pdf pagination-test/doc_000.pdf \
  --out out.json

# Tesseract-only — no API key needed
python3 src/pipeline/run_pipeline.py \
  --pdf pagination-test/doc_000.pdf \
  --out out.json \
  --no-vlm

# Evaluate classifier accuracy on all 40 packages
./run.sh eval

# 4. Web UI
cd web && npm install
cp .env.local.example .env.local   # add OPENAI_API_KEY
npm run dev                         # http://localhost:3000
```

---

## Key thresholds — quick reference

| Component | Constant | Value | Meaning |
|---|---|---|---|
| Extract | `MIN_TEXT_CHARS` | 30 | Below this → page is scanned |
| Extract | `VLM_RENDER_DPI` | 100 | Resolution for VLM rendering |
| Extract | `VLM_MAX_WIDTH` | 512 px | Max image width (token savings) |
| Extract | `DEFAULT_WORKERS` | 50 | Concurrent VLM async workers |
| Classify | `HEURISTIC_SHORTCUT_THRESHOLD` | 0.60 | Skip LLM if heuristic ≥ this |
| Classify | `_MAX_CARRY` | 8 | Max consecutive carry-forwards |
| Classify | `_MAX_TEXT` | 1200 chars | Text sent to LLM |
| Stitch | `PRIOR_SAME` | 0.60 | Prior P(same table) |
| Stitch | `THRESHOLD_MERGE` | 0.90 | Auto-merge threshold |
| Stitch | `THRESHOLD_LLM` | 0.70 | Escalate-to-LLM threshold |
| Stitch | `THRESHOLD_REJECT` | 0.30 | Reject-edge threshold |
| Stitch | `spatial` weight | **1.8** | Best SNR signal |
| Stitch | `subtotal` weight | **1.5** | Negative signal (table end) |
| Stitch | `header` weight | 1.2 | Column header similarity |
| Stitch | `fingerprint` weight | 1.1 | Column-width match |
| Stitch | `value_type` weight | 1.0 | Data type continuity |

---

*Team Noobda · REVA University · InfrX 2026 · 2nd Place · ₹75,000*
*See `Team_Noobda_REVA University.pdf` for the full presentation. See `Algorithm.md` for deeper algorithm notes.*
