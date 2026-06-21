# InfrX Mortgage Pipeline
**InfrX 2026 Hackathon — Problem Statement B · 2nd Place**
*Team Noobda · REVA University, Bengaluru*

> Took home **2nd place** out of all competing teams at InfrX 2026 for building an end-to-end mortgage document compiler that handles 100–2,000 page PDFs with 91%+ classification accuracy.

---

## What it does

Takes a large, unstructured multi-page mortgage loan PDF and gives it structure in two ways:

1. **Logical Pagination** — splits the blob into individual document instances with exact start/end pages, including multiple back-to-back instances of the same type (e.g. 3× Form 1040, 9× paystubs)
2. **Table Recovery** — reconstructs tables that span page boundaries using **Probabilistic Table Threading (PTT)**, a 5-signal Naive Bayes model that decides whether two adjacent table fragments belong to the same logical table

---

## Results

| Metric | Value |
|---|---|
| **Hackathon placement** | **2nd place — InfrX 2026** |
| Classification accuracy | **91.0%** (40 packages, 1,845 pages) |
| Test set | 17 PDFs, 2,295 pages, 742 documents detected |
| Pipeline speed (VLM mode) | ~3 min for a 100-page scanned PDF |
| Pipeline speed (Tesseract mode) | ~6.5 pages/sec, no API cost |
| Doc types supported | 27 mortgage document types |
| Tables extracted (sample) | 66 logical tables from a single 150-page package |

---

## Architecture

```
Raw PDF (scan / native / photo)
        │
        ▼
┌─────────────────────────────────────────────┐
│           STAGE 1 — EXTRACTION              │
│  extract.py                                 │
│  • pdfplumber  → native pages (free, fast)  │
│  • GPT-4o-mini VLM → scanned pages only     │
│  • Tesseract triage → text vs table pages   │
│  Output: PageRecord + FragmentRecord/page   │
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
│  STAGE 2         │   │  STAGE 3 — PTT           │
│  classify.py     │   │  stitch.py               │
│                  │   │                          │
│  27 doc types    │   │  5-signal Bayes fusion   │
│  heuristic first │   │  per adjacent frag pair  │
│  GPT-4o-mini par │   │                          │
│  carry-forward   │   │  P ≥ 0.75 → merge        │
│  91% accuracy    │   │  P 0.5–0.75 → LLM judge  │
└──────────────────┘   │  P < 0.3  → reject       │
                       └─────────────────────────┘
                                    │
                                    ▼
                         pipeline_output.json
                   documents[] · tables[] · pages[]
```

---

## Repository structure

```
src/
├── extraction/
│   ├── extract.py          # Two-pass PDF extraction (pdfplumber + GPT-4o-mini VLM)
│   └── run_extraction.py   # CLI wrapper
├── classification/
│   ├── classify.py         # 4-level cascade classifier (27 types, parallel async)
│   └── eval_classify.py    # Accuracy evaluation vs labels.json ground truth
├── segmentation/
│   └── segment.py          # PSS boundary detection + coreference merge
├── stitching/
│   └── stitch.py           # Naive Bayes PTT + LLM arbiter
├── pipeline/
│   └── run_pipeline.py     # End-to-end orchestrator
└── output/
    └── render.py           # Final JSON output assembler

web/                        # Next.js web UI — upload PDF, view results live
├── app/
│   ├── page.tsx            # Main UI (upload, log stream, split view)
│   └── api/
│       ├── process/route.ts  # SSE endpoint — streams pipeline progress
│       └── pdf/[id]/route.ts # Serves PDF for viewer
└── components/
    ├── PdfViewer.tsx        # PDF.js viewer with page navigation
    ├── DocSidebar.tsx       # Document list sidebar with type colours
    └── UploadZone.tsx       # Drag-and-drop upload

DataSet /
├── pkg_000000/ … pkg_000039/   # 40 packages with labels.json ground truth

pagination-test/
├── doc_000.pdf … doc_016.pdf   # 17 test PDFs (2,295 pages total)

results/
├── summary.json                # Aggregated results for all 17 test PDFs
├── README.md                   # Verification guide
├── Pipeline-Architecture.pdf   # Architecture deck
└── doc_000/ … doc_016/
    └── pipeline_output.json    # Full output per test PDF

Algorithm.md            # Detailed algorithm reference
requirements.txt        # Python dependencies
run.sh                  # Convenience runner (eval / pipeline modes)
```

---

## Classification — 3-tier cascade

Every page goes through three tiers in order, stopping at the first confident answer:

```
Page text
   │
   ├─ 1. Table-header fingerprint (27 hardcoded column combos)
   │       e.g. {date, description, withdrawals, deposits, balance} → bank_stmt_checking @ 0.98
   │       [0 cost, runs first]
   │
   ├─ 2. Keyword heuristic score ≥ 0.6
   │       hits / total_patterns for each of 27 doc types
   │       [0 cost]
   │
   ├─ 3. Continuation check (carry-forward)
   │       blank / mid-table pages inherit last confident type
   │       capped at 8 consecutive carries to prevent boundary bleed
   │       [0 cost]
   │
   └─ 4. GPT-4o-mini (parallel async, closed-set or open-set prompt)
           only ambiguous pages that pass all above without a hit
           [~$0.0003/page]
```

**27 supported document types:**

| ID | Type | Category |
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

## PTT — Probabilistic Table Threading

The core innovation. For every pair of adjacent table fragments, PTT fuses 5 independent signals via Naive Bayes belief fusion to decide: **same logical table or different table?**

```
log_odds      = log(prior) + Σ weight × log( P(signal|same) / P(signal|different) )
P(same_table) = sigmoid(log_odds)
```

| Signal | What it checks | Weight |
|---|---|---|
| `spatial` | Do column X-positions line up across the break? | **1.8** |
| `subtotal` | Did the last page end with a TOTAL row? (table ended) | **1.5** (negative) |
| `header` | Are the column headers the same? | 1.2 |
| `fingerprint` | Do column widths match? | 1.1 |
| `value_type` | Do column data types match (dates, $, text)? | 1.0 |

Weights are calibrated from **40 same-table + 47 different-table pairs** measured on `pkg_000000`:
- `weight = (mu_same − mu_diff) / sigma` — higher SNR → higher weight
- `spatial` scores 1.8 because column misalignment has a 0.608 separation gap, the strongest discriminator

**Thresholds:**
- `P ≥ 0.75` → auto-merge (~85% of pairs, zero LLM cost)
- `P 0.5–0.75` → LLM arbiter (~15%, structured judgment call)
- `P < 0.3` → reject (different table)

---

## Web UI

A Next.js app that wraps the pipeline with a live streaming interface:

- Drag-and-drop PDF upload
- Choose mode: **VLM + Tesseract** (accurate) or **Tesseract-only** (fast, free)
- Live log stream via SSE while pipeline runs
- Split view: PDF viewer + document sidebar with type-colour coded sections
- Click any document in the sidebar to jump to that page in the viewer

```bash
cd web
npm install
npm run dev      # http://localhost:3000
```

Set `OPENAI_API_KEY` in `web/.env.local` for VLM mode.

---

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# macOS: brew install poppler tesseract

export OPENAI_API_KEY=sk-...

# Run full pipeline on a test PDF
python3 src/pipeline/run_pipeline.py --pdf pagination-test/doc_000.pdf --out out.json

# Tesseract-only (no API key needed)
python3 src/pipeline/run_pipeline.py --pdf pagination-test/doc_000.pdf --out out.json --no-vlm

# Evaluate classifier on all 40 packages
./run.sh eval

# Run full pipeline on a package
./run.sh pipeline pkg_000005
```

---

## Speed optimisations

| Optimisation | Impact |
|---|---|
| Tesseract triage | Text-only pages get lightweight OCR prompt (1,500 tokens) instead of full extraction (4,096 tokens) |
| Image resize to 512px | Reduces vision API token count ~4× vs full resolution |
| JPEG quality 70 | Smaller payloads → faster upload |
| Batch rendering (50 pages/batch) | Avoids OOM on large PDFs |
| 50 concurrent VLM workers | Saturates API rate limits |
| Blank page detection | Pixel density check skips near-white pages |
| Parallel Tesseract | `ThreadPoolExecutor(8)` for OCR triage |
| Carry-forward (capped at 8) | Continuation pages skip classifier entirely |

---

## Team

**Team Noobda — REVA University, Bengaluru**
InfrX 2026 Hackathon · Problem Statement B · **2nd Place**

See `Team_Noobda_REVA University.pdf` for the full presentation deck.
See `Algorithm.md` for the detailed algorithm reference.
