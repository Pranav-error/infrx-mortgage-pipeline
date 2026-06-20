# DocCompiler-Lite: Logical Pagination & Cross-Page Table Threading

**Event:** InfrX 2026 — Build Round
**Problem Statement:** B — Structuring the 2,000-Page File: Tables & Logical Pagination
**Spirit of it:** Give structure to the blob, efficiently

---

## 1. One-line pitch

We treat page-boundary stitching as a **coreference problem**, not a segmentation problem — instead of asking "where does this table end," we ask "do these two fragments share the same logical identity" — resolved by a lightweight, mostly-deterministic multi-signal score, with AI used only where it's actually necessary.

---

## 2. The problem we're solving

A mortgage loan file can run to 2,000 pages: dozens of distinct documents scanned and merged into one continuous stream, no table of contents, no markers. Two things block everything downstream:

1. **Logical pagination** — knowing which pages belong together as one document instance, including telling apart several instances of the *same* document type sitting back to back (e.g. three years of Form 1040s, each 2 pages, stacked one after another — must resolve into 3 distinct instances, not one smeared 6-page block).
2. **Table structuring across page breaks** — bank-statement transaction histories, payment schedules, and itemized fees often span several pages, with repeating headers, interrupting subtotals, and column drift. A naive stitcher assumes column 3 on page 12 = column 3 on page 13. It's often wrong, and wrong silently.

The brief is explicit that **efficiency is part of the grade**: approaches should lean on smaller/open-source models and modest compute, not heavy dependence on the largest hosted models, and should be able to plausibly run at 2,000-page scale.

---

## 3. Our core insight

Two ideas, combined:

**(1) Compile before you reason.** Don't reason on raw OCR text. Extract each page into a small structured "fragment" object first (headers, bounding box, column positions, value types, last-row type) — a lightweight intermediate representation, not a full document-compiler IR, but the same philosophy: structure first, judgment second.

**(2) Stitching is coreference, not segmentation.** Segmentation asks "where does this table end?" — a single boundary-finding question that breaks under header drift, skew, and repeated headers. Coreference asks "do fragment A and fragment B share the same logical table identity?" — a pairwise same/different decision, which degrades gracefully: a wrong answer on one pair doesn't cascade into the rest of the document, and a low-confidence pair can be flagged loudly instead of merged silently.

**(3) Deterministic first, AI only at the edges.** Most of the actual decision-making is plain math (string similarity, bounding-box comparison, position checks) — zero cost, zero latency, fully auditable. AI is used only in three narrow, necessary places (see Section 6). This directly answers the brief's efficiency ask.

---

## 4. Full architecture (end to end)

```
[1] RAW PDF INPUT
    A 100+ page mortgage file — mix of native text PDFs and scanned/photo pages
         |
         v
[2] PAGE-LEVEL EXTRACTION
    Native PDF page  -> pdfplumber: text, table cells, bounding boxes (NO AI, free, fast)
    Scanned/photo page -> VLM call, selective fallback ONLY for these pages
    Output: one "fragment" object per detected table per page
      { fragment_id, page, bbox, headers, rows, last_row, page_height }
         |
         v
[3] DOCUMENT-TYPE CLASSIFICATION (per page)
    Cheap keyword/heuristic match first (e.g. "W-2" text present -> likely W-2)
    LLM fallback ONLY when heuristic is ambiguous
         |
         v
[4] DOCUMENT-INSTANCE SEGMENTATION
    Consecutive same-type pages grouped into instances
    Each instance gets a distinguishing attribute (tax year, statement date)
    -> splits "6 pages of 1040" into 3 separate 2-page instances
    Output: [{doc_type, page_start, page_end, distinguishing_attr}, ...]
         |
         v
[5] CROSS-PAGE FRAGMENT MATCHING (the core mechanism)
    For tables that might span a page boundary, compute 3 signals
    between adjacent fragments:
      - header_similarity()           -> do column headers match?
      - column_position_similarity()  -> do column widths/counts match?
      - spatial_flow_score()          -> does A end near page bottom,
                                          B start near top of next page?
    Fuse into one weighted score (fusion_score)
    NO AI — pure deterministic math
         |
         v
[6] THRESHOLD DECISION
    score >= 0.75        -> AUTO-MERGE (same logical table, stitch silently)
    0.5 <= score < 0.75  -> FLAG "needs review" (loud failure, not silent)
    score < 0.5          -> DIFFERENT TABLE (stop, don't merge)
    [Optional, if time allows: LLM arbiter call ONLY on flagged edges]
         |
         v
[7] THREADED OUTPUT
    - Logical tables: continuous rows across pages, correctly stitched
    - Document instance list: [{doc_type, page_start, page_end, distinguishing_attr}]
    - Flagged edges: uncertain merges with their signal breakdown, for human review
         |
         v
[8] GROUNDED OUTPUT / DEMO VIEW
    Before: raw jumbled pages
    After: clean instance list + stitched tables
    Every cell traceable to {page, bbox}
    Flagged cases shown explicitly with their 3 signal scores
```

---

## 5. The two classifications — kept separate (important for Q&A)

There are two distinct decisions in this system. Do not conflate them when explaining to judges.

| | What it answers | Where | AI involved? |
|---|---|---|---|
| **Document-type classification** | "What kind of document is this page?" (paystub / W-2 / 1040 / bank statement) | Step 3, once per page | Heuristic first, LLM fallback only if ambiguous |
| **Fragment coreference (stitching)** | "Do these two adjacent table fragments belong to the same logical table, or different ones?" | Steps 5-6, pairwise between adjacent fragments | No — pure deterministic signal fusion |

Table "classification" in the category-labeling sense does not happen separately — a table inherits its document type from the page-level classification it falls inside. What happens independently is the **pairwise continuity decision**, which is the actual core insight of this project.

---

## 6. Exactly where AI is used (and where it deliberately isn't)

| Stage | AI used? | Detail |
|---|---|---|
| Native PDF extraction | **No** | `pdfplumber` — deterministic |
| Scanned/photo page extraction | **Yes** | VLM call, selective — only pages with no extractable text layer |
| Document-type classification | **Conditional** | Heuristic/keyword match first; LLM fallback only on ambiguous pages |
| Instance segmentation (distinguishing attribute) | **Conditional** | Regex/pattern match first; LLM fallback only on unexpected formats |
| 3-signal fragment scoring | **No** | Pure math — string similarity, bbox comparison, position checks |
| Threshold decision | **No** | Numeric comparison |
| Flagged/uncertain edge arbitration | **Optional / cuttable** | LLM arbiter call, only on the ~15-20% gray-zone pairs — can be replaced with "flagged for human review" if time-constrained |

**The pitch line:** *"AI only touches three places: reading scanned pages with no text layer, classifying a page when a keyword match fails, and optionally arbitrating the small fraction of page-boundary cases where our three structural signals disagree. Everything that does the actual hard work — deciding whether two table fragments are the same logical table — is pure deterministic math. At 2,000 pages, most of the file never touches a model at all."*

---

## 7. What we deliberately cut from the bigger vision (say this openly in the deck)

| Bigger version (future work) | What we built instead (tonight) |
|---|---|
| 5-signal Bayesian belief fusion | 3 signals (header similarity, column position, spatial flow), simple weighted sum |
| LLM arbiter on every uncertain edge | Flagged for human review; LLM arbiter added only if time allows |
| Correction bank -> few-shot RAG -> distilled fine-tuned Fragment Detector | Not built — named explicitly as Month 1 / Year 1 future work |
| Per-customer compression rule tuning | Not built — out of scope for a single demo file |
| Fixed accuracy targets (F1 > 0.90, row-boundary > 0.92, false-merge < 2%) | We report actual measured numbers on our own test set, however small, instead of claiming targets we haven't verified |

Naming these cuts explicitly is not a weakness — the rubric rewards "naming your tradeoffs and limits openly" and punishes "claims with no evidence behind them."

---

## 8. Build plan — hour by hour (5-6 hour budget)

| Time | Task | Risk |
|---|---|---|
| 0:00–0:30 | Get sample loan file(s), inspect real page-boundary cases by hand | Low |
| 0:30–1:30 | Page extraction: `pdfplumber` for native pages, VLM fallback wired for scanned pages | Medium — native vs. scanned paths differ |
| 1:30–2:30 | Build fragment objects (headers, bbox, column positions, value types, last-row type) | Medium |
| 2:30–3:30 | Implement 3-signal scorer + weighted fusion score | Medium — conceptually simple once fragments exist |
| 3:30–4:00 | Threshold + auto-merge / flag logic; stitch matched fragments into one logical table | Low |
| 4:00–5:00 | Document-instance segmentation (telling apart back-to-back same-type documents) | **High** — needs reliable distinguishing-attribute extraction |
| 5:00–5:45 | Output/visualization: before (jumbled pages) vs after (clean instances + stitched tables) | Low–Medium |
| 5:45–6:00 | Test on a document the team hasn't tuned against; fix what breaks; prep demo script | High if skipped — do not skip this |

**Cut list if running behind, in order of what to drop first:**
1. LLM arbiter on flagged edges (just show "flagged for review" instead)
2. Spatial flow signal (keep header + column signals only)
3. Fancy visualization polish (a clean table/list is enough, doesn't need to be fancy)

**Never cut:** testing on an unseen document before demo time. This is the single most commonly skipped step and the riskiest one to skip, since judges may hand you a fresh document live.

---

## 9. Core code skeleton

### 9.1 Page extraction (`extract.py`)

```python
import pdfplumber

def extract_page_fragments(pdf_path):
    fragments = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            tables = page.find_tables()
            for t_idx, table in enumerate(tables):
                rows = table.extract()
                if not rows or len(rows) < 1:
                    continue
                headers = rows[0]
                bbox = table.bbox  # (x0, top, x1, bottom)

                fragments.append({
                    "fragment_id": f"frag_{page_num}_{t_idx}",
                    "page": page_num,
                    "bbox": bbox,
                    "headers": headers,
                    "rows": rows[1:],
                    "last_row": rows[-1] if len(rows) > 1 else None,
                    "page_height": page.height,
                })
    return fragments
```

For scanned/photo pages with no extractable text layer, fall back to a VLM call requesting the same fragment schema (headers, approximate bbox, rows) — only for those pages.

### 9.2 Three-signal scorer (`stitch.py`)

```python
from difflib import SequenceMatcher

def header_similarity(frag_a, frag_b):
    if not frag_a["headers"] or not frag_b["headers"]:
        return 0.0
    matches = sum(
        SequenceMatcher(None, str(a).lower().strip(), str(b).lower().strip()).ratio() > 0.7
        for a, b in zip(frag_a["headers"], frag_b["headers"])
    )
    return matches / max(len(frag_a["headers"]), len(frag_b["headers"]))

def column_position_similarity(frag_a, frag_b):
    width_a = frag_a["bbox"][2] - frag_a["bbox"][0]
    width_b = frag_b["bbox"][2] - frag_b["bbox"][0]
    if max(width_a, width_b) == 0:
        return 0.0
    width_diff = abs(width_a - width_b) / max(width_a, width_b)
    col_count_match = 1.0 if len(frag_a["headers"]) == len(frag_b["headers"]) else 0.3
    return col_count_match * (1 - width_diff)

def spatial_flow_score(frag_a, frag_b):
    a_ends_near_bottom = (frag_a["bbox"][3] / frag_a["page_height"]) > 0.75
    b_starts_near_top = (frag_b["bbox"][1] / frag_b["page_height"]) < 0.25
    consecutive_pages = (frag_b["page"] - frag_a["page"]) == 1
    if not consecutive_pages:
        return 0.0
    return (0.5 if a_ends_near_bottom else 0.0) + (0.5 if b_starts_near_top else 0.0)

def fusion_score(frag_a, frag_b, weights=(0.4, 0.4, 0.2)):
    h = header_similarity(frag_a, frag_b)
    c = column_position_similarity(frag_a, frag_b)
    s = spatial_flow_score(frag_a, frag_b)
    score = weights[0]*h + weights[1]*c + weights[2]*s
    return {"score": round(score, 3), "header": round(h, 2), "column": round(c, 2), "spatial": round(s, 2)}
```

### 9.3 Threading fragments into logical tables (`thread.py`)

```python
def thread_fragments(fragments, threshold_merge=0.75, threshold_review=0.5):
    fragments_sorted = sorted(fragments, key=lambda f: f["page"])
    threads = []
    used = set()

    for i, frag in enumerate(fragments_sorted):
        if frag["fragment_id"] in used:
            continue
        thread = [frag]
        used.add(frag["fragment_id"])
        current = frag

        for j in range(i + 1, len(fragments_sorted)):
            candidate = fragments_sorted[j]
            if candidate["fragment_id"] in used:
                continue
            result = fusion_score(current, candidate)
            if result["score"] >= threshold_merge:
                thread.append(candidate)
                used.add(candidate["fragment_id"])
                current = candidate
            elif result["score"] >= threshold_review:
                thread.append({**candidate, "_flag": "needs_review", "_score": result})
                used.add(candidate["fragment_id"])
                current = candidate
                break  # stop extending automatically past an uncertain edge
            else:
                break  # different table, stop searching forward from here

        threads.append(thread)
    return threads
```

### 9.4 Document-instance segmentation (near-duplicate disambiguation)

```python
def segment_instances(page_types, page_texts):
    # page_types e.g. ["1040","1040","1040","1040","1040","1040"] for 3 stacked 2-pagers
    instances = []
    i = 0
    while i < len(page_types):
        doc_type = page_types[i]
        expected_len = {"1040": 2, "w2": 1, "paystub": 1, "bank_statement": None}.get(doc_type, 1)
        if expected_len:
            span = page_texts[i:i + expected_len]
            instances.append({
                "doc_type": doc_type,
                "page_start": i + 1,
                "page_end": i + expected_len,
                "distinguishing_attr": extract_year_or_date(span)  # regex/pattern match, LLM fallback
            })
            i += expected_len
        else:
            i += 1
    return instances
```

This uses a fixed-length-per-type heuristic deliberately — simple, fast, and honest about being a heuristic rather than a learned model. State this plainly as a known limitation.

---

## 10. Evaluation — what we'll actually report (not aspirational targets)

Report real, measured numbers from your own test set, however small:

- Number of test documents / pages used
- Number of true page-boundary-spanning tables in the test set
- How many were correctly auto-merged
- How many were correctly flagged (true positives on uncertainty)
- How many were incorrectly merged (false merge — most important number to be honest about)
- LLM call count vs. total page count (your efficiency proof point)

Do not state F1/accuracy targets unless you've actually measured them on real data tonight.

---

## 11. "Won't catch" — say so openly

- Free-form text rows with no clear column structure
- Intentionally adversarial reformatting
- Handwritten tables on paper
- Novel document types outside our fixed-length heuristic (until a heuristic/LLM fallback is added)

---

## 12. Demo script (5 minutes)

1. **Show the problem** — open a real multi-page bank statement, point at a table that spans pages 12→13, explain why naive stitching breaks here (header drift / skew / repeated headers).
2. **Show the pipeline running** — page extraction -> fragment objects (show the actual JSON, prove it's not faked).
3. **Show the 3 signals computed** for the real boundary pair — header/column/spatial scores displayed.
4. **Show the auto-merge result** — before (2 separate tables) / after (1 continuous stitched table).
5. **Show a deliberately tricky case that gets correctly flagged**, not silently merged wrong — this is the strongest moment: prove the system fails loudly, not silently.
6. **Close on the efficiency story** — state how many pages touched an LLM vs. how many didn't.
7. **If handed a fresh/unseen document live** — run it through end to end, narrate what you see as it happens.

---

## 13. Deck outline (6–8 slides, per brief)

1. Title & pick — Problem B, one-line pitch
2. Approach — "stitching is coreference, not segmentation," in one sentence, then how
3. Architecture — one readable diagram (Section 4, simplified for slide)
4. Walkthrough — one real document, end to end through the system
5. How well it works — honest measured numbers (Section 10)
6. Limits & what's next — Sections 7 and 11, stated plainly
7. (Optional) Cost/efficiency breakdown — AI usage map (Section 6)
8. (Optional) Key screens — before/after visualization

---

## 14. One-sentence answer for any "why is this not just X" question

- *"Why not just use GPT-4o on every page?"* — Cost and latency at scale; 2,000 pages through a frontier model is not a deployable answer, and our deterministic-first approach handles the majority of pages for free.
- *"Why not a fixed template/regex stitcher?"* — That's exactly the failure mode we're fixing; it breaks silently on header drift and skew, which is the real-world case the brief describes.
- *"Why not the full 5-signal Bayesian fusion from the bigger vision?"* — Calibrating 5 probabilistic signals properly needs labeled data and iteration time we don't have tonight; 3 signals with a simple weighted score captures the same core insight honestly, within scope.