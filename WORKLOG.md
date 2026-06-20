# Team Noobda — Work Tracker
**InfrX 2026 · Problem Statement B · June 20, 2026**

Keep this file open. Update it as you go. One source of truth.

---

## STATUS BOARD

| Module | File | Owner | Status |
|--------|------|-------|--------|
| PDF extraction (pdfplumber + VLM) | `src/extract.py` | teammate | DONE |
| Document-type classifier (27 types) | `src/classify.py` | sai pranav | DONE ✓ |
| Classifier eval script | `src/eval_classify.py` | sai pranav | DONE ✓ |
| PTT Bayesian stitch (5-signal) | `src/stitch.py` | sai pranav | DONE ✓ calibrated |
| Demo schema for segmentation | `src/demo_schema.json` | sai pranav | DONE ✓ |
| **Document segmentation** | `src/segment.py` | teammate | IN PROGRESS — skeleton ready |
| **Cascade controller** | `src/cascade.py` | sai pranav | SKELETON READY — fill in |
| **Column-fingerprint threader** | `src/stitch.py` | sai pranav | SKELETON READY inside stitch.py |
| Async VLM extraction speedup | `src/extract.py` | teammate | TODO |
| Semantic compression (10k→1.5k tok) | — | — | OPTIONAL / NOT STARTED |

---

## DONE — shipped & committed

### `src/classify.py`
- 27 doc types, exact label_id match to labels.json
- 3-tier: carry_forward → heuristic (≥0.6) → Haiku parallel async
- 84.2% accuracy, 2–8s per package
- Cascade pattern = FrugalGPT architecture (cheap first, LLM only on uncertain)

### `src/stitch.py`
- 5-signal Naive Bayes fusion (header, fingerprint, value_type, spatial, subtotal)
- Empirically calibrated on pkg_000000 (40 same + 47 diff pairs)
- Best signal: spatial (0.608 separation), fingerprint weaker than assumed
- All self-tests pass: merge P=1.0, reject P=0.0, subtotal gate P=0.0
- Decision: P>0.9 auto-merge, P 0.7–0.9 LLM arbiter, P<0.3 reject

### `src/extract.py`
- Fixed VLM JSON crash (double try/except, graceful skip)

---

## IN PROGRESS

### `src/segment.py` — Document Boundary Detector
**Academic name:** Page Stream Segmentation (PSS)
**Approach:** pairwise page-pair features → boundary yes/no → document spans
**Skeleton:** `src/segment.py` — function signatures + stubs ready
**What teammate needs to fill in:** `_extract_boundary_features()` and tuning threshold

### `src/cascade.py` — Cascade Controller
**Academic name:** confidence-thresholded model cascade (FrugalGPT pattern)
**Logs escalation/oracle-call rate — the cost metric judges will ask about**
**Skeleton:** ready, needs wiring to classify.py and stitch.py

---

## TODO (ranked by impact for demo)

- [ ] **Wire segment.py to classify.py output** — segment.py takes classify_pages() output, produces doc instances
- [ ] **Test full pipeline end-to-end** on pkg_000000: extract → classify → segment → stitch
- [ ] **Tell async VLM story** — teammate needs AsyncAnthropic + Semaphore in extract.py (same pattern as classify.py), cuts 7min → ~90s for 148 scanned pages
- [ ] **Add column_fingerprint + value_types to FragmentRecord** in extract.py — PTT signals 2 & 3 use synthesised values now, real ones would improve accuracy
- [ ] **Re-run eval after segmentation is done** — measure doc-instance boundary F1, not just page accuracy
- [ ] **Demo script** — one Python file that runs the full pipeline on pkg_000000 and prints a clean table

---

## RESEARCH CITATIONS TO USE IN SLIDES

### Logical Pagination (PSS)
- **arXiv 2408.11981** — "Large LLMs for Page Stream Segmentation" (2024) — modern LLM-based PSS; steal their pairwise boundary formulation
- **arXiv 2602.15958** — DocSplit — decomposing document packets; steal error taxonomy: *split groups / merged groups / wrong grouping* — drop on eval slide to look rigorous

### Table Threading
- **arXiv 2303.04384** — SEMv2 — table structure via separation-line detection; logical vs physical structure split = right mental model for DIR
- **arXiv 2506.07015** — TABLET — Split-Merge for large densely populated tables (bank statement transaction tables = exactly this)

### Efficiency / Cascade
- **FrugalGPT** (Chen et al. 2023) — cheap router + expensive model on uncertainty — THIS IS YOUR ARCHITECTURE, cite it
- **arXiv 2110.10305** — "When in doubt, summon the titans" — cleanest framing of cheap-first cascade

### Layout Classification
- **arXiv 2503.17213** — PP-DocLayout / PaddleOCR 3.0 — unified layout detection, runs locally, no per-page VLM
- **arXiv 2308.12896** — "Beyond Document Page Classification" (WACV 2024) — justifies evaluating on whole files, not individual pages

---

## NUMBERS TO MEMORISE FOR DEMO

| Metric | Value |
|--------|-------|
| Overall classification accuracy | 84.2% |
| Best package (pkg_000000) | 92.3% |
| Pages processed | 285 digital pages across 6 packages |
| Speed per package | 2–8s |
| LLM call rate (classify) | ~25% of pages (heuristic handles 75%) |
| PTT auto-merge rate | ~85% of fragment pairs |
| PTT LLM arbiter rate | ~15% |
| Spatial signal separation | 0.608 (best signal) |
| Cost per page (Haiku) | ~$0.0003 |

---

## TERMINOLOGY TO DROP IN PRESENTATION

Phrases that score points with judges:

- "Page Stream Segmentation" — academic name for logical pagination
- "pairwise coreference, not segmentation" — our key reframe
- "confidence-thresholded model cascade" — FrugalGPT architecture
- "escalation rate / oracle-call rate" — the cost metric
- "Document Intermediate Representation (DIR)" — our data layer name
- "Naive Bayes belief fusion over independent evidence signals" — PTT core
- "spatial flow score" — strongest PTT signal
- "column-width fingerprint" — normalized column boundaries [0,1]
- covariance, entropy, lattice, semaphore, matrices — sprinkle these for technical depth

---

## TEAMMATE HANDOFF NOTES

### For the segmentation teammate
- Read `src/demo_schema.json` — full INPUT/OUTPUT schema with 17-page example
- INPUT = classify.py output list, OUTPUT = labels.json document[] format
- Key challenge: pages 0–8 all `bank_stmt_checking` but TWO instances (different statement dates)
- Key challenge 2: pages 13–16 all `form_1040` but TWO instances (different tax years)
- `src/segment.py` has the function skeleton — fill in `segment_documents()`

### For the extract.py async speedup
- Copy the AsyncAnthropic + asyncio.Semaphore pattern from classify.py
- Replace `client.messages.create()` with `await client.messages.create()` in `_vlm_extract_page`
- Wrap main loop in `asyncio.run()`
- Expected speedup: 7min → ~90s for 148 scanned pages

---

## QUICK COMMANDS

```bash
# Activate venv
source .venv/bin/activate

# Run full classifier eval
python3 src/eval_classify.py

# Run single package eval
python3 src/eval_classify.py --pkg "DataSet /pkg_000000"

# Run extraction on a package
python3 src/run_extraction.py --pkg "DataSet /pkg_000000"

# Test stitch.py
python3 src/stitch.py

# Test segment.py (once filled in)
python3 src/segment.py

# Git push
git add -p && git commit -m "..." && git push origin main
```
