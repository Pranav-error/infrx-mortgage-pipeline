"""
test_reorder_pdf.py — Test content-based page reordering on any PDF

Takes the Computer Networks PDF (or any PDF), SHUFFLES the pages,
then runs the reorder algorithm purely on content — NO page numbers used.

Uses the real page numbers ONLY to measure accuracy at the end.

Usage:
    python3 test_reorder_pdf.py
    python3 test_reorder_pdf.py --pdf some_other.pdf
    python3 test_reorder_pdf.py --api-key sk-ant-...   # enable LLM boost
    python3 test_reorder_pdf.py --no-shuffle           # test on already-jumbled PDF
"""

from __future__ import annotations

import argparse
import random
import re
import sys
import time
from pathlib import Path

import pdfplumber

sys.path.insert(0, str(Path(__file__).parent / "src"))
from segmentation.reorder import reorder_pages

PDF_PATH = "Computer_Networks_Complete_Notes-Shivansh-Vasu (1).pdf"

# Ground truth: the footer on this PDF has "| Page X" — used ONLY for eval
_GT_RE = re.compile(r"\|\s*Page\s+(\d+)", re.IGNORECASE)


def get_true_page_num(text: str) -> int | None:
    """Extract the true logical page number from the footer (ground truth only)."""
    m = _GT_RE.search(text)
    return int(m.group(1)) if m else None


def extract_pages(pdf_path: str, max_pages: int = None) -> list[dict]:
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        total = len(pdf.pages)
        limit = min(total, max_pages) if max_pages else total
        for i in range(limit):
            text = pdf.pages[i].extract_text() or ""
            pages.append({"page_index": i, "text": text})
    return pages


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf",       default=PDF_PATH)
    ap.add_argument("--api-key",   default=None)
    ap.add_argument("--no-shuffle",action="store_true", help="Don't shuffle (PDF already jumbled)")
    ap.add_argument("--max-pages", type=int, default=None)
    ap.add_argument("--seed",      type=int, default=42)
    args = ap.parse_args()

    print("=" * 65)
    print("  CONTENT-BASED PAGE REORDER TEST")
    print(f"  PDF: {args.pdf}")
    print("=" * 65)

    # ── Extract ───────────────────────────────────────────────────────────────
    print("\n[1] Extracting pages...")
    t0 = time.perf_counter()
    pages = extract_pages(args.pdf, args.max_pages)
    print(f"    {len(pages)} pages in {(time.perf_counter()-t0)*1000:.0f} ms")

    # Record ground truth page numbers (for eval only — algorithm never sees these)
    ground_truth = {}   # page_index → true logical page num
    for p in pages:
        tn = get_true_page_num(p["text"])
        if tn:
            ground_truth[p["page_index"]] = tn

    # ── Shuffle ───────────────────────────────────────────────────────────────
    if not args.no_shuffle:
        random.seed(args.seed)
        shuffled = pages[:]
        random.shuffle(shuffled)
        for new_idx, p in enumerate(shuffled):
            p["page_index"] = new_idx
        pages = shuffled
        print(f"\n[2] Pages SHUFFLED (seed={args.seed})")
    else:
        print(f"\n[2] Using PDF order as-is (--no-shuffle)")

    # Show before state
    print(f"\n    BEFORE — shuffled order:")
    print(f"    {'PDF pos':>7}  {'True pg#':>9}  Content preview")
    print("    " + "-"*65)
    for p in pages[:20]:
        orig_idx = next((k for k, v in ground_truth.items()
                         if v == ground_truth.get(
                             next((pp["page_index"] for pp in pages
                                   if pp is p), -1), -999)), None)
        tn = get_true_page_num(p["text"])
        snippet = p["text"].replace("\n", " ").strip()[:45]
        print(f"    p{p['page_index']:>02d}      "
              f"{'pg '+str(tn) if tn else '?':>8}    {snippet}")
    if len(pages) > 20:
        print(f"    ... ({len(pages)-20} more pages)")

    # ── Reorder — NO page numbers, purely content ─────────────────────────────
    print(f"\n[3] REORDERING (content only, no page numbers)...")
    t0 = time.perf_counter()
    ordered = reorder_pages(pages, api_key=args.api_key)
    t_reorder = time.perf_counter() - t0

    # Sort by sorted_page_index to get final sequence
    ordered_seq = sorted(ordered, key=lambda p: p["sorted_page_index"])

    print(f"\n    AFTER — reordered sequence:")
    print(f"    {'Rank':>5}  {'True pg#':>9}  {'Method':>16}  Content preview")
    print("    " + "-"*70)
    for p in ordered_seq:
        tn = get_true_page_num(p["text"])
        snippet = p["text"].replace("\n", " ").strip()[:45]
        method = p.get("reorder_method", "?")
        print(f"    {p['sorted_page_index']:>4}   "
              f"{'pg '+str(tn) if tn else '?':>8}   "
              f"{method:>15}    {snippet}")

    # ── Accuracy ──────────────────────────────────────────────────────────────
    print(f"\n[4] ACCURACY  (ground truth = footer page numbers)")
    print("    " + "-"*45)

    true_seq = [get_true_page_num(p["text"]) for p in ordered_seq]
    valid    = [(i, t) for i, t in enumerate(true_seq) if t is not None]

    # Kendall's Tau — fraction of pairs in correct relative order
    correct_pairs = 0
    total_pairs   = 0
    for x in range(len(valid)):
        for y in range(x+1, len(valid)):
            rank_x, pg_x = valid[x]
            rank_y, pg_y = valid[y]
            total_pairs += 1
            if pg_x < pg_y:   # correct relative order
                correct_pairs += 1

    tau = correct_pairs / total_pairs if total_pairs else 0.0

    # Perfect consecutive pairs
    consec_correct = sum(
        1 for i in range(len(true_seq)-1)
        if true_seq[i] is not None and true_seq[i+1] is not None
        and true_seq[i+1] == true_seq[i] + 1
    )
    consec_total = sum(
        1 for i in range(len(true_seq)-1)
        if true_seq[i] is not None and true_seq[i+1] is not None
    )

    print(f"\n    Pages tested:              {len(pages)}")
    print(f"    Pages with ground truth:   {len(valid)}")
    print(f"    Kendall's Tau (order):     {tau:.3f}  ({tau*100:.1f}% pairs in correct order)")
    print(f"    Consecutive pairs correct: {consec_correct}/{consec_total}"
          f"  ({consec_correct/consec_total*100:.1f}%)" if consec_total else "    N/A")
    print(f"\n    True page sequence output: {true_seq}")
    print(f"\n    Wall time: {t_reorder*1000:.1f} ms  ({len(pages)} pages)")

    quality = "PERFECT" if tau > 0.99 else \
              "EXCELLENT" if tau > 0.90 else \
              "GOOD" if tau > 0.75 else \
              "PARTIAL" if tau > 0.50 else "POOR"
    print(f"\n    Result: {quality}  (Kendall Tau = {tau:.3f})")

    # ── Write Output.pdf ──────────────────────────────────────────────────────
    _write_output_pdf(args.pdf, ordered_seq, out_path="Output.pdf")
    print()


def _write_output_pdf(original_pdf: str, ordered_pages: list[dict], out_path: str = "Output.pdf"):
    """
    Write a new PDF with pages in the reordered sequence.
    Uses the original page content (not re-rendered) — pixel-perfect copy.
    """
    from pypdf import PdfReader, PdfWriter

    reader = PdfReader(original_pdf)
    writer = PdfWriter()

    # Map original page_index → original PDF page object
    # ordered_pages each have their original page_index stored before shuffling
    # We need to recover original_page_index from the page dict.
    # After shuffling, page_index was reassigned — we stored _orig_page_index if shuffle happened.
    # Fall back: use the order in ordered_pages directly matched to reader pages by position.

    # Build lookup: true_page_num (from footer) → reader page index
    # Since we can't always trust page_index after shuffle, match by text content
    import pdfplumber

    print(f"\n[5] Writing Output.pdf...")

    # Extract text from original PDF pages for matching
    orig_texts = {}
    with pdfplumber.open(original_pdf) as pdf:
        for i, page in enumerate(pdf.pages):
            text = (page.extract_text() or "")[:200]   # first 200 chars as fingerprint
            orig_texts[i] = text

    def _find_original_index(page_dict: dict) -> int:
        """Match a reordered page back to its original PDF position."""
        target = (page_dict.get("text") or "")[:200]
        for orig_i, orig_t in orig_texts.items():
            if orig_t.strip() == target.strip():
                return orig_i
        # Fuzzy fallback — find closest match
        best_i, best_len = 0, 0
        for orig_i, orig_t in orig_texts.items():
            common = len(set(target.split()) & set(orig_t.split()))
            if common > best_len:
                best_len, best_i = common, orig_i
        return best_i

    added = 0
    for p in ordered_pages:
        orig_i = _find_original_index(p)
        writer.add_page(reader.pages[orig_i])
        added += 1

    with open(out_path, "wb") as f:
        writer.write(f)

    size_kb = Path(out_path).stat().st_size // 1024
    print(f"    Saved → {out_path}  ({added} pages, {size_kb} KB)")


if __name__ == "__main__":
    main()
