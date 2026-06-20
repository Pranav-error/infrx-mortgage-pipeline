"""
test_chapter_boundaries.py — Detect chapter boundaries in any story/textbook PDF

Uses our pipeline (classify + segment) WITHOUT page numbers.
Outputs a JSON listing where each chapter starts and ends.

Usage:
    python3 test_chapter_boundaries.py
    python3 test_chapter_boundaries.py --pdf "some_book.pdf"
    python3 test_chapter_boundaries.py --out chapters.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pdfplumber

sys.path.insert(0, str(Path(__file__).parent / "src"))
from classification.classify import classify_pages
from segmentation.segment import segment_documents

DEFAULT_PDF = "Grandma's Bag of Stories by Sudha Murthy.pdf"


def extract_pages(pdf_path: str, max_pages: int | None = None) -> list[dict]:
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        limit = min(len(pdf.pages), max_pages) if max_pages else len(pdf.pages)
        for i in range(limit):
            text = pdf.pages[i].extract_text() or ""
            pages.append({"page_index": i, "text": text})
    return pages


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf",       default=DEFAULT_PDF)
    ap.add_argument("--out",       default=None,  help="Output JSON path")
    ap.add_argument("--max-pages", type=int, default=None)
    ap.add_argument("--api-key",   default=None,  help="Anthropic key (enables LLM classify)")
    args = ap.parse_args()

    print("=" * 65)
    print("  CHAPTER BOUNDARY DETECTOR")
    print(f"  PDF: {args.pdf}")
    print("=" * 65)

    # ── Extract ──────────────────────────────────────────────────────────────
    print("\n[1] Extracting pages...")
    t0 = time.perf_counter()
    pages = extract_pages(args.pdf, args.max_pages)
    print(f"    {len(pages)} pages extracted in {(time.perf_counter()-t0)*1000:.0f} ms")

    # ── Classify ─────────────────────────────────────────────────────────────
    print("\n[2] Classifying pages...")
    t0 = time.perf_counter()
    records = [
        SimpleNamespace(page_index=p["page_index"], text=p["text"], fragment_headers=[])
        for p in pages
    ]
    classifications = classify_pages(records, max_concurrent=20, api_key=args.api_key)
    elapsed_ms = (time.perf_counter()-t0)*1000

    type_counts: dict[str, int] = {}
    for c in classifications:
        dt = c.get("doc_type", "unknown")
        type_counts[dt] = type_counts.get(dt, 0) + 1
    print(f"    Done in {elapsed_ms:.0f} ms")
    print(f"    doc_type distribution: {json.dumps(type_counts)}")

    # ── Segment ──────────────────────────────────────────────────────────────
    print("\n[3] Segmenting (finding chapter/document boundaries)...")
    t0 = time.perf_counter()
    page_texts = [p["text"] for p in sorted(pages, key=lambda p: p["page_index"])]
    instances = segment_documents(classifications, page_texts)
    elapsed_ms = (time.perf_counter()-t0)*1000
    print(f"    Done in {elapsed_ms:.0f} ms  — {len(instances)} segments found")

    # ── Display ──────────────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"  SEGMENTS FOUND ({len(instances)})")
    print(f"{'='*65}")
    print(f"  {'#':>3}  {'Doc type':22}  {'Pages':12}  {'Attr (chapter/date)'}")
    print("  " + "-"*60)
    for i, inst in enumerate(instances, 1):
        attr = f"  [{inst.distinguishing_attr}]" if inst.distinguishing_attr else ""
        page_range = f"p{inst.start_page+1}–p{inst.end_page+1}"
        print(f"  {i:>3}  {inst.doc_type:22}  {page_range:12}  "
              f"({inst.page_count} pages){attr}")

    # ── Post-process: merge narrative_chapter segments split by filler pages ─────
    # Filler pages (blank pages, illustrations) can appear mid-story without ending it.
    # Merge consecutive narrative_chapter segments where the second segment does NOT
    # start with a new chapter_header (i.e. it's a story body continuation).
    classify_by_pi = {c["page_index"]: c for c in classifications}
    page_text_by_pi = {p["page_index"]: p["text"] for p in pages}

    def _is_chapter_start(inst) -> bool:
        """True if this instance begins with an explicit chapter/story header."""
        cl = classify_by_pi.get(inst.start_page, {})
        return cl.get("method") == "chapter_header"

    from dataclasses import replace as dc_replace

    merged: list = []
    for inst in instances:
        # Find the last NON-filler segment in merged (look back past filler pages)
        last_story = next(
            (m for m in reversed(merged) if m.doc_type != "filler"),
            None,
        )
        if (
            last_story is not None
            and inst.doc_type == "narrative_chapter"
            and last_story.doc_type == "narrative_chapter"
            and not _is_chapter_start(inst)   # body continuation, not a new chapter title
        ):
            # Extend the previous narrative segment (in-place replacement)
            idx = next(i for i, m in enumerate(merged) if m is last_story)
            merged[idx] = dc_replace(
                last_story,
                end_page   = inst.end_page,
                page_count = last_story.page_count + inst.page_count,
            )
        else:
            merged.append(inst)

    print(f"\n  After merging filler-interrupted stories: {len(merged)} chapters/segments")

    def _story_title(inst) -> str | None:
        """Extract the story/chapter title for an instance."""
        if inst.distinguishing_attr:
            return inst.distinguishing_attr
        cl = classify_by_pi.get(inst.start_page, {})
        if cl.get("method") == "chapter_header":
            text = page_text_by_pi.get(inst.start_page, "")
            first_line = text.strip().split("\n")[0].strip()
            if first_line and len(first_line) <= 80:
                return first_line
        return None

    result = {
        "pdf": args.pdf,
        "total_pages": len(pages),
        "chapters": [
            {
                "segment_id":  inst.doc_instance_id,
                "doc_type":    inst.doc_type,
                "title":       _story_title(inst),
                "start_page":  inst.start_page + 1,
                "end_page":    inst.end_page   + 1,
                "page_count":  inst.page_count,
            }
            for inst in merged
            if inst.doc_type == "narrative_chapter"  # only story chapters in output
        ],
        "all_segments": [
            {
                "segment_id":  inst.doc_instance_id,
                "doc_type":    inst.doc_type,
                "title":       _story_title(inst),
                "start_page":  inst.start_page + 1,
                "end_page":    inst.end_page   + 1,
                "page_count":  inst.page_count,
            }
            for inst in merged
        ],
    }

    # Print clean story list
    chapters = result["chapters"]
    print(f"\n{'='*65}")
    print(f"  STORIES / CHAPTERS ({len(chapters)})")
    print(f"{'='*65}")
    print(f"  {'#':>3}  {'Title':40}  Pages")
    print("  " + "-"*60)
    for i, ch in enumerate(chapters, 1):
        title = (ch["title"] or "—")[:40]
        print(f"  {i:>3}  {title:40}  p{ch['start_page']}–p{ch['end_page']}  ({ch['page_count']} pages)")

    out_path = args.out or Path(args.pdf).stem + "_chapters.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n[4] Saved → {out_path}\n")


if __name__ == "__main__":
    main()
