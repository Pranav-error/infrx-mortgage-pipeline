"""
eval_classify.py — Evaluate classify.py against labels.json ground truth.

Usage:
    python3 src/eval_classify.py
    python3 src/eval_classify.py --pkg "DataSet /pkg_000000"  # single package

Reads labels.json for ground truth doc_type per page.
Extracts text via pdfplumber (digital pages).
For scanned pages: renders as image → GPT-4o-mini vision OCR → classify.
Runs classify_pages and compares against ground truth.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import io
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import pdfplumber

sys.path.insert(0, str(Path(__file__).parent))
from classify import classify_pages, LABEL_ID

# OCR backend for scanned pages
_USE_OPENAI = bool(os.environ.get("OPENAI_API_KEY"))

_OCR_PROMPT = """Extract ALL readable text from this scanned document page.
Return ONLY the text content as a plain string — no JSON, no markdown, no explanation.
Preserve the layout as much as possible (headers, tables, paragraphs)."""

# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class FakePage:
    """Minimal stand-in for PageRecord so we can call classify_pages without extract.py."""
    page_index: int
    text: str


# ── OCR for scanned pages ────────────────────────────────────────────────────

async def _ocr_scanned_pages(pdf_path: str, scanned_indices: list[int]) -> dict[int, str]:
    """
    Render scanned pages as images and use GPT-4o-mini vision to extract text.
    Returns {page_index: extracted_text}.
    """
    if not scanned_indices or not _USE_OPENAI:
        return {}

    try:
        from pdf2image import convert_from_path
    except ImportError:
        print("[WARN] pdf2image not installed — pip install pdf2image && brew install poppler")
        return {}

    from openai import AsyncOpenAI

    # Render all scanned pages
    rendered: dict[int, str] = {}
    sorted_indices = sorted(scanned_indices)

    # Batch render
    runs: list[tuple[int, int]] = []
    run_start = sorted_indices[0]
    run_end = sorted_indices[0]
    for idx in sorted_indices[1:]:
        if idx == run_end + 1:
            run_end = idx
        else:
            runs.append((run_start + 1, run_end + 1))
            run_start = run_end = idx
    runs.append((run_start + 1, run_end + 1))

    for first, last in runs:
        imgs = convert_from_path(str(pdf_path), first_page=first, last_page=last, dpi=150)
        for offset, img in enumerate(imgs):
            page_index = first - 1 + offset
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            rendered[page_index] = base64.standard_b64encode(buf.getvalue()).decode()

    # OCR via GPT-4o-mini vision
    client = AsyncOpenAI()
    sem = asyncio.Semaphore(15)
    results: dict[int, str] = {}

    async def _ocr_one(pi: int, img_b64: str) -> tuple[int, str]:
        async with sem:
            try:
                resp = await client.chat.completions.create(
                    model="gpt-4o-mini",
                    max_tokens=2000,
                    messages=[{
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {
                                "url": f"data:image/png;base64,{img_b64}",
                            }},
                            {"type": "text", "text": _OCR_PROMPT},
                        ],
                    }],
                )
                return pi, resp.choices[0].message.content.strip()
            except Exception as e:
                print(f"[WARN] OCR page {pi}: {type(e).__name__}")
                return pi, ""

    tasks = [_ocr_one(pi, img_b64) for pi, img_b64 in rendered.items()]
    ocr_results = await asyncio.gather(*tasks)
    await client.close()

    for pi, text in ocr_results:
        if text:
            results[pi] = text

    return results


# ── Per-package eval ──────────────────────────────────────────────────────────

def eval_package(pkg_dir: str) -> dict:
    pkg_dir = Path(pkg_dir)
    pdf_path = pkg_dir / "package.pdf"
    labels_path = pkg_dir / "labels.json"

    with open(labels_path) as f:
        labels = json.load(f)

    # Ground truth: page_index -> doc_type
    page_gt: dict[int, str] = {p["page_index"]: p["doc_type"] for p in labels["pages"]}

    # Extract text for all pages
    pages: list[FakePage] = []
    scanned_indices: list[int] = []

    with pdfplumber.open(str(pdf_path)) as pdf:
        for page_index, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            if text.strip():
                pages.append(FakePage(page_index=page_index, text=text))
            else:
                scanned_indices.append(page_index)

    # OCR scanned pages if OpenAI key is available
    ocr_count = 0
    if scanned_indices and _USE_OPENAI:
        ocr_texts = asyncio.run(_ocr_scanned_pages(pdf_path, scanned_indices))
        for pi, text in ocr_texts.items():
            if text.strip():
                pages.append(FakePage(page_index=pi, text=text))
                ocr_count += 1
        # Re-sort by page_index
        pages.sort(key=lambda p: p.page_index)

    skipped_scanned = len(scanned_indices) - ocr_count

    if not pages:
        print(f"[{pkg_dir.name}] No pages found — skipping")
        return {}

    digital_count = len(pages) - ocr_count
    if ocr_count > 0:
        print(f"\n[{pkg_dir.name}] {digital_count} digital + {ocr_count} OCR'd + {skipped_scanned} skipped")
    else:
        print(f"\n[{pkg_dir.name}] {digital_count} digital pages + {len(scanned_indices)} scanned (skipped)")

    t0 = time.time()
    results = classify_pages(pages)
    elapsed = time.time() - t0

    # Score
    per_type: dict[str, dict] = {}
    for r in results:
        gt = page_gt.get(r["page_index"], "unknown")
        pred = r["doc_type"]
        method = r["method"]

        if gt not in per_type:
            per_type[gt] = {"correct": 0, "total": 0, "wrong_as": {}}
        per_type[gt]["total"] += 1
        if pred == gt:
            per_type[gt]["correct"] += 1
        else:
            wa = per_type[gt]["wrong_as"]
            wa[pred] = wa.get(pred, 0) + 1

    correct = sum(v["correct"] for v in per_type.values())
    total   = sum(v["total"]   for v in per_type.values())

    print(f"  Elapsed: {elapsed:.1f}s  |  Accuracy: {correct}/{total} ({100*correct/total:.1f}%)")

    return {
        "pkg": pkg_dir.name,
        "digital_pages": len(pages),
        "skipped_scanned": skipped_scanned,
        "elapsed": round(elapsed, 2),
        "correct": correct,
        "total": total,
        "per_type": per_type,
    }


# ── Aggregate report ──────────────────────────────────────────────────────────

def print_report(all_results: list[dict]):
    # Merge per_type across packages
    merged: dict[str, dict] = {}
    total_correct = 0
    total_pages   = 0

    for r in all_results:
        total_correct += r["correct"]
        total_pages   += r["total"]
        for doc_type, stats in r["per_type"].items():
            if doc_type not in merged:
                merged[doc_type] = {"correct": 0, "total": 0, "wrong_as": {}}
            merged[doc_type]["correct"] += stats["correct"]
            merged[doc_type]["total"]   += stats["total"]
            for pred, cnt in stats["wrong_as"].items():
                wa = merged[doc_type]["wrong_as"]
                wa[pred] = wa.get(pred, 0) + cnt

    print("\n" + "="*70)
    print(f"OVERALL ACCURACY: {total_correct}/{total_pages} ({100*total_correct/total_pages:.1f}%)")
    print("="*70)
    print(f"\n{'DOC TYPE':<28} {'CORRECT':>8} {'TOTAL':>7} {'ACC':>7}  CONFUSED AS")
    print("-"*70)

    for doc_type in sorted(merged, key=lambda x: -merged[x]["total"]):
        s = merged[doc_type]
        acc = 100 * s["correct"] / s["total"] if s["total"] else 0
        confused = ", ".join(
            f"{p}({n})" for p, n in sorted(s["wrong_as"].items(), key=lambda x: -x[1])
        )
        flag = "" if acc >= 80 else " <--"
        print(f"  {doc_type:<26} {s['correct']:>7}/{s['total']:<6} {acc:>5.0f}%  {confused}{flag}")

    print()
    bad = [(k, v) for k, v in merged.items()
           if v["total"] > 0 and 100*v["correct"]/v["total"] < 80]
    if bad:
        print(f"Types below 80% accuracy: {', '.join(k for k, _ in bad)}")
    else:
        print("All doc types >= 80% accuracy on digital pages.")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pkg", default=None, help="Single package dir to eval (default: all 6)")
    args = parser.parse_args()

    base = Path("DataSet ")

    if args.pkg:
        pkgs = [Path(args.pkg)]
    else:
        pkgs = sorted([p for p in base.iterdir() if p.is_dir() and p.name.startswith("pkg_")])

    all_results = []
    for pkg in pkgs:
        if not (pkg / "labels.json").exists():
            print(f"SKIP {pkg} — no labels.json")
            continue
        result = eval_package(str(pkg))
        if result:
            all_results.append(result)

    if all_results:
        print_report(all_results)


if __name__ == "__main__":
    main()
