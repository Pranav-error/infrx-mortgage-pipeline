"""
test_segment_dataset.py — Run classify+segment on a DataSet package and score vs labels.json

Uses the pre-extracted extraction_results.json (avoids re-running VLM).
Outputs predicted document boundaries and compares against ground truth.

Usage:
    python3 test_segment_dataset.py
    python3 test_segment_dataset.py --pkg DataSet/pkg_000001
    python3 test_segment_dataset.py --pkg DataSet/pkg_000000 --api-key sk-ant-...
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).parent / "src"))
from classification.classify import classify_pages
from segmentation.segment import segment_documents


def load_extraction(pkg_dir: Path) -> list[dict]:
    """Load pre-extracted pages from extraction_results.json."""
    path = pkg_dir / "extraction_results.json"
    if not path.exists():
        raise FileNotFoundError(f"No extraction_results.json in {pkg_dir}")
    with open(path) as f:
        data = json.load(f)
    pages = data.get("pages", [])
    tables = data.get("tables", [])
    print(f"    Loaded {len(pages)} pages, {len(tables)} table fragments")
    text_count = sum(1 for p in pages if len(p.get("text") or "") > 50)
    print(f"    Pages with extractable text: {text_count}/{len(pages)}")
    return pages, tables


def load_labels(pkg_dir: Path) -> list[dict]:
    """Load ground truth document boundaries from labels.json."""
    path = pkg_dir / "labels.json"
    if not path.exists():
        return []
    with open(path) as f:
        data = json.load(f)
    return data.get("documents", [])


def score_boundaries(predicted: list, ground_truth: list) -> dict:
    """
    Precision / Recall / F1 for document boundary detection.
    A predicted boundary matches a ground truth boundary if start_page matches exactly.
    """
    pred_starts = {inst.start_page for inst in predicted}
    true_starts = {d["start_page"] for d in ground_truth}

    tp = len(pred_starts & true_starts)
    fp = len(pred_starts - true_starts)
    fn = len(true_starts - pred_starts)

    prec   = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1     = 2 * prec * recall / (prec + recall) if (prec + recall) else 0.0

    return {
        "tp": tp, "fp": fp, "fn": fn,
        "precision": round(prec, 3),
        "recall":    round(recall, 3),
        "f1":        round(f1, 3),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkg",     default="DataSet/pkg_000000")
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--out",     default=None)
    args = ap.parse_args()

    pkg_dir = Path(args.pkg)
    print("=" * 65)
    print(f"  SEGMENT DETECTION — {pkg_dir.name}")
    print("=" * 65)

    # ── Load extraction results ───────────────────────────────────────────────
    print("\n[1] Loading extraction results...")
    pages, tables = load_extraction(pkg_dir)

    # ── Classify ─────────────────────────────────────────────────────────────
    print("\n[2] Classifying pages...")
    t0 = time.perf_counter()

    # Build fragment_headers map for table-header fingerprint
    headers_by_page: dict[int, list] = {}
    for frag in tables:
        pi = frag.get("page_index", -1)
        headers_by_page.setdefault(pi, []).append(frag.get("headers") or [])

    records = [
        SimpleNamespace(
            page_index       = p["page_index"],
            text             = p.get("text") or "",
            fragment_headers = headers_by_page.get(p["page_index"], []),
        )
        for p in sorted(pages, key=lambda p: p["page_index"])
    ]
    classifications = classify_pages(records, max_concurrent=20, api_key=args.api_key)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    type_counts: dict[str, int] = {}
    for c in classifications:
        dt = c.get("doc_type", "unknown")
        type_counts[dt] = type_counts.get(dt, 0) + 1
    print(f"    Done in {elapsed_ms:.0f} ms")
    print(f"    doc_type distribution: {json.dumps(type_counts)}")

    # ── Segment ──────────────────────────────────────────────────────────────
    print("\n[3] Segmenting...")
    t0 = time.perf_counter()
    sorted_pages = sorted(pages, key=lambda p: p["page_index"])
    page_texts = [p.get("text") or "" for p in sorted_pages]
    instances = segment_documents(classifications, page_texts)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    print(f"    Done in {elapsed_ms:.0f} ms  — {len(instances)} segments found")

    # ── Load ground truth ─────────────────────────────────────────────────────
    ground_truth = load_labels(pkg_dir)
    print(f"\n    Ground truth: {len(ground_truth)} document instances")

    # ── Display predicted segments ────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"  PREDICTED SEGMENTS ({len(instances)})")
    print(f"{'='*65}")
    print(f"  {'#':>3}  {'Doc type':25}  {'Pages':12}  {'Attr'}")
    print("  " + "-"*62)
    for i, inst in enumerate(instances, 1):
        attr = f"  [{inst.distinguishing_attr}]" if inst.distinguishing_attr else ""
        page_range = f"p{inst.start_page+1}–p{inst.end_page+1}"
        print(f"  {i:>3}  {inst.doc_type:25}  {page_range:12}  "
              f"({inst.page_count} pages){attr}")

    # ── Display ground truth ──────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"  GROUND TRUTH ({len(ground_truth)} instances)")
    print(f"{'='*65}")
    print(f"  {'#':>3}  {'Doc type':25}  {'Pages':12}")
    print("  " + "-"*55)
    for i, d in enumerate(ground_truth, 1):
        page_range = f"p{d['start_page']+1}–p{d['end_page']+1}"
        print(f"  {i:>3}  {d.get('key','?'):25}  {page_range:12}  ({d['page_count']} pages)")

    # ── Score ─────────────────────────────────────────────────────────────────
    if ground_truth:
        scores = score_boundaries(instances, ground_truth)
        print(f"\n{'='*65}")
        print(f"  BOUNDARY DETECTION SCORE")
        print(f"{'='*65}")
        print(f"  Predicted boundaries:  {len(instances)}")
        print(f"  Ground truth docs:     {len(ground_truth)}")
        print(f"  True positives:        {scores['tp']}")
        print(f"  False positives:       {scores['fp']}")
        print(f"  False negatives:       {scores['fn']}")
        print(f"\n  Precision:  {scores['precision']:.3f}  ({scores['precision']*100:.1f}%)")
        print(f"  Recall:     {scores['recall']:.3f}  ({scores['recall']*100:.1f}%)")
        print(f"  F1:         {scores['f1']:.3f}  ({scores['f1']*100:.1f}%)")

        missed = {d["start_page"] for d in ground_truth} - {i.start_page for i in instances}
        if missed:
            print(f"\n  Missed boundaries at pages: {sorted(p+1 for p in missed)}")

    # ── Save output ───────────────────────────────────────────────────────────
    out_path = args.out or str(pkg_dir / "pipeline_segments.json")
    result = {
        "package":   pkg_dir.name,
        "predicted": [
            {
                "doc_instance_id": inst.doc_instance_id,
                "doc_type":        inst.doc_type,
                "start_page":      inst.start_page + 1,
                "end_page":        inst.end_page   + 1,
                "page_count":      inst.page_count,
                "attr":            inst.distinguishing_attr,
            }
            for inst in instances
        ],
        "ground_truth": [
            {
                "doc_instance_id": d["doc_instance_id"],
                "doc_type":        d.get("key"),
                "start_page":      d["start_page"] + 1,
                "end_page":        d["end_page"]   + 1,
                "page_count":      d["page_count"],
            }
            for d in ground_truth
        ],
        "score": scores if ground_truth else None,
    }
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n[4] Saved → {out_path}\n")


if __name__ == "__main__":
    main()
