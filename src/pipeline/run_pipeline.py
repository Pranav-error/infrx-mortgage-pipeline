"""
run_pipeline.py — Full end-to-end orchestrator

Stages:
  1. extract  — pdfplumber (digital) + async Claude Haiku VLM (scanned)
  2. classify — 3-level cascade: table-header fingerprint → keyword → async Haiku
  3. segment  — PSS boundary detection → DocInstance list
  4. stitch   — O(n) PTT per DocInstance: group frags by doc → sort → thread_fragments
  5. render   — assemble final labels.json-compatible dict + demo summary

Usage:
  python3 src/pipeline/run_pipeline.py --pkg DataSet/pkg_000000 --api-key sk-ant-...
  python3 src/pipeline/run_pipeline.py --pdf path/to/file.pdf --api-key sk-ant-... --out out.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

# Make sure src/ is on the path regardless of CWD
_SRC = Path(__file__).resolve().parent.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from extraction.extract import extract_pdf
from classification.classify import classify_pages
from segmentation.segment import segment_documents
from segmentation.reorder import reorder_pages
from stitching.stitch import thread_fragments
from pipeline.cascade import CascadeController
from output.render import render_output


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_page_records(pages: list[dict], tables: list[dict]) -> list:
    """
    Adapt extraction page dicts to the objects classify_pages() expects.

    classify_pages() accesses:
      pr.page_index     — int
      pr.text           — str
      pr.fragment_headers — list of header lists (one per fragment on this page)
    """
    # Build a map: page_index → list of header lists from its fragments
    headers_by_page: dict[int, list] = {}
    for frag in tables:
        pi = frag.get("page_index", -1)
        if pi not in headers_by_page:
            headers_by_page[pi] = []
        headers_by_page[pi].append(frag.get("headers") or [])

    records = []
    for p in pages:
        pi = p["page_index"]
        rec = SimpleNamespace(
            page_index       = pi,
            text             = p.get("text") or "",
            fragment_headers = headers_by_page.get(pi, []),
        )
        records.append(rec)

    # classify_pages sorts internally, but deliver in page_index order for clarity
    records.sort(key=lambda r: r.page_index)
    return records


def _build_page_instance_map(doc_instances) -> dict[int, str]:
    """
    Return {page_index: doc_instance_id} from segmented DocInstances.
    Covers every page in each instance's [start_page, end_page] range.
    """
    mapping: dict[int, str] = {}
    for inst in doc_instances:
        for pi in range(inst.start_page, inst.end_page + 1):
            mapping[pi] = inst.doc_instance_id
    return mapping


def _group_fragments_by_instance(
    tables: list[dict],
    page_instance_map: dict[int, str],
) -> dict[str, list[dict]]:
    """
    Group table fragments by their doc_instance_id.
    Fragments on pages not covered by any instance go into a '_unassigned' bucket.
    """
    groups: dict[str, list[dict]] = {}
    for frag in tables:
        pi  = frag.get("page_index", -1)
        iid = page_instance_map.get(pi, "_unassigned")
        groups.setdefault(iid, []).append(frag)
    return groups


# ---------------------------------------------------------------------------
# Stage 2.5 — Reorder helper (for jumbled PDFs)
# ---------------------------------------------------------------------------

def _reorder_by_doc_type(
    pages: list[dict],
    classifications: list[dict],
    api_key: str | None = None,
) -> tuple[list[dict], dict[int, int]]:
    """
    Two-phase reorder for a completely jumbled mixed-type PDF:

    Phase 1 — Cluster by doc_type.
        Group all pages by their classified doc_type.
        Pages of the same type (e.g. all bank statement pages, all story pages)
        are pulled together into one group.

    Phase 2 — Order within each cluster.
        For each cluster, run reorder_pages() to recover the correct page
        sequence using chapter-structure (PATH A), structural signals (PATH B),
        or LLM (PATH C) — same algorithm as the standalone reorder test.

    After this stage, page_index values are reassigned sequentially so that
    segment_documents() sees a clean ordered stream where adjacent pages of
    the same doc_type truly belong together.

    Returns:
        reordered_pages  — list[dict] with updated page_index values
        orig_to_new      — dict mapping original_page_index → new page_index
                           (needed to write the Output.pdf in correct order)
    """
    classify_by_page = {c["page_index"]: c for c in classifications}

    # Group page dicts by doc_type (preserve a copy to avoid in-place mutation)
    groups: dict[str, list[dict]] = {}
    for p in pages:
        dt = classify_by_page.get(p["page_index"], {}).get("doc_type", "unknown")
        groups.setdefault(dt, []).append(dict(p))

    all_reordered: list[dict] = []
    orig_to_new: dict[int, int] = {}
    new_idx = 0

    for dt in sorted(groups.keys()):          # sorted for determinism
        group = groups[dt]
        n_group = len(group)

        if n_group == 1:
            orig_idx = group[0]["page_index"]
            group[0]["original_page_index"] = orig_idx
            group[0]["page_index"] = new_idx
            orig_to_new[orig_idx] = new_idx
            all_reordered.append(group[0])
            new_idx += 1
        else:
            print(f"[reorder] doc_type='{dt}'  {n_group} pages → running reorder...")
            ordered = reorder_pages(group, api_key=api_key)
            ordered_seq = sorted(ordered, key=lambda p: p["sorted_page_index"])
            for p in ordered_seq:
                orig_idx = p["page_index"]
                p["original_page_index"] = orig_idx
                p["page_index"] = new_idx
                orig_to_new[orig_idx] = new_idx
                all_reordered.append(p)
                new_idx += 1

    return all_reordered, orig_to_new


# ---------------------------------------------------------------------------
# Output PDF writer
# ---------------------------------------------------------------------------

def _write_output_pdf(
    original_pdf: str,
    ordered_pages: list[dict],
    out_path: str = "Output.pdf",
) -> None:
    """
    Write a new PDF with pages in the pipeline's recovered order.
    Uses original_page_index (set during reorder) to pick the right source page.
    Falls back to text-content matching when original_page_index is unavailable.
    """
    try:
        from pypdf import PdfReader, PdfWriter
        import pdfplumber
    except ImportError:
        print("[pipeline] pypdf/pdfplumber not installed — skipping Output.pdf")
        return

    print(f"\n[pipeline] Writing {out_path}...")
    reader = PdfReader(original_pdf)

    # Build fingerprint: first 200 chars of text → original page index
    orig_texts: dict[int, str] = {}
    with pdfplumber.open(original_pdf) as pdf:
        for i, pg in enumerate(pdf.pages):
            orig_texts[i] = (pg.extract_text() or "")[:200]

    def _resolve_orig(page_dict: dict) -> int:
        # Prefer stored original_page_index
        if "original_page_index" in page_dict:
            return int(page_dict["original_page_index"])
        # Fallback: match by text fingerprint
        target = (page_dict.get("text") or "")[:200].strip()
        best_i, best_len = 0, 0
        for oi, ot in orig_texts.items():
            common = len(set(target.split()) & set(ot.split()))
            if common > best_len:
                best_len, best_i = common, oi
        return best_i

    writer = PdfWriter()
    for p in ordered_pages:
        oi = _resolve_orig(p)
        writer.add_page(reader.pages[oi])

    with open(out_path, "wb") as f:
        writer.write(f)

    size_kb = Path(out_path).stat().st_size // 1024
    print(f"[pipeline] Saved → {out_path}  ({len(ordered_pages)} pages, {size_kb} KB)")


# ---------------------------------------------------------------------------
# Document summary — normal + comparison modes
# ---------------------------------------------------------------------------

def _print_document_summary(doc_instances):
    """Print a clean summary of detected documents (no ground truth needed)."""
    print("\n" + "=" * 70)
    print("  DOCUMENT SUMMARY — Pipeline Output")
    print("=" * 70)
    print(f"  {'#':<4} {'DOC TYPE':<30} {'PAGES':<15} {'COUNT':<6} ATTR")
    print("-" * 70)
    for i, inst in enumerate(doc_instances, 1):
        if inst.start_page == inst.end_page:
            page_range = f"p{inst.start_page}"
        else:
            page_range = f"p{inst.start_page} – p{inst.end_page}"
        attr = f"[{inst.distinguishing_attr}]" if inst.distinguishing_attr else ""
        print(f"  {i:<4} {inst.doc_type:<30} {page_range:<15} {inst.page_count:<6} {attr}")

    # Type summary
    type_counts: dict[str, int] = {}
    type_pages: dict[str, int] = {}
    for inst in doc_instances:
        type_counts[inst.doc_type] = type_counts.get(inst.doc_type, 0) + 1
        type_pages[inst.doc_type] = type_pages.get(inst.doc_type, 0) + inst.page_count

    print(f"\n  Total: {len(doc_instances)} documents, "
          f"{sum(type_pages.values())} pages, "
          f"{len(type_counts)} unique types")
    print("-" * 70)
    print(f"  {'TYPE':<30} {'INSTANCES':<12} {'PAGES':<8}")
    print("-" * 70)
    for dt in sorted(type_counts, key=lambda x: -type_pages[x]):
        print(f"  {dt:<30} {type_counts[dt]:<12} {type_pages[dt]:<8}")
    print("=" * 70)


def _print_comparison_summary(doc_instances, labels_path: str):
    """
    Compare pipeline output against ground truth labels.json.
    Prints side-by-side comparison + accuracy stats.
    """
    try:
        with open(labels_path) as f:
            labels = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return

    # Build ground truth: page_index → doc_type
    gt_pages: dict[int, str] = {}
    for p in labels.get("pages", []):
        gt_pages[p["page_index"]] = p["doc_type"]

    # Build ground truth document instances
    gt_docs = labels.get("documents", [])

    # Build pipeline: page_index → doc_type
    pred_pages: dict[int, str] = {}
    for inst in doc_instances:
        for pi in range(inst.start_page, inst.end_page + 1):
            pred_pages[pi] = inst.doc_type

    # Page-level accuracy
    total_pages = len(gt_pages)
    correct = sum(1 for pi in gt_pages if gt_pages[pi] == pred_pages.get(pi))

    print("\n" + "=" * 70)
    print("  COMPARISON — Pipeline vs Ground Truth")
    print("=" * 70)

    # Document-level comparison
    print(f"\n  {'GROUND TRUTH':<35} {'PIPELINE OUTPUT':<35}")
    print("-" * 70)

    gt_idx = 0
    pred_idx = 0
    gt_doc_list = sorted(gt_docs, key=lambda d: d.get("start_page", d.get("order_index", 0)))
    pred_doc_list = list(doc_instances)

    # Walk through pages and show documents side by side
    all_pages = sorted(set(list(gt_pages.keys()) + list(pred_pages.keys())))
    shown_gt = set()
    shown_pred = set()

    for inst_gt in gt_doc_list:
        sp = inst_gt.get("start_page", 0)
        ep = inst_gt.get("end_page", sp)
        gt_type = inst_gt.get("key", inst_gt.get("doc_type", "?"))
        gt_str = f"{gt_type} (p{sp}-p{ep})"

        # Find matching pipeline doc(s) for this page range
        matched_preds = []
        for inst_p in pred_doc_list:
            if inst_p.start_page <= ep and inst_p.end_page >= sp:
                if id(inst_p) not in shown_pred:
                    matched_preds.append(inst_p)

        if matched_preds:
            first = matched_preds[0]
            shown_pred.add(id(first))
            pred_str = f"{first.doc_type} (p{first.start_page}-p{first.end_page})"

            # Check match
            pages_in_gt = set(range(sp, ep + 1))
            pages_correct = sum(1 for pi in pages_in_gt if gt_pages.get(pi) == pred_pages.get(pi))
            if pages_correct == len(pages_in_gt):
                mark = "  OK"
            else:
                mark = f"  MISS ({pages_correct}/{len(pages_in_gt)})"

            print(f"  {gt_str:<35} {pred_str:<30} {mark}")

            # Show extra predictions if the pipeline split this doc
            for extra in matched_preds[1:]:
                shown_pred.add(id(extra))
                pred_str2 = f"{extra.doc_type} (p{extra.start_page}-p{extra.end_page})"
                print(f"  {'':<35} {pred_str2:<30}   ^split")
        else:
            print(f"  {gt_str:<35} {'--- MISSING ---':<30}   MISS")

    # Per-type accuracy
    type_stats: dict[str, dict] = {}
    for pi in gt_pages:
        gt = gt_pages[pi]
        pred = pred_pages.get(pi, "???")
        if gt not in type_stats:
            type_stats[gt] = {"correct": 0, "total": 0, "wrong_as": {}}
        type_stats[gt]["total"] += 1
        if pred == gt:
            type_stats[gt]["correct"] += 1
        else:
            type_stats[gt]["wrong_as"][pred] = type_stats[gt]["wrong_as"].get(pred, 0) + 1

    print(f"\n  PAGE-LEVEL ACCURACY: {correct}/{total_pages} ({100*correct/total_pages:.1f}%)")
    print("-" * 70)
    print(f"  {'DOC TYPE':<28} {'CORRECT':>8} {'TOTAL':>6} {'ACC':>6}  CONFUSED AS")
    print("-" * 70)
    for dt in sorted(type_stats, key=lambda x: -type_stats[x]["total"]):
        s = type_stats[dt]
        acc = 100 * s["correct"] / s["total"] if s["total"] else 0
        confused = ", ".join(f"{p}({n})" for p, n in sorted(s["wrong_as"].items(), key=lambda x: -x[1]))
        flag = "" if acc >= 80 else " <--"
        print(f"  {dt:<28} {s['correct']:>7}/{s['total']:<5} {acc:>5.0f}%  {confused}{flag}")
    print("=" * 70)


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    pdf_path: str,
    api_key: str | None = None,
    max_extract_workers: int = 10,
    max_classify_concurrent: int = 20,
    use_llm_stitcher: bool = True,
    jumbled: bool = False,
    output_pdf: str | None = None,
    labels_path: str | None = None,
) -> dict:
    """
    Run the full pipeline on a single PDF.
    Returns the final labels.json-compatible dict.

    Args:
        jumbled:    If True, run Stage 2.5 — group pages by doc_type and
                    reorder within each group before segmentation.
                    Use this when the input PDF has completely shuffled pages
                    (e.g. the hackathon test files).
        output_pdf: If set, write a reordered Output PDF to this path.
    """
    cascade = CascadeController()
    t_total = time.time()
    orig_to_new: dict[int, int] = {}   # populated by Stage 2.5 if jumbled=True

    # ------------------------------------------------------------------ #
    # Stage 1 — Extract                                                   #
    # ------------------------------------------------------------------ #
    print("\n" + "="*60)
    print("  STAGE 1 / 5 — EXTRACTION")
    print("="*60)
    extraction = extract_pdf(pdf_path, api_key, max_extract_workers)
    pages  = extraction["pages"]
    tables = extraction["tables"]
    print(f"  => {len(pages)} pages, {len(tables)} fragments extracted")

    # ------------------------------------------------------------------ #
    # Stage 2 — Classify                                                  #
    # ------------------------------------------------------------------ #
    print("\n" + "="*60)
    print("  STAGE 2 / 5 — CLASSIFICATION")
    print("="*60)
    page_records = _build_page_records(pages, tables)
    classifications = classify_pages(page_records, max_concurrent=max_classify_concurrent)

    # Record into cascade controller
    for cl in classifications:
        cascade.record_classification(
            page_index = cl["page_index"],
            method     = cl.get("method", "heuristic"),
            confidence = cl.get("confidence", 0.0),
            doc_type   = cl.get("doc_type", "unknown"),
        )

    # Patch doc_type / doc_type_label_id back into pages
    classify_by_page = {c["page_index"]: c for c in classifications}
    for p in pages:
        cl = classify_by_page.get(p["page_index"], {})
        p["doc_type"]          = cl.get("doc_type")
        p["doc_type_label_id"] = cl.get("doc_type_label_id")

    type_counts = {}
    for cl in classifications:
        dt = cl.get("doc_type", "unknown")
        type_counts[dt] = type_counts.get(dt, 0) + 1
    print(f"  => doc_type distribution: {json.dumps(type_counts, indent=None)}")

    cs = cascade.classification_stats()
    print(f"  => LLM escalations: {cs.get('llm_calls', 0)}/{cs.get('total_pages', 0)} "
          f"({cs.get('escalation_pct', '0%')})  "
          f"est. cost: ${cs.get('estimated_cost_usd', 0):.5f}")

    # ------------------------------------------------------------------ #
    # Stage 2.5 — Reorder (jumbled PDFs only)                            #
    # ------------------------------------------------------------------ #
    if jumbled:
        print("\n" + "="*60)
        print("  STAGE 2.5 — REORDER  (jumbled PDF — group + order by doc_type)")
        print("="*60)
        t0 = time.time()

        pages, orig_to_new = _reorder_by_doc_type(pages, classifications, api_key)

        # Remap classifications page_index to the new order
        new_classify: list[dict] = []
        for c in classifications:
            ni = orig_to_new.get(c["page_index"])
            if ni is not None:
                nc = dict(c)
                nc["page_index"] = ni
                new_classify.append(nc)
        classifications = sorted(new_classify, key=lambda c: c["page_index"])

        # Remap fragments page_index
        for frag in tables:
            old_pi = frag.get("page_index", -1)
            frag["page_index"] = orig_to_new.get(old_pi, old_pi)

        classify_by_page = {c["page_index"]: c for c in classifications}

        print(f"  => Reorder complete in {time.time()-t0:.1f}s")
        print(f"     Doc-type groups processed: "
              f"{len(set(c.get('doc_type','?') for c in classifications))}")

    # ------------------------------------------------------------------ #
    # Stage 3 — Segment                                                   #
    # ------------------------------------------------------------------ #
    print("\n" + "="*60)
    print("  STAGE 3 / 5 — SEGMENTATION")
    print("="*60)
    page_texts = [p.get("text") or "" for p in sorted(pages, key=lambda p: p["page_index"])]
    doc_instances = segment_documents(classifications, page_texts)

    print(f"  => {len(doc_instances)} document instances found")
    for inst in doc_instances:
        attr = f"  [{inst.distinguishing_attr}]" if inst.distinguishing_attr else ""
        print(f"     {inst.doc_instance_id:30s}  "
              f"pages {inst.start_page}–{inst.end_page}  "
              f"({inst.page_count} pages){attr}")

    # Build page → instance map
    page_instance_map = _build_page_instance_map(doc_instances)

    # Patch doc_instance_id / boundary fields onto pages
    for p in sorted(pages, key=lambda p: p["page_index"]):
        pi  = p["page_index"]
        iid = page_instance_map.get(pi)
        p["doc_instance_id"] = iid

        # Find the instance for this page
        inst_match = next((i for i in doc_instances if i.doc_instance_id == iid), None)
        if inst_match:
            page_in_doc = pi - inst_match.start_page + 1
            p["page_in_doc"]          = page_in_doc
            p["total_pages_in_doc"]   = inst_match.page_count
            p["is_first_page_of_doc"] = (pi == inst_match.start_page)
            p["is_last_page_of_doc"]  = (pi == inst_match.end_page)
            p["boundary"]             = "start" if pi == inst_match.start_page else (
                                        "end"   if pi == inst_match.end_page   else "middle")

    # Patch doc_instance_id onto fragments
    for frag in tables:
        pi  = frag.get("page_index", -1)
        iid = page_instance_map.get(pi)
        frag["doc_instance_id"] = iid
        frag["doctype"]         = classify_by_page.get(pi, {}).get("doc_type")

    # ------------------------------------------------------------------ #
    # Stage 4 — Stitch (O(n) — per-instance, adjacent only)              #
    # ------------------------------------------------------------------ #
    print("\n" + "="*60)
    print("  STAGE 4 / 5 — STITCHING  (PTT — per-instance, O(n))")
    print("="*60)

    frag_groups = _group_fragments_by_instance(tables, page_instance_map)
    all_threads: list[dict] = []
    table_id_counter = 0

    for iid, frags in frag_groups.items():
        if not frags:
            continue

        # Sort by page_index within instance (key correctness guarantee)
        frags_sorted = sorted(frags, key=lambda f: f.get("page_index", 0))

        threads = thread_fragments(frags_sorted, use_llm_arbiter=use_llm_stitcher)
        all_threads.extend(threads)

        # Record stitch decisions in cascade
        for thread in threads:
            for edge in thread.get("edges", []):
                decision = edge.get("decision", "reject")
                score    = edge.get("score", 0.0)
                frag_id  = edge.get("frag_a", "?")
                cascade.record_stitch(frag_id=frag_id, decision=decision, score=score)

    print(f"  => {len(all_threads)} logical table threads from {len(tables)} fragments")

    ss = cascade.stitch_stats()
    if ss:
        print(f"  => LLM arbitrations: {ss.get('llm_calls', 0)}/{ss.get('total_pairs', 0)} "
              f"({ss.get('escalation_pct', '0%')})  "
              f"est. cost: ${ss.get('estimated_cost_usd', 0):.5f}")

    # Assign table_id to each fragment based on its thread
    frag_to_table_id: dict[str, str] = {}
    for thread in all_threads:
        table_id = f"table_{table_id_counter:04d}"
        table_id_counter += 1
        for frag in thread.get("fragments", []):
            fid = frag.get("fragment_id")
            if fid:
                frag_to_table_id[fid] = table_id

    # Patch table_id + page_span onto fragments; collect table_ids per page
    page_table_ids: dict[int, list[str]] = {}
    for frag in tables:
        fid = frag.get("fragment_id")
        tid = frag_to_table_id.get(fid) if fid else None
        if tid:
            frag["table_id"] = tid
            # Update page_span across all fragments in same thread
            # (simplest: span is just this page unless part of a multi-page thread)
            pi = frag.get("page_index", -1)
            page_table_ids.setdefault(pi, [])
            if tid not in page_table_ids[pi]:
                page_table_ids[pi].append(tid)

    # Update table_ids on pages
    for p in pages:
        pi = p["page_index"]
        p["table_ids"] = page_table_ids.get(pi, [])

    # Update page_span on each fragment to reflect full thread span
    thread_span: dict[str, tuple[int, int]] = {}
    for thread in all_threads:
        frags_in_thread = thread.get("fragments", [])
        if not frags_in_thread:
            continue
        fids = [f.get("fragment_id") for f in frags_in_thread]
        tid  = frag_to_table_id.get(fids[0]) if fids[0] else None
        if tid:
            thread_span[tid] = (thread["page_start"], thread["page_end"])

    for frag in tables:
        fid = frag.get("fragment_id")
        tid = frag_to_table_id.get(fid) if fid else None
        if tid and tid in thread_span:
            frag["page_span"] = {
                "start_page": thread_span[tid][0],
                "end_page":   thread_span[tid][1],
            }
            frag["header_repeats_each_page"] = thread_span[tid][0] != thread_span[tid][1]

    # ------------------------------------------------------------------ #
    # Stage 5 — Render                                                    #
    # ------------------------------------------------------------------ #
    print("\n" + "="*60)
    print("  STAGE 5 / 5 — RENDER")
    print("="*60)

    final = render_output(extraction, doc_instances, all_threads, cascade)

    total_elapsed = time.time() - t_total
    print(f"\n  => Pipeline complete in {total_elapsed:.1f}s  "
          f"({len(pages)/total_elapsed:.1f} pages/s)")

    cascade.print_summary()

    # ------------------------------------------------------------------ #
    # Summary — always printed                                            #
    # ------------------------------------------------------------------ #
    _print_document_summary(doc_instances)

    # Comparison with ground truth (if labels.json exists)
    if labels_path and Path(labels_path).exists():
        _print_comparison_summary(doc_instances, labels_path)

    # ------------------------------------------------------------------ #
    # Output PDF (optional)                                               #
    # ------------------------------------------------------------------ #
    if output_pdf:
        # ordered_pages: pages in the order they should appear in the output PDF.
        # If jumbled=True, pages have already been reordered (page_index = new order).
        ordered_pages = sorted(pages, key=lambda p: p["page_index"])
        _write_output_pdf(pdf_path, ordered_pages, out_path=output_pdf)

    return final


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="InfrX full pipeline: extract → classify → segment → stitch → render"
    )
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--pkg", help="Path to a DataSet package dir (contains *.pdf)")
    group.add_argument("--pdf", help="Direct path to a PDF file")

    p.add_argument("--api-key",  default=os.getenv("ANTHROPIC_API_KEY"),
                   help="Anthropic API key (or set ANTHROPIC_API_KEY env var)")
    p.add_argument("--out",      default=None,
                   help="Output JSON path (default: <pkg_dir>/pipeline_output.json)")
    p.add_argument("--workers",  type=int, default=10,
                   help="Max concurrent VLM workers for extraction (default: 10)")
    p.add_argument("--no-stitch-llm", action="store_true",
                   help="Disable LLM arbiter in stitching (faster, slightly less accurate)")
    p.add_argument("--jumbled", action="store_true",
                   help="Input PDF has shuffled pages — run Stage 2.5 reorder before segment")
    p.add_argument("--output-pdf", default=None,
                   help="Write reordered Output.pdf to this path (e.g. Output.pdf)")
    return p.parse_args()


def main():
    args = _parse_args()

    # Resolve PDF path
    if args.pkg:
        pkg_dir = Path(args.pkg)
        pdfs = list(pkg_dir.glob("*.pdf"))
        if not pdfs:
            print(f"[ERROR] No PDF found in {pkg_dir}")
            sys.exit(1)
        pdf_path = str(pdfs[0])
        out_path = args.out or str(pkg_dir / "pipeline_output.json")
    else:
        pdf_path = args.pdf
        out_path = args.out or str(Path(pdf_path).with_suffix("_pipeline_output.json"))

    print(f"[pipeline] PDF:    {pdf_path}")
    print(f"[pipeline] Output: {out_path}")

    # Auto-detect labels.json for comparison mode
    labels_file = None
    if args.pkg:
        lp = pkg_dir / "labels.json"
        if lp.exists():
            labels_file = str(lp)

    result = run_pipeline(
        pdf_path             = pdf_path,
        api_key              = args.api_key,
        max_extract_workers  = args.workers,
        use_llm_stitcher     = not args.no_stitch_llm,
        jumbled              = args.jumbled,
        output_pdf           = args.output_pdf,
        labels_path          = labels_file,
    )

    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\n[pipeline] Saved → {out_path}  ({out_file.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
