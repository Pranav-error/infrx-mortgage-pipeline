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
# Core pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    pdf_path: str,
    api_key: str | None = None,
    max_extract_workers: int = 10,
    max_classify_concurrent: int = 20,
    use_llm_stitcher: bool = True,
) -> dict:
    """
    Run the full pipeline on a single PDF.
    Returns the final labels.json-compatible dict.
    """
    cascade = CascadeController()
    t_total = time.time()

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

    result = run_pipeline(
        pdf_path             = pdf_path,
        api_key              = args.api_key,
        max_extract_workers  = args.workers,
        use_llm_stitcher     = not args.no_stitch_llm,
    )

    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\n[pipeline] Saved → {out_path}  ({out_file.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
