"""
render.py — Stage 5: Final structured output assembler

Input:  extraction dict + doc_instances + stitched table threads + cascade stats
Output: final labels.json-compatible dict
        + console demo view (jumbled pages → clean document list + stitched tables)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from segmentation.segment import DocInstance
    from pipeline.cascade import CascadeController


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def render_output(
    extraction: dict,
    doc_instances: list,       # list[DocInstance]
    table_threads: list[dict],
    cascade=None,              # CascadeController | None
) -> dict:
    """
    Assemble all pipeline stages into a single labels.json-compatible dict.

    Args:
        extraction:    output of extract_pdf()
        doc_instances: output of segment_documents()
        table_threads: output of thread_fragments() (all instances combined)
        cascade:       CascadeController (for cost summary)

    Returns:
        dict mirroring the labels.json schema, fully populated.
    """
    pages  = extraction.get("pages", [])
    tables = extraction.get("tables", [])

    # ------------------------------------------------------------------ #
    # Build documents[] — one entry per DocInstance                       #
    # ------------------------------------------------------------------ #
    documents = []
    for inst in doc_instances:
        documents.append({
            "doc_instance_id":   inst.doc_instance_id,
            "doc_type":          inst.doc_type,
            "doc_type_label_id": inst.doc_type_label_id,
            "start_page":        inst.start_page,
            "end_page":          inst.end_page,
            "page_count":        inst.page_count,
            "instance_ordinal":  inst.instance_ordinal,
            "distinguishing_attr": inst.distinguishing_attr,
        })

    # ------------------------------------------------------------------ #
    # Build tables[] — stitched logical tables (thread-level view)        #
    # ------------------------------------------------------------------ #
    stitched_tables = []
    for thread in table_threads:
        frags = thread.get("fragments", [])
        if not frags:
            continue

        # Collect all cells across fragments in page order
        all_cells = []
        total_rows = 0
        for frag in sorted(frags, key=lambda f: f.get("page_index", 0)):
            # Renumber row_idx so they are globally monotone
            frag_cells = frag.get("cells", [])
            for cell in frag_cells:
                c = dict(cell)
                if not c.get("is_header"):
                    c["row_idx"] = total_rows + c.get("row_idx", 0)
                all_cells.append(c)
            total_rows += frag.get("row_count_logical", 0)

        # Representative header from first fragment
        first_frag = frags[0]
        stitched_tables.append({
            "table_id":                 first_frag.get("table_id"),
            "doc_instance_id":          first_frag.get("doc_instance_id"),
            "doctype":                  first_frag.get("doctype"),
            "page_span": {
                "start_page": thread["page_start"],
                "end_page":   thread["page_end"],
            },
            "header_repeats_each_page": thread["page_start"] != thread["page_end"],
            "columns":                  first_frag.get("columns", []),
            "row_count_logical":        total_rows,
            "cells":                    all_cells,
            # stitching metadata
            "n_fragments":              len(frags),
            "flagged":                  thread.get("flagged", False),
        })

    # ------------------------------------------------------------------ #
    # Build cascade summary                                                #
    # ------------------------------------------------------------------ #
    cascade_summary = None
    if cascade is not None:
        cascade_summary = cascade.to_dict()

    # ------------------------------------------------------------------ #
    # Assemble final dict                                                 #
    # ------------------------------------------------------------------ #
    final = {
        "schema_version": extraction.get("schema_version", "1.0.0"),
        "package_id":     extraction.get("package_id", ""),
        "total_pages":    extraction.get("total_pages", len(pages)),
        "coord_system":   extraction.get("coord_system"),
        "render_mode":    extraction.get("render_mode", "mixed"),
        "documents":      documents,
        "pages":          pages,
        "tables":         stitched_tables,
        "charts":         extraction.get("charts", []),
        "cascade":        cascade_summary,
    }

    # ------------------------------------------------------------------ #
    # Demo output                                                         #
    # ------------------------------------------------------------------ #
    _print_demo(final)

    return final


# ---------------------------------------------------------------------------
# Demo / before-after view
# ---------------------------------------------------------------------------

def _print_demo(result: dict) -> None:
    """
    Print a clean before/after demo view:
      BEFORE: raw page stream (index + render_mode)
      AFTER:  structured document list + stitched table summary
    Suitable for live hackathon demo.
    """
    pages     = result.get("pages", [])
    documents = result.get("documents", [])
    tables    = result.get("tables", [])

    print("\n" + "="*60)
    print("  DEMO — BEFORE (raw page stream)")
    print("="*60)
    for p in pages[:20]:   # show first 20 pages max
        mode = p.get("render_mode", "?")[0].upper()   # D / S
        doc  = p.get("doc_type") or "?"
        tid  = ",".join(p.get("table_ids") or []) or "-"
        print(f"  p{p['page_index']:03d} [{mode}]  {doc:25s}  tables: {tid}")
    if len(pages) > 20:
        print(f"  ... ({len(pages)-20} more pages)")

    print("\n" + "="*60)
    print("  DEMO — AFTER (structured documents)")
    print("="*60)
    for doc in documents:
        attr = f"  [{doc['distinguishing_attr']}]" if doc.get("distinguishing_attr") else ""
        print(f"  {doc['doc_instance_id']:35s}  "
              f"p{doc['start_page']}–p{doc['end_page']}  "
              f"({doc['page_count']} pages){attr}")

    print("\n" + "="*60)
    print(f"  STITCHED TABLES ({len(tables)} logical tables)")
    print("="*60)
    for tbl in tables[:30]:   # show first 30 tables
        span  = tbl.get("page_span", {})
        nfrags = tbl.get("n_fragments", 1)
        rows   = tbl.get("row_count_logical", 0)
        flag   = "  [FLAGGED]" if tbl.get("flagged") else ""
        multi  = f"  ({nfrags} fragments stitched)" if nfrags > 1 else ""
        print(f"  {tbl.get('table_id','?'):12s}  "
              f"p{span.get('start_page','?')}–p{span.get('end_page','?')}  "
              f"{rows:4d} rows  {tbl.get('doctype','?'):25s}{multi}{flag}")
    if len(tables) > 30:
        print(f"  ... ({len(tables)-30} more tables)")

    print()


# ---------------------------------------------------------------------------
# Standalone eval helper
# ---------------------------------------------------------------------------

def score_vs_labels(pipeline_output: dict, labels: dict) -> dict:
    """
    Compare pipeline output against ground-truth labels.json.
    Returns precision/recall/F1 for document boundary detection and table coverage.
    """
    # Document boundary F1
    pred_docs = pipeline_output.get("documents", [])
    true_docs = labels.get("documents", [])

    pred_spans = {(d["start_page"], d["end_page"]) for d in pred_docs}
    true_spans = {(d["start_page"], d["end_page"]) for d in true_docs}

    tp = len(pred_spans & true_spans)
    fp = len(pred_spans - true_spans)
    fn = len(true_spans - pred_spans)

    prec   = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1     = 2 * prec * recall / (prec + recall) if (prec + recall) else 0.0

    # Table coverage: fraction of true tables covered by at least one predicted table
    true_tables = labels.get("tables", [])
    pred_tables = pipeline_output.get("tables", [])
    pred_page_spans = set()
    for t in pred_tables:
        span = t.get("page_span", {})
        for pi in range(span.get("start_page", 0), span.get("end_page", 0) + 1):
            pred_page_spans.add(pi)

    covered = 0
    for tt in true_tables:
        sp = tt.get("page_span", {})
        for pi in range(sp.get("start_page", 0), sp.get("end_page", 0) + 1):
            if pi in pred_page_spans:
                covered += 1
                break

    table_coverage = covered / len(true_tables) if true_tables else 1.0

    return {
        "doc_boundary_precision": round(prec, 3),
        "doc_boundary_recall":    round(recall, 3),
        "doc_boundary_f1":        round(f1, 3),
        "doc_count_pred":         len(pred_docs),
        "doc_count_true":         len(true_docs),
        "table_coverage":         round(table_coverage, 3),
        "table_count_pred":       len(pred_tables),
        "table_count_true":       len(true_tables),
    }
