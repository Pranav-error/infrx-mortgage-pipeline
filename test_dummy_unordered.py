"""
test_dummy_unordered.py — Simulate a jumbled multi-document PDF

Creates a fake extraction dict with 16 pages in WRONG order
(bank stmts split by W2/paystub, loan docs mixed in),
then runs classify → segment → stitch → render and measures timing.

No API key needed — all pages have strong keyword text that hits
the heuristic classifier (no LLM calls fired).

Usage:
    python3 test_dummy_unordered.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).parent / "src"))

from classification.classify import classify_pages
from segmentation.segment import segment_documents
from stitching.stitch import thread_fragments
from pipeline.cascade import CascadeController
from output.render import render_output

# ── 1. Dummy page text templates ──────────────────────────────────────────────
# Each string hits ≥ 60% of the keyword patterns for its doc_type,
# so the heuristic fires immediately — zero LLM calls.

def _bank_stmt_text(month: str, account: str, page_num: int) -> str:
    return f"""
    Chase Bank — Bank Statement
    Checking Account Statement — {month}
    Account Number: ****{account}
    Statement Period: {month} 1 – {month} 31, 2024

    Beginning Balance: $12,450.00
    Total Deposits: $4,500.00
    Total Withdrawals: $3,200.00
    Ending Balance: $13,750.00

    Daily Account Activity — Page {page_num}
    Date        Description                 Withdrawals    Deposits    Balance
    01/{month[:2]} PAYROLL DIRECT DEPOSIT                   $4,500.00  $16,950.00
    03/{month[:2]} AMAZON PURCHASE              $129.99                $16,820.01
    07/{month[:2]} UTILITY PAYMENT              $210.00                $16,610.01
    """

def _w2_text(year: str, employer: str) -> str:
    return f"""
    W-2 Wage and Tax Statement {year}
    Employer Identification Number: 12-3456789
    Employee's Social Security Number: ***-**-6789

    Box 1 Wages, tips, other compensation: $85,000.00
    Box 2 Federal income tax withheld:      $12,500.00
    Box 3 Social security wages:            $85,000.00

    Employer: {employer}
    Employee: JOHN BORROWER
    """

def _paystub_text(period_end: str) -> str:
    return f"""
    EARNINGS STATEMENT — Pay Stub
    Employee ID: EMP-004892
    Pay Period: 11/01/2024 – {period_end}
    Check Date: {period_end}

    Gross Pay:       $3,541.67
    Federal Tax:      $530.25
    State Tax:        $177.08
    Net Pay:         $2,834.34

    Year-to-Date Summary
    YTD Gross Pay:  $85,000.00
    YTD Net Pay:    $68,024.16
    """

def _urla_text(page_num: int) -> str:
    return f"""
    Uniform Residential Loan Application
    Form 1003 — Page {page_num}
    Application Date: 12/01/2024
    Lender Loan No.: 6061178222

    SECTION 1: BORROWER INFORMATION
    Borrower: John Borrower
    Co-Borrower: Jane Borrower
    Property and Loan Information

    To be completed by the borrower
    Loan Purpose: Purchase
    Loan Amount: $425,000
    """

def _loan_estimate_text(page_num: int) -> str:
    return f"""
    LOAN ESTIMATE — Page {page_num}
    Save this Loan Estimate to compare with your Closing Disclosure.

    Loan Terms
    Projected Payments — Principal, Interest & Mortgage Insurance
    Before You Close, review all costs carefully.
    Comparisons in 5 years: $85,000 in principal paid

    Estimated Total Monthly Payment: $2,250
    Use these measures to compare loan offers.
    Other Considerations: Appraisal required.
    """

def _closing_disclosure_text(page_num: int) -> str:
    return f"""
    CLOSING DISCLOSURE — Page {page_num}
    Closing Cost Details
    Final Loan Terms

    Cash to Close: $24,350.00
    Loan Amount: $425,000.00
    Interest Rate: 6.875%

    Closing Costs Breakdown
    Origination Charges: $1,200
    Services Borrower Did Not Shop For: $850
    Total Closing Costs: $8,750
    """


# ── 2. Define TRUE logical document order ─────────────────────────────────────
# Each entry: (page_index_in_true_order, doc_type, text, has_table)

TRUE_PAGES = [
    # URLA — 3 pages
    (0,  "urla_1003",          _urla_text(1),                          False),
    (1,  "urla_1003",          _urla_text(2),                          False),
    (2,  "urla_1003",          _urla_text(3),                          False),
    # W2 × 2 — employer A and B
    (3,  "w2",                 _w2_text("2023", "TechCorp Inc"),        False),
    (4,  "w2",                 _w2_text("2023", "Consulting LLC"),      False),
    # Paystub × 2
    (5,  "paystub",            _paystub_text("11/15/2024"),             False),
    (6,  "paystub",            _paystub_text("11/30/2024"),             False),
    # Bank Stmt January — 4 pages with transaction tables
    (7,  "bank_stmt_checking", _bank_stmt_text("January",  "1234", 1), True),
    (8,  "bank_stmt_checking", _bank_stmt_text("January",  "1234", 2), True),
    (9,  "bank_stmt_checking", _bank_stmt_text("January",  "1234", 3), True),
    (10, "bank_stmt_checking", _bank_stmt_text("January",  "1234", 4), True),
    # Bank Stmt February — 3 pages
    (11, "bank_stmt_checking", _bank_stmt_text("February", "1234", 1), True),
    (12, "bank_stmt_checking", _bank_stmt_text("February", "1234", 2), True),
    (13, "bank_stmt_checking", _bank_stmt_text("February", "1234", 3), True),
    # Loan Estimate — 3 pages
    (14, "loan_estimate",      _loan_estimate_text(1),                  False),
    (15, "loan_estimate",      _loan_estimate_text(2),                  False),
    (16, "loan_estimate",      _loan_estimate_text(3),                  False),
    # Closing Disclosure — 3 pages
    (17, "closing_disclosure", _closing_disclosure_text(1),             False),
    (18, "closing_disclosure", _closing_disclosure_text(2),             False),
    (19, "closing_disclosure", _closing_disclosure_text(3),             False),
]

# ── 3. JUMBLED order — simulate a real out-of-order mortgage PDF ──────────────
# Take the true page indices and shuffle them.
# The page_index values are PRESERVED (they reflect position in the jumbled PDF),
# but the doc_types they belong to are scrambled across the file.

JUMBLED_ORDER = [
    # bank stmt Jan p2 appears first
    8,
    # W2 #1 stuck right after bank stmt
    3,
    # loan estimate p1 randomly placed early
    14,
    # URLA p2 then p1 then p3 out of order
    1, 0, 2,
    # paystub #2 before paystub #1
    6, 5,
    # W2 #2
    4,
    # bank stmt Feb p3 then p1 then p2 — all out of order
    13, 11, 12,
    # bank stmt Jan p4, p1, p3 — fragmented
    10, 7, 9,
    # closing disclosure p3 then p2 then p1
    19, 17, 18,
    # loan estimate p3 then p2
    16, 15,
]

# ── 4. Build dummy extraction dict ────────────────────────────────────────────

def _make_fragment(page_index: int, t_idx: int = 0) -> dict:
    """Fake table fragment for bank statement pages."""
    return {
        "table_id":                 None,
        "doc_instance_id":          None,
        "doctype":                  None,
        "page_span":                {"start_page": page_index, "end_page": page_index},
        "header_repeats_each_page": None,
        "columns":                  [{"col_idx": i} for i in range(5)],
        "row_count_logical":        3,
        "cells": [
            {"page_index": page_index, "row_idx": -1, "col_idx": i,
             "is_header": True,  "text": h, "bbox": [72+i*90, 100, 72+(i+1)*90, 120],
             "bbox_px": [150+i*187, 208, 337+i*187, 250]}
            for i, h in enumerate(["Date", "Description", "Withdrawals", "Deposits", "Balance"])
        ] + [
            {"page_index": page_index, "row_idx": r, "col_idx": c,
             "is_header": False, "text": v,
             "bbox": [72+c*90, 120+r*20, 162+c*90, 140+r*20],
             "bbox_px": [150+c*187, 250+r*42, 337+c*187, 292+r*42]}
            for r, row in enumerate([
                ["01/15", "PAYROLL DEPOSIT",  "",         "4500.00", "16950.00"],
                ["01/17", "AMAZON PURCHASE",  "129.99",   "",        "16820.01"],
                ["01/22", "UTILITY PAYMENT",  "210.00",   "",        "16610.01"],
            ])
            for c, v in enumerate(row)
        ],
        "column_fingerprint": [0.0, 0.2, 0.5, 0.65, 0.82],
        "value_types":        ["date", "text", "currency", "currency", "currency"],
        "fragment_id":        f"frag_{page_index}_{t_idx}",
        "page_index":         page_index,
        "bbox":               [72, 100, 540, 700],
        "headers":            ["Date", "Description", "Withdrawals", "Deposits", "Balance"],
        "last_row_text":      ["01/22", "UTILITY PAYMENT", "210.00", "", "16610.01"],
        "page_height":        792.0,
        "page_width":         612.0,
        "source":             "dummy",
    }


def build_dummy_extraction() -> dict:
    """Build a fake extraction dict in JUMBLED page order."""
    # Map true_page_index → (doc_type, text, has_table)
    true_map = {pi: (dt, txt, ht) for pi, dt, txt, ht in TRUE_PAGES}

    pages_out = []
    tables_out = []

    for jumbled_pos, true_pi in enumerate(JUMBLED_ORDER):
        _, text, has_table = true_map[true_pi]

        # The page_index in the jumbled PDF is its position (0-based)
        # but we KEEP the true_pi so we can compare later — or we can
        # assign new indices. For realism, assign jumbled_pos as page_index.
        page_index = jumbled_pos

        frags = []
        if has_table:
            frag = _make_fragment(page_index)
            frags.append(frag)
            tables_out.append(frag)

        pages_out.append({
            "page_index":           page_index,
            "doc_type":             None,
            "doc_type_label_id":    None,
            "doc_instance_id":      None,
            "section":              None,
            "is_first_page_of_doc": None,
            "is_last_page_of_doc":  None,
            "page_in_doc":          None,
            "total_pages_in_doc":   None,
            "boundary":             None,
            "width":                612.0,
            "height":               792.0,
            "has_table":            has_table,
            "table_ids":            [],
            "has_chart":            False,
            "chart_ids":            [],
            "render_mode":          "digital",
            "scan_transform":       [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            "rotation":             0,
            "scan_image_size_px":   [1275, 1650],
            "text":                 text,
            "fragment_ids":         [f["fragment_id"] for f in frags],
            # keep true origin for verification
            "_true_origin_page":    true_pi,
            "_true_doc_type":       true_map[true_pi][0],
        })

    return {
        "schema_version": "1.0.0",
        "package_id":     "dummy_pkg",
        "total_pages":    len(pages_out),
        "coord_system":   {"space": "pdf_points", "origin": "top_left",
                           "bbox_format": "x0_y0_x1_y1", "raster_dpi": 150},
        "documents":      [],
        "pages":          pages_out,
        "tables":         tables_out,
        "charts":         [],
        "render_mode":    "digital",
    }


# ── 5. Run pipeline stages ─────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("  DUMMY UNORDERED PDF TEST")
    print("=" * 65)

    extraction = build_dummy_extraction()
    pages  = extraction["pages"]
    tables = extraction["tables"]

    # ── BEFORE view ─────────────────────────────────────────────────────────
    print(f"\n  BEFORE — Jumbled input ({len(pages)} pages, wrong order)\n")
    print(f"  {'Pos':>3}  {'page_index':>10}  {'True origin':>10}  {'True doc_type':<25}  has_table")
    print("  " + "-"*72)
    for p in pages:
        print(f"  {p['page_index']:>3}   p{p['page_index']:>02d}        "
              f"p{p['_true_origin_page']:>02d}         "
              f"{p['_true_doc_type']:<25}  {'YES' if p['has_table'] else '-'}")

    t_start = time.perf_counter()

    # ── Stage 2: Classify ────────────────────────────────────────────────────
    t0 = time.perf_counter()
    page_records = [
        SimpleNamespace(
            page_index       = p["page_index"],
            text             = p.get("text") or "",
            fragment_headers = [[c["text"] for c in f.get("cells", []) if c.get("is_header")]
                                 for f in tables if f["page_index"] == p["page_index"]],
        )
        for p in pages
    ]
    page_records.sort(key=lambda r: r.page_index)

    from classification.classify import classify_pages
    classifications = classify_pages(page_records, max_concurrent=20)
    t_classify = time.perf_counter() - t0

    classify_by_page = {c["page_index"]: c for c in classifications}
    for p in pages:
        cl = classify_by_page.get(p["page_index"], {})
        p["doc_type"]          = cl.get("doc_type")
        p["doc_type_label_id"] = cl.get("doc_type_label_id")

    llm_calls = sum(1 for c in classifications if c.get("method") == "llm")
    print(f"\n  CLASSIFY done in {t_classify*1000:.1f} ms  "
          f"(LLM calls: {llm_calls}/{len(classifications)} — "
          f"{'zero cost, all heuristic!' if llm_calls == 0 else f'{llm_calls} LLM escalations'})")

    # ── Stage 3: Segment ─────────────────────────────────────────────────────
    t0 = time.perf_counter()
    page_texts = [p.get("text") or "" for p in sorted(pages, key=lambda p: p["page_index"])]
    doc_instances = segment_documents(classifications, page_texts)
    t_segment = time.perf_counter() - t0
    print(f"  SEGMENT done in {t_segment*1000:.1f} ms  "
          f"({len(doc_instances)} doc instances)")

    # Patch fields
    page_instance_map: dict[int, str] = {}
    for inst in doc_instances:
        for pi in range(inst.start_page, inst.end_page + 1):
            page_instance_map[pi] = inst.doc_instance_id

    for p in pages:
        pi  = p["page_index"]
        iid = page_instance_map.get(pi)
        p["doc_instance_id"] = iid
        inst_match = next((i for i in doc_instances if i.doc_instance_id == iid), None)
        if inst_match:
            p["page_in_doc"]          = pi - inst_match.start_page + 1
            p["total_pages_in_doc"]   = inst_match.page_count
            p["is_first_page_of_doc"] = (pi == inst_match.start_page)
            p["is_last_page_of_doc"]  = (pi == inst_match.end_page)
            p["boundary"] = ("start" if pi == inst_match.start_page else
                             ("end" if pi == inst_match.end_page else "middle"))

    for frag in tables:
        pi = frag.get("page_index", -1)
        frag["doc_instance_id"] = page_instance_map.get(pi)
        frag["doctype"] = classify_by_page.get(pi, {}).get("doc_type")

    # ── Stage 4: Stitch ──────────────────────────────────────────────────────
    t0 = time.perf_counter()
    cascade = CascadeController()

    frag_groups: dict[str, list] = {}
    for frag in tables:
        iid = frag.get("doc_instance_id") or "_unassigned"
        frag_groups.setdefault(iid, []).append(frag)

    all_threads = []
    table_id_counter = 0
    for iid, frags in frag_groups.items():
        frags_sorted = sorted(frags, key=lambda f: f.get("page_index", 0))
        threads = thread_fragments(frags_sorted, use_llm_arbiter=False)
        all_threads.extend(threads)
        for thread in threads:
            for edge in thread.get("edges", []):
                cascade.record_stitch(
                    frag_id  = edge.get("frag_a", "?"),
                    decision = edge.get("decision", "reject"),
                    score    = edge.get("score", 0.0),
                )

    frag_to_table_id: dict[str, str] = {}
    for thread in all_threads:
        tid = f"table_{table_id_counter:03d}"
        table_id_counter += 1
        for frag in thread.get("fragments", []):
            if frag.get("fragment_id"):
                frag_to_table_id[frag["fragment_id"]] = tid
        for frag in thread.get("fragments", []):
            if frag.get("fragment_id"):
                frag["table_id"] = frag_to_table_id[frag["fragment_id"]]

    t_stitch = time.perf_counter() - t0
    merged = sum(1 for t in all_threads if t["n_pages"] > 1)
    print(f"  STITCH  done in {t_stitch*1000:.1f} ms  "
          f"({len(all_threads)} threads, {merged} multi-page merges)")

    # ── Stage 5: Render ──────────────────────────────────────────────────────
    t0 = time.perf_counter()
    result = render_output(extraction, doc_instances, all_threads, cascade)
    t_render = time.perf_counter() - t0
    print(f"  RENDER  done in {t_render*1000:.1f} ms\n")

    total_ms = (time.perf_counter() - t_start) * 1000
    print(f"  TOTAL pipeline time: {total_ms:.1f} ms  ({len(pages)} pages)\n")

    # ── AFTER view: structured documents ─────────────────────────────────────
    print("=" * 65)
    print("  AFTER — Structured output (doc instances in page order)")
    print("=" * 65)

    result_docs = result.get("documents", [])
    print(f"\n  {'doc_instance_id':<35}  {'pages':>14}  {'count':>5}  attr")
    print("  " + "-"*70)
    for doc in result_docs:
        attr = f"  [{doc['distinguishing_attr']}]" if doc.get("distinguishing_attr") else ""
        print(f"  {doc['doc_instance_id']:<35}  "
              f"p{doc['start_page']:02d} – p{doc['end_page']:02d}  "
              f"{doc['page_count']:>5}{attr}")

    result_tables = result.get("tables", [])
    print(f"\n  STITCHED TABLES ({len(result_tables)} logical tables):")
    print(f"  {'table_id':<12}  {'pages':>14}  {'frags':>5}  {'rows':>5}  stitched?")
    print("  " + "-"*55)
    for tbl in result_tables:
        span   = tbl.get("page_span", {})
        nfrags = tbl.get("n_fragments", 1)
        rows   = tbl.get("row_count_logical", 0)
        flag   = "  FLAGGED" if tbl.get("flagged") else ""
        stitched = "YES" if nfrags > 1 else "-"
        print(f"  {tbl.get('table_id','?'):<12}  "
              f"p{span.get('start_page','?'):02} – p{span.get('end_page','?'):02}  "
              f"{nfrags:>5}  {rows:>5}  {stitched}{flag}")

    # ── Accuracy check ───────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  CLASSIFICATION ACCURACY CHECK")
    print("=" * 65)

    correct = 0
    wrong_list = []
    for p in pages:
        pred = p.get("doc_type")
        true = p["_true_doc_type"]
        if pred == true:
            correct += 1
        else:
            wrong_list.append((p["page_index"], true, pred))

    acc = correct / len(pages) * 100
    print(f"\n  Correct: {correct}/{len(pages)} pages = {acc:.1f}%")
    if wrong_list:
        print(f"\n  Misclassified pages:")
        for pi, true, pred in wrong_list:
            print(f"    p{pi:02d}: true={true}  pred={pred}")
    else:
        print("  All pages classified correctly!")

    # ── Save output ──────────────────────────────────────────────────────────
    out_path = "dummy_pipeline_output.json"
    with open(out_path, "w") as f:
        # Remove internal _true_* keys before saving
        clean_pages = []
        for p in result.get("pages", []):
            cp = {k: v for k, v in p.items() if not k.startswith("_true_")}
            clean_pages.append(cp)
        result["pages"] = clean_pages
        json.dump(result, f, indent=2, default=str)
    print(f"\n  Saved output → {out_path}")
    print()


if __name__ == "__main__":
    main()
