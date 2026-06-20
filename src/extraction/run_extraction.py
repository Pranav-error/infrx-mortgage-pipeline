"""
run_extraction.py — Run page extraction on a dataset package and save results.

Usage:
    python3 src/run_extraction.py --pkg DataSet/pkg_000000 [--api-key sk-ant-...]
    python3 src/run_extraction.py --pkg DataSet/pkg_000000 --no-vlm  # skip scanned pages

Output (written to pkg dir):
    extraction_results.json   — pages + fragments
    extraction_summary.txt    — quick stats + comparison vs labels ground truth
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from extraction.extract import extract_pdf


# ---------------------------------------------------------------------------
# Ground-truth loader (labels.json)
# ---------------------------------------------------------------------------

def load_labels(pkg_dir: Path) -> dict:
    labels_path = pkg_dir / "labels.json"
    with open(labels_path) as f:
        return json.load(f)


def summarise_labels(labels: dict) -> dict:
    """Pull the key ground-truth facts we'll compare against."""
    pages = labels["pages"]
    docs  = labels["documents"]
    tables = labels["tables"]

    digital_pages  = [p for p in pages if p["render_mode"] == "digital"]
    scanned_pages  = [p for p in pages if p["render_mode"] == "scanned"]
    pages_w_tables = [p for p in pages if p["has_table"]]
    multi_span_tbl = [t for t in tables
                      if t["page_span"]["end_page"] > t["page_span"]["start_page"]]

    return {
        "total_pages":        len(pages),
        "digital_pages":      len(digital_pages),
        "scanned_pages":      len(scanned_pages),
        "document_instances": len(docs),
        "total_tables":       len(tables),
        "multi_page_tables":  len(multi_span_tbl),
        "pages_with_tables":  len(pages_w_tables),
    }


# ---------------------------------------------------------------------------
# Quick extraction stats
# ---------------------------------------------------------------------------

def summarise_extraction(pages, fragments) -> dict:
    digital = sum(1 for p in pages if p["render_mode"] == "digital")
    scanned = len(pages) - digital
    frags_from_plumber = sum(1 for f in fragments if f["source"] == "pdfplumber")
    frags_from_vlm     = sum(1 for f in fragments if f["source"] == "vlm")
    pages_with_frags   = len({f["page_index"] for f in fragments})

    total_cells = sum(len(f.get("cells", [])) for f in fragments)
    return {
        "total_pages":       len(pages),
        "digital_pages":     digital,
        "scanned_pages":     scanned,
        "total_fragments":   len(fragments),
        "frags_pdfplumber":  frags_from_plumber,
        "frags_vlm":         frags_from_vlm,
        "pages_with_tables": pages_with_frags,
        "total_cells":       total_cells,
    }


# ---------------------------------------------------------------------------
# Compare extraction vs ground truth
# ---------------------------------------------------------------------------

def compare_vs_labels(ext_summary: dict, gt_summary: dict) -> list[str]:
    """Return a list of comparison lines for the summary report."""
    lines = []

    def row(label, extracted, expected):
        match = "OK" if extracted == expected else f"DIFF (got {extracted}, expected {expected})"
        lines.append(f"  {label:<35} {match}")

    row("total pages",         ext_summary["total_pages"],       gt_summary["total_pages"])
    row("digital pages",       ext_summary["digital_pages"],     gt_summary["digital_pages"])
    row("scanned pages",       ext_summary["scanned_pages"],     gt_summary["scanned_pages"])
    row("pages with tables",   ext_summary["pages_with_tables"], gt_summary["pages_with_tables"])

    # fragments vs tables is not 1:1 (fragment = table on one page, GT table can span pages)
    lines.append(
        f"  {'fragments extracted':<35} {ext_summary['total_fragments']}"
        f"  (GT has {gt_summary['total_tables']} logical tables,"
        f" {gt_summary['multi_page_tables']} span multiple pages)"
    )
    return lines


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pkg",     required=True,  help="Path to a dataset package dir (e.g. DataSet/pkg_000000)")
    parser.add_argument("--api-key", default=None,   help="Anthropic API key for VLM fallback on scanned pages")
    parser.add_argument("--no-vlm",  action="store_true", help="Skip VLM fallback (scanned pages get empty text)")
    args = parser.parse_args()

    pkg_dir  = Path(args.pkg).resolve()
    pdf_path = pkg_dir / "package.pdf"

    if not pdf_path.exists():
        print(f"ERROR: {pdf_path} not found")
        sys.exit(1)

    api_key = None if args.no_vlm else (args.api_key or os.environ.get("ANTHROPIC_API_KEY"))
    if not api_key and not args.no_vlm:
        print("[WARN] No Anthropic API key provided. Scanned pages will be skipped.")
        print("       Pass --api-key or set ANTHROPIC_API_KEY, or use --no-vlm to suppress this warning.")

    # --- Run extraction ---
    t0 = time.time()
    result = extract_pdf(str(pdf_path), anthropic_api_key=api_key)
    elapsed = time.time() - t0

    # --- Save results ---
    out_path = pkg_dir / "extraction_results.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n[run] Results saved to {out_path}")

    pages     = result["pages"]
    fragments = result["tables"]

    # --- Compare vs ground truth ---
    labels      = load_labels(pkg_dir)
    gt_summary  = summarise_labels(labels)
    ext_summary = summarise_extraction(pages, fragments)

    report_lines = [
        f"Package:      {pkg_dir.name}",
        f"PDF:          {pdf_path.name}  ({pdf_path.stat().st_size / 1e6:.1f} MB)",
        f"Elapsed:      {elapsed:.1f}s",
        "",
        "=== Extraction results ===",
        f"  Pages processed:              {ext_summary['total_pages']}",
        f"  Digital (pdfplumber):         {ext_summary['digital_pages']}",
        f"  Scanned (VLM / skipped):      {ext_summary['scanned_pages']}",
        f"  Total fragments:              {ext_summary['total_fragments']}",
        f"    from pdfplumber:            {ext_summary['frags_pdfplumber']}",
        f"    from VLM:                   {ext_summary['frags_vlm']}",
        f"  Total cells extracted:        {ext_summary['total_cells']}",
        f"  Pages with >= 1 fragment:     {ext_summary['pages_with_tables']}",
        "",
        "=== vs Ground Truth (labels.json) ===",
    ] + compare_vs_labels(ext_summary, gt_summary) + [
        "",
        "=== Ground Truth reference ===",
        f"  Document instances:           {gt_summary['document_instances']}",
        f"  Logical tables total:         {gt_summary['total_tables']}",
        f"  Multi-page tables:            {gt_summary['multi_page_tables']}",
    ]

    report = "\n".join(report_lines)
    print("\n" + report)

    summary_path = pkg_dir / "extraction_summary.txt"
    with open(summary_path, "w") as f:
        f.write(report + "\n")
    print(f"\n[run] Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
