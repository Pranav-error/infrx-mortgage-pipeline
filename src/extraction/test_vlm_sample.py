"""
test_vlm_sample.py — Test VLM extraction on a small sample of scanned pages.

Usage:
    python3 src/test_vlm_sample.py --pkg DataSet/pkg_000000 --api-key sk-ant-...
    python3 src/test_vlm_sample.py --pkg DataSet/pkg_000000  # uses ANTHROPIC_API_KEY env var
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from extract import extract_pdf, PageRecord, FragmentRecord


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pkg",     required=True)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--pages",   type=int, default=3, help="How many scanned pages to test (default: 3)")
    args = parser.parse_args()

    pkg_dir = Path(args.pkg).resolve()
    api_key = args.api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: Provide --api-key or set ANTHROPIC_API_KEY")
        sys.exit(1)

    with open(pkg_dir / "labels.json") as f:
        labels = json.load(f)

    # Pick scanned pages that have tables (best test case)
    scanned_with_tables = [
        p for p in labels["pages"]
        if p["render_mode"] == "scanned" and p["has_table"]
    ][:args.pages]

    print(f"Testing VLM on {len(scanned_with_tables)} scanned pages with known tables:")
    for p in scanned_with_tables:
        print(f"  page_index={p['page_index']}  doc_type={p['doc_type']}  table_ids={p['table_ids']}")

    print()

    import pdfplumber
    import base64, io, json as _json, re
    import anthropic

    client = anthropic.Anthropic(api_key=api_key)

    try:
        from pdf2image import convert_from_path
    except ImportError:
        print("ERROR: pdf2image not installed. Run: pip3 install pdf2image")
        sys.exit(1)

    from extract import _VLM_PROMPT, VLM_MODEL

    pdf_path = str(pkg_dir / "package.pdf")

    for gt_page in scanned_with_tables:
        page_index = gt_page["page_index"]
        print(f"\n{'='*60}")
        print(f"page_index={page_index}  doc_type={gt_page['doc_type']}")

        # Render page (pdf2image is 1-indexed)
        images = convert_from_path(pdf_path, first_page=page_index + 1, last_page=page_index + 1, dpi=150)
        if not images:
            print("  Could not render page")
            continue

        buf = io.BytesIO()
        images[0].save(buf, format="PNG")
        img_b64 = base64.standard_b64encode(buf.getvalue()).decode("utf-8")
        img_size_kb = len(buf.getvalue()) // 1024
        print(f"  Rendered: {images[0].size[0]}x{images[0].size[1]}px  ({img_size_kb} KB)")

        response = client.messages.create(
            model=VLM_MODEL,
            max_tokens=4096,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": img_b64}},
                    {"type": "text", "text": _VLM_PROMPT},
                ],
            }],
        )

        raw = response.content[0].text.strip()
        usage = response.usage
        print(f"  Tokens — input: {usage.input_tokens}  output: {usage.output_tokens}")

        try:
            data = _json.loads(raw)
        except _json.JSONDecodeError:
            m = re.search(r"\{.*\}", raw, re.DOTALL)
            data = _json.loads(m.group()) if m else {}

        text_preview = data.get("page_text", "")[:120].replace("\n", " ")
        tables = data.get("tables", [])
        print(f"  Text preview: {repr(text_preview)}")
        print(f"  Tables found: {len(tables)}")

        for i, tbl in enumerate(tables):
            headers = tbl.get("headers", [])
            rows    = tbl.get("rows", [])
            print(f"    Table {i}: headers={headers}  rows={len(rows)}")
            if rows:
                print(f"      First row: {rows[0]}")
                print(f"      Last row:  {rows[-1]}")

        # Ground truth table info for this page
        print(f"\n  GT table_ids on this page: {gt_page['table_ids']}")
        for tid in gt_page["table_ids"]:
            gt_tbl = next((t for t in labels["tables"] if t["table_id"] == tid), None)
            if gt_tbl:
                print(f"    {tid}: {gt_tbl['row_count_logical']} logical rows, "
                      f"spans pages {gt_tbl['page_span']['start_page']}-{gt_tbl['page_span']['end_page']}, "
                      f"header_repeats={gt_tbl['header_repeats_each_page']}")

    print("\nDone.")


if __name__ == "__main__":
    main()
