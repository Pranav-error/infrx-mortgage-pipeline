"""
extract.py — Page-level PDF extraction.

Step [2] of the DocCompiler-Lite pipeline:
  - Native PDF pages  -> pdfplumber (free, deterministic, zero AI cost)
  - Scanned/photo pages -> Claude Haiku VLM (selective fallback only)

Outputs:
  PageRecord    — one per PDF page (text, has_text_layer flag, fragment ids)
  FragmentRecord — one per detected table per page (headers, rows, bbox, ...)
"""

from __future__ import annotations

import base64
import io
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import pdfplumber

# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------

# A page with fewer than this many non-whitespace chars is treated as scanned.
MIN_TEXT_CHARS_FOR_NATIVE = 30

# VLM model to use for scanned pages — Haiku is fast and cheap.
VLM_MODEL = "claude-haiku-4-5-20251001"

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class FragmentRecord:
    """One detected table (or table continuation) on a single page."""

    fragment_id: str        # e.g. "frag_12_0" (page_index 12, table 0 on that page)
    page_index: int         # 0-indexed page number (matches labels.json page_index)
    bbox: tuple             # (x0, top, x1, bottom) in PDF points
    headers: list           # first row extracted as column headers
    rows: list              # data rows — list of lists
    last_row: list          # last data row (used by spatial_flow_score in stitch.py)
    page_height: float
    page_width: float
    source: str = "pdfplumber"  # "pdfplumber" | "vlm"


@dataclass
class PageRecord:
    """All extracted data for one PDF page."""

    page_index: int         # 0-indexed (matches labels.json page_index)
    text: str               # full page text (empty string for unprocessed scanned pages)
    has_text_layer: bool    # True = digital PDF, False = scanned/photo
    page_height: float
    page_width: float
    fragment_ids: list = field(default_factory=list)  # fragment_id strings on this page


# ---------------------------------------------------------------------------
# Internal helpers — native pages
# ---------------------------------------------------------------------------


def _has_text_layer(page: pdfplumber.page.Page) -> bool:
    """Return True if pdfplumber can extract meaningful text from the page."""
    text = page.extract_text() or ""
    return len(text.replace(" ", "").replace("\n", "")) >= MIN_TEXT_CHARS_FOR_NATIVE


def _clean_rows(rows: list[list]) -> list[list]:
    """Replace None cells with empty strings."""
    return [[cell if cell is not None else "" for cell in row] for row in rows]


def _extract_native_fragments(page: pdfplumber.page.Page, page_index: int) -> list[FragmentRecord]:
    """Extract all table fragments from a native-text PDF page using pdfplumber."""
    fragments = []
    tables = page.find_tables()

    for t_idx, table in enumerate(tables):
        raw_rows = table.extract()
        if not raw_rows:
            continue

        rows = _clean_rows(raw_rows)
        headers = rows[0]
        data_rows = rows[1:]
        bbox = table.bbox  # (x0, top, x1, bottom)

        fragments.append(FragmentRecord(
            fragment_id=f"frag_{page_index}_{t_idx}",
            page_index=page_index,
            bbox=bbox,
            headers=headers,
            rows=data_rows,
            last_row=data_rows[-1] if data_rows else headers,
            page_height=page.height,
            page_width=page.width,
            source="pdfplumber",
        ))

    return fragments


# ---------------------------------------------------------------------------
# Internal helpers — scanned pages (VLM fallback)
# ---------------------------------------------------------------------------


def _render_page_to_base64(pdf_path: str, page_num: int) -> Optional[str]:
    """
    Render a single PDF page to a PNG and return it as a base64 string.
    Requires pdf2image + poppler. Returns None if unavailable.
    """
    try:
        from pdf2image import convert_from_path  # type: ignore
    except ImportError:
        print(
            "[WARN] pdf2image not installed — cannot render scanned pages.\n"
            "       Install with: pip install pdf2image\n"
            "       Also ensure poppler is installed (brew install poppler on macOS)."
        )
        return None

    images = convert_from_path(pdf_path, first_page=page_num, last_page=page_num, dpi=150)
    if not images:
        return None

    buf = io.BytesIO()
    images[0].save(buf, format="PNG")
    return base64.standard_b64encode(buf.getvalue()).decode("utf-8")


_VLM_PROMPT = """You are a document parser. Extract all tables from this scanned document page.

Return a JSON object with EXACTLY this schema (no markdown, no explanation — raw JSON only):
{
  "page_text": "<all readable text on the page as a single string>",
  "tables": [
    {
      "headers": ["col1", "col2", "..."],
      "rows": [["val", "val", "..."], "..."],
      "bbox_pct": [x0, top, x1, bottom]
    }
  ]
}

bbox_pct values are fractions of page width/height in range [0.0, 1.0].
If there are no tables, return an empty tables array.
Return ONLY the JSON object."""


def _vlm_extract_page(
    pdf_path: str,
    page_index: int,
    page: pdfplumber.page.Page,
    client,  # anthropic.Anthropic
) -> tuple[str, list[FragmentRecord]]:
    """Use Claude Haiku VLM to extract text and tables from a scanned page."""
    # pdf2image is 1-indexed for first_page/last_page
    img_b64 = _render_page_to_base64(pdf_path, page_index + 1)
    if img_b64 is None:
        return "", []

    response = client.messages.create(
        model=VLM_MODEL,
        max_tokens=4096,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": img_b64,
                        },
                    },
                    {"type": "text", "text": _VLM_PROMPT},
                ],
            }
        ],
    )

    raw = response.content[0].text.strip()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            print(f"[WARN] VLM returned unparseable JSON for page_index {page_index}")
            return "", []
        data = json.loads(match.group())

    page_text = data.get("page_text", "")
    fragments: list[FragmentRecord] = []

    for t_idx, tbl in enumerate(data.get("tables", [])):
        headers = tbl.get("headers", [])
        rows = tbl.get("rows", [])
        bbox_pct = tbl.get("bbox_pct", [0.0, 0.0, 1.0, 1.0])

        # Convert percentage bbox -> PDF points
        bbox = (
            bbox_pct[0] * page.width,
            bbox_pct[1] * page.height,
            bbox_pct[2] * page.width,
            bbox_pct[3] * page.height,
        )

        fragments.append(FragmentRecord(
            fragment_id=f"frag_{page_index}_{t_idx}",
            page_index=page_index,
            bbox=bbox,
            headers=headers,
            rows=rows,
            last_row=rows[-1] if rows else headers,
            page_height=page.height,
            page_width=page.width,
            source="vlm",
        ))

    return page_text, fragments


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def extract_pdf(
    pdf_path: str,
    anthropic_api_key: Optional[str] = None,
) -> tuple[list[PageRecord], list[FragmentRecord]]:
    """
    Main extraction entry point.

    Args:
        pdf_path:           Path to the multi-page PDF.
        anthropic_api_key:  Anthropic API key for VLM fallback on scanned pages.
                            If None, scanned pages produce empty text and no fragments.

    Returns:
        (pages, fragments)
        - pages:     one PageRecord per PDF page
        - fragments: one FragmentRecord per detected table per page
    """
    pdf_path = str(Path(pdf_path).resolve())

    client = None
    if anthropic_api_key:
        import anthropic as _anthropic
        client = _anthropic.Anthropic(api_key=anthropic_api_key)

    all_pages: list[PageRecord] = []
    all_fragments: list[FragmentRecord] = []
    vlm_call_count = 0

    with pdfplumber.open(pdf_path) as pdf:
        total = len(pdf.pages)
        print(f"[extract] {pdf_path}: {total} page(s)")

        for page_index, page in enumerate(pdf.pages):  # 0-indexed, matches labels.json
            native = _has_text_layer(page)

            if native:
                text = page.extract_text() or ""
                frags = _extract_native_fragments(page, page_index)
            else:
                if client:
                    print(f"[extract] page_index={page_index}: scanned — VLM fallback")
                    text, frags = _vlm_extract_page(pdf_path, page_index, page, client)
                    vlm_call_count += 1
                else:
                    text, frags = "", []

            pr = PageRecord(
                page_index=page_index,
                text=text,
                has_text_layer=native,
                page_height=page.height,
                page_width=page.width,
                fragment_ids=[f.fragment_id for f in frags],
            )
            all_pages.append(pr)
            all_fragments.extend(frags)

            if (page_index + 1) % 100 == 0 or page_index == total - 1:
                print(f"[extract]   {page_index + 1}/{total} done")

    native_count = sum(1 for p in all_pages if p.has_text_layer)
    scanned_count = len(all_pages) - native_count
    print(
        f"[extract] Complete — {len(all_pages)} pages, {len(all_fragments)} fragments\n"
        f"          Native (pdfplumber): {native_count}  |  Scanned (VLM): {scanned_count}"
        + (f"  |  VLM calls: {vlm_call_count}" if vlm_call_count else "")
    )

    return all_pages, all_fragments


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------


def pages_to_dict(pages: list[PageRecord]) -> list[dict]:
    return [asdict(p) for p in pages]


def fragments_to_dict(fragments: list[FragmentRecord]) -> list[dict]:
    return [asdict(f) for f in fragments]
