"""
extract.py — Page-level PDF extraction (parallel-optimised).

Two-pass pipeline:
  Pass 1 (serial)   — pdfplumber over ALL pages: free, deterministic, ~0.1s/page
                       Identifies digital vs scanned, extracts digital fragments.
  Pass 2 (parallel) — VLM (Claude Haiku) over SCANNED pages only.
                       Render + API call run concurrently in a thread pool.
                       Default 15 workers → ~20 s for 148 pages, ~3-4 min for 2000 pages.

Output structure mirrors labels.json exactly:
  {
    "schema_version", "package_id", "total_pages", "coord_system",
    "documents",   # [] — filled by segment.py
    "pages",       # one record per PDF page
    "tables",      # one fragment per detected table per page
    "charts",      # [] — out of scope
    "render_mode"
  }

Fields filled here:      page_index, width, height, render_mode, has_table,
                         scan_image_size_px, scan_transform (digital=identity),
                         cells (row_idx, col_idx, bbox, bbox_px, text, is_header)
Fields left null (→):   doc_type, doc_instance_id, boundary  → classify.py / segment.py
                         table_id, page_span, header_repeats  → stitch.py
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

import os as _os
import pdfplumber

# ---------------------------------------------------------------------------
# Backend selection — OpenAI fallback when Anthropic credits are exhausted
# ---------------------------------------------------------------------------
_USE_OPENAI = bool(_os.environ.get("OPENAI_API_KEY"))
if _USE_OPENAI:
    from openai import AsyncOpenAI as _AsyncOpenAI

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SCHEMA_VERSION   = "1.0.0"
MIN_TEXT_CHARS   = 30          # below this → page is treated as scanned
RENDER_DPI       = 150         # full DPI for coordinate system (bbox_px mapping)
VLM_RENDER_DPI   = 100         # lower DPI for VLM — still legible, 44% smaller images
VLM_MAX_WIDTH    = 512         # resize images to max 512px wide — massive token savings
PT_TO_PX         = RENDER_DPI / 72.0
VLM_MODEL        = "gpt-4o-mini" if _USE_OPENAI else "claude-haiku-4-5-20251001"
DEFAULT_WORKERS  = 50          # concurrent async VLM requests — push rate limits hard
MAX_VLM_RETRIES  = 4           # retry on transient API / connection errors
VLM_MAX_TOKENS   = 4096        # sufficient for most pages; reduces worst-case latency
VLM_IMAGE_FORMAT = "JPEG"      # JPEG is 5-10x smaller than PNG for scanned pages
VLM_JPEG_QUALITY = 70          # lower quality OK at 512px — faster transfers

IDENTITY_TRANSFORM = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
COORD_SYSTEM = {
    "space": "pdf_points",
    "origin": "top_left",
    "bbox_format": "x0_y0_x1_y1",
    "raster_dpi": RENDER_DPI,
}

# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _has_text_layer(page: pdfplumber.page.Page) -> bool:
    text = page.extract_text() or ""
    return len(text.replace(" ", "").replace("\n", "")) >= MIN_TEXT_CHARS


def _pt_to_px(bbox: list | tuple) -> list:
    return [round(v * PT_TO_PX, 3) for v in bbox]


# Regex patterns for value_type inference (order matters)
_DATE_RE     = re.compile(r"^\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4}$")
_CURRENCY_RE = re.compile(r"^[\$\-\+\(]?[\d,]+\.\d{2}\)?$")
_INTEGER_RE  = re.compile(r"^[\-\+]?\d{1,3}(,\d{3})*$")
_PCT_RE      = re.compile(r"^[\d\.]+\s*%$")


def _infer_value_types(rows: list, num_cols: int) -> list[str]:
    """
    Infer column value types from the first non-empty data row.
    Returns a list of type strings, one per column:
      "date" | "currency" | "integer" | "percent" | "text"
    Works on any document — no mortgage-specific knowledge assumed.
    """
    for row in rows:
        if any(str(c).strip() for c in row):
            types = []
            for i in range(num_cols):
                s = str(row[i]).strip() if i < len(row) else ""
                if   _DATE_RE.match(s):     types.append("date")
                elif _CURRENCY_RE.match(s): types.append("currency")
                elif _PCT_RE.match(s):      types.append("percent")
                elif _INTEGER_RE.match(s):  types.append("integer")
                else:                       types.append("text")
            return types
    return ["text"] * num_cols


def _column_fingerprint_native(table, page_width: float) -> list[float]:
    """
    Compute normalised x-start positions of each column from pdfplumber header row.
    Example: [0.06, 0.18, 0.64, 0.73, 0.86] for a 5-column bank statement.
    """
    header_row = table.rows[0]
    fps = []
    for cell_bbox in header_row.cells:
        if cell_bbox is not None:
            fps.append(round(cell_bbox[0] / page_width, 3))
    return fps


def _column_fingerprint_synthetic(num_cols: int) -> list[float]:
    """
    Fallback fingerprint when real x-positions are unavailable (VLM pages).
    Evenly-spaced positions are approximate but *consistent across pages*,
    which is all the stitcher needs to compare two fragments.
    """
    if num_cols == 0:
        return []
    return [round(i / num_cols, 3) for i in range(num_cols)]


# ---------------------------------------------------------------------------
# PASS 1 — pdfplumber (digital pages)
# ---------------------------------------------------------------------------


def _build_cells_native(table, page_index: int) -> tuple[list, list, int]:
    """Per-cell extraction from a pdfplumber Table. Returns (cells, columns, row_count)."""
    raw_rows = table.extract()
    if not raw_rows:
        return [], [], 0

    num_cols = max(len(r) for r in raw_rows)
    columns  = [{"col_idx": i} for i in range(num_cols)]
    cells    = []

    for r_local, (row_data, pdfrow) in enumerate(zip(raw_rows, table.rows)):
        is_header  = r_local == 0
        row_idx    = -1 if is_header else r_local - 1

        for col_idx, (text, cell_bbox) in enumerate(zip(row_data, pdfrow.cells)):
            if cell_bbox is None:
                continue
            bbox = list(cell_bbox)          # (x0, top, x1, bottom)
            cells.append({
                "page_index": page_index,
                "row_idx":    row_idx,
                "col_idx":    col_idx,
                "is_header":  is_header,
                "text":       str(text) if text is not None else "",
                "bbox":       bbox,
                "bbox_px":    _pt_to_px(bbox),
            })

    return cells, columns, max(len(raw_rows) - 1, 0)


def _extract_native_fragments(page: pdfplumber.page.Page, page_index: int) -> list[dict]:
    frags = []
    for t_idx, table in enumerate(page.find_tables()):
        cells, columns, row_count = _build_cells_native(table, page_index)
        if not cells:
            continue

        num_cols      = len(columns)
        headers       = [c["text"] for c in cells if c["is_header"]]
        last_row_data = [c["text"] for c in cells if c["row_idx"] == row_count - 1]
        data_rows     = [[c["text"] for c in cells if c["row_idx"] == r]
                         for r in range(row_count)]

        frags.append({
            # labels.json table fields
            "table_id":                 None,           # [STITCH]
            "doc_instance_id":          None,           # [SEGMENT]
            "doctype":                  None,           # [CLASSIFY]
            "page_span":                {"start_page": page_index, "end_page": page_index},
            "header_repeats_each_page": None,           # [STITCH]
            "columns":                  columns,
            "row_count_logical":        row_count,
            "cells":                    cells,
            # stitcher signals
            "column_fingerprint": _column_fingerprint_native(table, page.width),
            "value_types":        _infer_value_types(data_rows, num_cols),
            # extraction extras
            "fragment_id":   f"frag_{page_index}_{t_idx}",
            "page_index":    page_index,
            "bbox":          list(table.bbox),
            "headers":       headers,
            "last_row_text": last_row_data,
            "page_height":   page.height,
            "page_width":    page.width,
            "source":        "pdfplumber",
        })
    return frags


# ---------------------------------------------------------------------------
# PASS 2 — VLM (scanned pages), parallelised
# ---------------------------------------------------------------------------

_VLM_PROMPT = """You are a document parser. Extract all tables from this scanned document page.

Return a JSON object with EXACTLY this schema (raw JSON only, no markdown):
{
  "page_text": "<all readable text as a single string>",
  "tables": [
    {
      "headers": ["col1", "col2", "..."],
      "rows": [["val", "val", "..."], "..."],
      "bbox_pct": [x0, top, x1, bottom]
    }
  ]
}

bbox_pct values are fractions of page dimensions in [0.0, 1.0].
If there are no tables, return an empty tables array."""

# Lightweight OCR-only prompt — just extract text, no table parsing.
# ~3x faster response because output is much shorter.
_VLM_PROMPT_OCR_ONLY = """Read all text from this scanned document page.
Return ONLY the text as a single string. No JSON, no markdown, no explanation.
Preserve layout: headers, paragraphs, labels, numbers."""


def _is_blank_page(img, threshold: int = 250, min_dark_pct: float = 0.02) -> bool:
    """
    Quick check: is this rendered page nearly blank (white/empty)?
    If less than min_dark_pct of pixels are darker than threshold, skip VLM.
    """
    import numpy as np
    arr = np.array(img.convert("L"))  # grayscale
    dark_pixels = np.sum(arr < threshold)
    return (dark_pixels / arr.size) < min_dark_pct


def _render_pages_batch(pdf_path: str, page_indices: list[int]) -> dict[int, str]:
    """
    Render scanned pages to base64 JPEG for VLM.
    Processes in chunks of RENDER_BATCH_SIZE to avoid loading the entire
    PDF into memory at once (critical for 2000+ page PDFs).
    Skips near-blank pages automatically.
    Returns {page_index: base64_str}.
    """
    try:
        from pdf2image import convert_from_path  # type: ignore
    except ImportError:
        print("[WARN] pdf2image not installed — pip install pdf2image && brew install poppler")
        return {}

    if not page_indices:
        return {}

    RENDER_BATCH_SIZE = 50  # render 50 pages at a time to limit memory

    # Group into contiguous runs to minimise poppler calls
    sorted_indices = sorted(page_indices)
    runs: list[tuple[int, int]] = []
    run_start = sorted_indices[0]
    run_end   = sorted_indices[0]
    for idx in sorted_indices[1:]:
        if idx == run_end + 1:
            run_end = idx
        else:
            runs.append((run_start + 1, run_end + 1))
            run_start = run_end = idx
    runs.append((run_start + 1, run_end + 1))

    # Split large runs into batches of RENDER_BATCH_SIZE
    batches: list[tuple[int, int]] = []
    for first, last in runs:
        while first <= last:
            batch_end = min(first + RENDER_BATCH_SIZE - 1, last)
            batches.append((first, batch_end))
            first = batch_end + 1

    result: dict[int, str] = {}
    skipped_blank = 0
    rendered_count = 0
    t0 = time.time()

    for batch_idx, (first, last) in enumerate(batches):
        imgs = convert_from_path(pdf_path, first_page=first, last_page=last, dpi=VLM_RENDER_DPI)
        for offset, img in enumerate(imgs):
            page_index = first - 1 + offset

            if _is_blank_page(img):
                skipped_blank += 1
                continue

            # Resize to max width — huge token savings for vision API
            if img.width > VLM_MAX_WIDTH:
                ratio = VLM_MAX_WIDTH / img.width
                new_size = (VLM_MAX_WIDTH, int(img.height * ratio))
                img = img.resize(new_size)

            buf = io.BytesIO()
            if VLM_IMAGE_FORMAT == "JPEG":
                img.save(buf, format="JPEG", quality=VLM_JPEG_QUALITY)
            else:
                img.save(buf, format="PNG")
            result[page_index] = base64.standard_b64encode(buf.getvalue()).decode()

        rendered_count += len(imgs)
        elapsed = time.time() - t0
        rate = rendered_count / elapsed if elapsed > 0 else 0
        remaining = len(sorted_indices) - rendered_count
        eta = remaining / rate if rate > 0 else 0
        print(f"[extract]   Render {rendered_count}/{len(sorted_indices)}  "
              f"{rate:.1f} pages/s  ETA {eta:.0f}s", flush=True)

    if skipped_blank:
        print(f"[extract]   Skipped {skipped_blank} blank pages (no content to OCR)")

    return result


def _build_cells_vlm(
    headers: list, rows: list, bbox_pct: list, page_index: int, pw: float, ph: float
) -> tuple[list, list, int]:
    """Build cells from VLM output. Column bboxes are evenly approximated."""
    num_cols = max(len(headers), max((len(r) for r in rows), default=0))
    if num_cols == 0:
        return [], [], 0
    columns  = [{"col_idx": i} for i in range(num_cols)]

    x0 = bbox_pct[0] * pw;  x1 = bbox_pct[2] * pw
    y0 = bbox_pct[1] * ph;  y1 = bbox_pct[3] * ph
    col_w  = (x1 - x0) / num_cols
    total_r = 1 + len(rows)
    row_h  = (y1 - y0) / total_r if total_r else 1

    cells = []
    for r_local, row_data in enumerate([headers] + rows):
        is_header = r_local == 0
        row_idx   = -1 if is_header else r_local - 1
        ry0 = y0 + r_local * row_h
        ry1 = ry0 + row_h
        for col_idx in range(num_cols):
            cx0   = x0 + col_idx * col_w
            bbox  = [round(cx0, 3), round(ry0, 3), round(cx0 + col_w, 3), round(ry1, 3)]
            text  = str(row_data[col_idx]) if col_idx < len(row_data) else ""
            cells.append({
                "page_index": page_index,
                "row_idx":    row_idx,
                "col_idx":    col_idx,
                "is_header":  is_header,
                "text":       text,
                "bbox":       bbox,
                "bbox_px":    _pt_to_px(bbox),
            })
    return cells, columns, len(rows)


def _parse_vlm_response(raw: str, page_index: int, pw: float, ph: float) -> tuple[str, list[dict]]:
    """Parse VLM JSON response into (page_text, fragments). Never raises."""
    data = {}
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        # Try to extract the outermost JSON object (handles markdown fences + truncation)
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            try:
                data = json.loads(m.group())
            except json.JSONDecodeError:
                print(f"[WARN] page {page_index}: VLM returned unparseable JSON — skipping tables")
                # Still try to extract page_text from raw string
                t = re.search(r'"page_text"\s*:\s*"(.*?)"', raw, re.DOTALL)
                return (t.group(1) if t else ""), []

    page_text = data.get("page_text", "")
    fragments = []

    for t_idx, tbl in enumerate(data.get("tables", [])):
        headers  = tbl.get("headers", [])
        rows     = tbl.get("rows", [])
        bbox_pct = tbl.get("bbox_pct", [0.0, 0.0, 1.0, 1.0])

        cells, columns, row_count = _build_cells_vlm(headers, rows, bbox_pct, page_index, pw, ph)
        if not cells:
            continue

        num_cols = len(columns)
        tbl_bbox = [bbox_pct[0]*pw, bbox_pct[1]*ph, bbox_pct[2]*pw, bbox_pct[3]*ph]
        fragments.append({
            "table_id":                 None,
            "doc_instance_id":          None,
            "doctype":                  None,
            "page_span":                {"start_page": page_index, "end_page": page_index},
            "header_repeats_each_page": None,
            "columns":                  columns,
            "row_count_logical":        row_count,
            "cells":                    cells,
            # stitcher signals
            # synthetic fingerprint: consistent across pages for same table structure
            "column_fingerprint": _column_fingerprint_synthetic(num_cols),
            "value_types":        _infer_value_types(rows, num_cols),
            # extraction extras
            "fragment_id":   f"frag_{page_index}_{t_idx}",
            "page_index":    page_index,
            "bbox":          [round(v, 3) for v in tbl_bbox],
            "headers":       headers,
            "last_row_text": rows[-1] if rows else headers,
            "page_height":   ph,
            "page_width":    pw,
            "source":        "vlm",
        })
    return page_text, fragments


async def _vlm_one_page_async(
    semaphore: asyncio.Semaphore,
    async_client,
    img_b64: str,
    page_index: int,
    pw: float,
    ph: float,
    progress: dict,
    ocr_only: bool = False,
) -> tuple[int, str, list[dict]]:
    """
    Pure-async VLM call for one pre-rendered page.
    ocr_only=True uses a lightweight prompt (text only, no table parsing) — 3x faster.
    Supports both OpenAI (GPT-4o-mini) and Anthropic (Claude Haiku) backends.
    """
    prompt = _VLM_PROMPT_OCR_ONLY if ocr_only else _VLM_PROMPT
    max_tok = 1500 if ocr_only else VLM_MAX_TOKENS
    async with semaphore:
        for attempt in range(MAX_VLM_RETRIES):
            try:
                _mime = "image/jpeg" if VLM_IMAGE_FORMAT == "JPEG" else "image/png"
                if _USE_OPENAI:
                    resp = await async_client.chat.completions.create(
                        model=VLM_MODEL,
                        max_tokens=max_tok,
                        messages=[{
                            "role": "user",
                            "content": [
                                {"type": "image_url", "image_url": {
                                    "url": f"data:{_mime};base64,{img_b64}",
                                }},
                                {"type": "text", "text": prompt},
                            ],
                        }],
                    )
                    raw_text = resp.choices[0].message.content.strip()
                else:
                    resp = await async_client.messages.create(
                        model=VLM_MODEL,
                        max_tokens=max_tok,
                        messages=[{
                            "role": "user",
                            "content": [
                                {"type": "image", "source": {
                                    "type": "base64", "media_type": _mime, "data": img_b64,
                                }},
                                {"type": "text", "text": prompt},
                            ],
                        }],
                    )
                    raw_text = resp.content[0].text.strip()
                break
            except Exception as e:
                if attempt == MAX_VLM_RETRIES - 1:
                    print(f"[WARN] page {page_index}: failed after {MAX_VLM_RETRIES} attempts — {type(e).__name__}")
                    return page_index, "", []
                await asyncio.sleep(2 ** attempt)   # 1s → 2s → 4s → 8s

        if ocr_only:
            page_text, frags = raw_text, []
        else:
            page_text, frags = _parse_vlm_response(raw_text, page_index, pw, ph)

        progress["done"] += 1
        done = progress["done"]
        if done % 10 == 0 or done == progress["total"]:
            elapsed = time.time() - progress["t0"]
            rate    = done / elapsed if elapsed > 0 else 0
            eta     = (progress["total"] - done) / rate if rate > 0 else 0
            print(f"[extract]   VLM  {done}/{progress['total']}  "
                  f"{rate:.1f} pages/s  ETA {eta:.0f}s", flush=True)

        return page_index, page_text, frags


def _tesseract_ocr_batch(rendered_images: dict[int, str]) -> dict[int, str]:
    """
    Run Tesseract OCR locally on rendered page images. FREE and parallelised.
    Uses ThreadPoolExecutor for ~4x speedup on multi-core machines.
    Returns {page_index: extracted_text}.
    """
    try:
        import pytesseract
        from PIL import Image
    except ImportError:
        print("[WARN] pytesseract not installed — pip install pytesseract")
        return {}

    def _ocr_one(item: tuple[int, str]) -> tuple[int, str]:
        pi, img_b64 = item
        try:
            img_bytes = base64.standard_b64decode(img_b64)
            img = Image.open(io.BytesIO(img_bytes))
            text = pytesseract.image_to_string(img)
            return pi, text
        except Exception as e:
            return pi, ""

    results: dict[int, str] = {}
    with ThreadPoolExecutor(max_workers=8) as pool:
        for pi, text in pool.map(_ocr_one, rendered_images.items()):
            results[pi] = text
    return results


# Heuristic: pages with these patterns likely have tables worth sending to VLM
_TABLE_HINT_RE = re.compile(
    r"(\$[\d,]+\.\d{2})|"                        # dollar amounts
    r"(\d{1,2}/\d{1,2}/\d{2,4})|"                # dates
    r"(beginning balance|ending balance|total)|"   # financial keywords
    r"(\|\s+\w+\s+\|)|"                           # pipe-delimited table
    r"(\d+\.\d{2}\s+\d+\.\d{2})",                # two decimal numbers in a row
    re.IGNORECASE,
)


def _page_needs_vlm(text: str) -> bool:
    """Check if OCR'd text suggests the page has financial/structured content.
    Mortgage docs almost always have dollar amounts, dates, or financial terms.
    Only skip VLM for truly plain text pages (letters, narratives, cover pages).
    """
    if not text or len(text.strip()) < 30:
        return True  # no Tesseract text = definitely need VLM
    hits = len(_TABLE_HINT_RE.findall(text))
    if hits >= 1:
        return True  # any financial pattern = send to VLM
    # Check for common financial/legal terms Tesseract might have caught
    lower = text.lower()
    financial_terms = ["bank", "account", "statement", "balance", "deposit",
                       "loan", "mortgage", "payment", "credit", "tax",
                       "employer", "income", "insurance", "contract"]
    if any(term in lower for term in financial_terms):
        return True
    return False  # truly non-financial text — Tesseract is sufficient


async def _run_vlm_pass(
    scanned_queue: list[dict],
    pdf_path: str,
    api_key: str,
    max_concurrent: int,
) -> dict[int, tuple[str, list]]:
    """
    Two-step async pass:
      Step A (sync): batch-render scanned pages in chunks of 50 (memory-safe).
      Step B (async): fire VLM API calls with high concurrency (25+).

    Blank pages are auto-skipped during rendering (no VLM cost).
    Tesseract is NOT used — VLM is more accurate and the rendering step
    is the real bottleneck, not the API calls.
    """
    # Step A — render scanned pages in batches
    page_indices = [item["page_index"] for item in scanned_queue]
    print(f"[extract]   Rendering {len(page_indices)} scanned pages...")
    t_render = time.time()
    rendered = _render_pages_batch(pdf_path, page_indices)
    print(f"[extract]   Rendered {len(rendered)} pages in {time.time()-t_render:.1f}s")

    # Step B — async VLM calls (high concurrency)
    out: dict[int, tuple[str, list]] = {}

    # Mark blank pages (skipped during render)
    for item in scanned_queue:
        pi = item["page_index"]
        if pi not in rendered:
            out[pi] = ("", [])

    vlm_items = [item for item in scanned_queue if item["page_index"] in rendered]

    # Quick Tesseract triage: decide which pages need full table extraction
    # vs lightweight OCR-only. This saves ~60% of VLM response tokens.
    t_triage = time.time()
    triage_texts = _tesseract_ocr_batch(rendered)
    table_pages = set()
    for pi, text in triage_texts.items():
        if _page_needs_vlm(text):
            table_pages.add(pi)
    ocr_only_count = len(vlm_items) - len(table_pages)
    print(f"[extract]   Triage ({time.time()-t_triage:.1f}s): "
          f"{len(table_pages)} pages need tables, "
          f"{ocr_only_count} OCR-only (lightweight)", flush=True)

    if vlm_items:
        semaphore = asyncio.Semaphore(max_concurrent)
        progress = {"done": 0, "total": len(vlm_items), "t0": time.time()}

        if _USE_OPENAI:
            async_client = _AsyncOpenAI()
            tasks = [
                _vlm_one_page_async(
                    semaphore, async_client,
                    rendered[item["page_index"]],
                    item["page_index"], item["pw"], item["ph"],
                    progress,
                    ocr_only=(item["page_index"] not in table_pages),
                )
                for item in vlm_items
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            await async_client.close()
        else:
            import anthropic as _ant
            async with _ant.AsyncAnthropic(api_key=api_key) as async_client:
                tasks = [
                    _vlm_one_page_async(
                        semaphore, async_client,
                        rendered[item["page_index"]],
                        item["page_index"], item["pw"], item["ph"],
                        progress,
                        ocr_only=(item["page_index"] not in table_pages),
                    )
                    for item in vlm_items
                ]
                results = await asyncio.gather(*tasks, return_exceptions=True)

        for r in results:
            if isinstance(r, Exception):
                print(f"[WARN] gather exception: {r}")
                continue
            pg_idx, text, frags = r
            out[pg_idx] = (text, frags)

    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def extract_pdf(
    pdf_path: str,
    anthropic_api_key: Optional[str] = None,
    max_workers: int = DEFAULT_WORKERS,
    use_vlm: bool = True,
) -> dict:
    """
    Extract all pages from a multi-page PDF.

    Pass 1 (serial, fast): pdfplumber over all pages.
    Pass 2 (parallel):     Claude Haiku VLM for scanned pages, up to max_workers concurrent.

    Returns a dict mirroring the labels.json schema.
    """
    pdf_path   = str(Path(pdf_path).resolve())
    package_id = Path(pdf_path).parent.name

    # ------------------------------------------------------------------ #
    # PASS 1 — pdfplumber: classify every page, extract digital fragments #
    # ------------------------------------------------------------------ #
    t0 = time.time()
    pages_out: list[dict] = []          # one per page, indexed by page_index
    tables_out: list[dict] = []         # fragments from digital pages
    scanned_queue: list[dict] = []      # metadata for scanned pages to process in Pass 2

    print(f"[extract] Pass 1 — pdfplumber  ({pdf_path})")
    with pdfplumber.open(pdf_path) as pdf:
        total = len(pdf.pages)
        for page_index, page in enumerate(pdf.pages):
            native = _has_text_layer(page)

            if native:
                text  = page.extract_text() or ""
                frags = _extract_native_fragments(page, page_index)
                scan_transform = IDENTITY_TRANSFORM
            else:
                text  = ""
                frags = []
                scan_transform = None
                scanned_queue.append({
                    "page_index": page_index,
                    "pw": page.width,
                    "ph": page.height,
                })

            pages_out.append({
                # labels.json page fields
                "page_index":           page_index,
                "doc_type":             None,   # [CLASSIFY]
                "doc_type_label_id":    None,   # [CLASSIFY]
                "doc_instance_id":      None,   # [SEGMENT]
                "section":              None,   # [SEGMENT]
                "is_first_page_of_doc": None,   # [SEGMENT]
                "is_last_page_of_doc":  None,   # [SEGMENT]
                "page_in_doc":          None,   # [SEGMENT]
                "total_pages_in_doc":   None,   # [SEGMENT]
                "boundary":             None,   # [SEGMENT]
                "width":                page.width,
                "height":               page.height,
                "has_table":            len(frags) > 0,
                "table_ids":            [],     # [STITCH]
                "has_chart":            False,
                "chart_ids":            [],
                "render_mode":          "digital" if native else "scanned",
                "scan_transform":       scan_transform,
                "rotation":             0,
                "scan_image_size_px":   [round(page.width * PT_TO_PX), round(page.height * PT_TO_PX)],
                # extras
                "text":         text,
                "fragment_ids": [f["fragment_id"] for f in frags],
            })
            tables_out.extend(frags)

    digital_count = total - len(scanned_queue)
    t1 = time.time()
    print(
        f"[extract] Pass 1 done in {t1-t0:.1f}s — "
        f"digital={digital_count}, scanned={len(scanned_queue)}, "
        f"fragments={len(tables_out)}"
    )

    # ------------------------------------------------------------------ #
    # PASS 2 — async parallel VLM for scanned pages                     #
    # ------------------------------------------------------------------ #
    has_vlm_key = (anthropic_api_key or _USE_OPENAI) and use_vlm
    if scanned_queue and has_vlm_key:
        n_scanned         = len(scanned_queue)
        effective_workers = min(max_workers, n_scanned)
        backend_name = "OpenAI GPT-4o-mini" if _USE_OPENAI else "Anthropic Haiku"
        print(
            f"[extract] Pass 2 — AsyncVLM ({backend_name})  {n_scanned} pages  "
            f"concurrency={effective_workers}  model={VLM_MODEL}"
        )

        vlm_results = asyncio.run(
            _run_vlm_pass(scanned_queue, pdf_path, anthropic_api_key, effective_workers)
        )

        # Merge VLM results back into pages_out (in page_index order)
        for pg_idx, (pg_text, frags) in vlm_results.items():
            p = pages_out[pg_idx]
            p["text"]         = pg_text
            p["has_table"]    = len(frags) > 0
            p["fragment_ids"] = [f["fragment_id"] for f in frags]
            tables_out.extend(frags)

        t2        = time.time()
        vlm_frags = sum(len(r[1]) for r in vlm_results.values())
        vlm_ok    = sum(1 for r in vlm_results.values() if r[0] or r[1])
        print(f"[extract] Pass 2 done in {t2-t1:.1f}s — "
              f"{vlm_frags} fragments from {vlm_ok}/{n_scanned} pages")

    elif scanned_queue and not has_vlm_key:
        if not use_vlm:
            # Tesseract-only mode: run OCR triage directly on all scanned pages
            print(f"[extract] Pass 2 — Tesseract-only mode ({len(scanned_queue)} scanned pages)")
            t1 = time.time()
            rendered = asyncio.run(_render_pages_batch(scanned_queue, pdf_path))
            ocr_texts = _tesseract_ocr_batch(rendered)
            for item in scanned_queue:
                pi = item["page_index"]
                text = ocr_texts.get(pi, "")
                pages_out[pi]["text"] = text
            print(f"[extract] Tesseract done in {time.time()-t1:.1f}s")
        else:
            print(f"[extract] Pass 2 skipped — no API key ({len(scanned_queue)} scanned pages unprocessed)")

    # ------------------------------------------------------------------ #
    # Assemble output                                                     #
    # ------------------------------------------------------------------ #
    digital_c = sum(1 for p in pages_out if p["render_mode"] == "digital")
    scanned_c = total - digital_c
    pkg_mode  = "digital" if scanned_c == 0 else ("scanned" if digital_c == 0 else "mixed")

    total_frags = len(tables_out)
    total_cells = sum(len(f.get("cells", [])) for f in tables_out)
    total_time  = time.time() - t0

    print(
        f"[extract] Complete — {total} pages in {total_time:.1f}s  "
        f"({total/total_time:.1f} pages/s)\n"
        f"          fragments={total_frags}  cells={total_cells}  "
        f"digital={digital_c}  scanned={scanned_c}"
    )

    return {
        "schema_version": SCHEMA_VERSION,
        "package_id":     package_id,
        "total_pages":    total,
        "coord_system":   COORD_SYSTEM,
        "documents":      [],           # [SEGMENT]
        "pages":          pages_out,
        "tables":         tables_out,
        "charts":         [],
        "render_mode":    pkg_mode,
    }
