"""
segment.py — Document Instance Segmentation (Page Stream Segmentation)

Academic framing: PSS (Page Stream Segmentation) — pairwise boundary detection.
  For each adjacent page pair (N, N+1): is there a document boundary between them?
  Ref: arXiv 2408.11981 (LLMs for PSS), arXiv 2602.15958 (DocSplit)

Input:  output of classify.py — list of per-page classifications
Output: list of document instances matching labels.json documents[] format

Key reframe: this is COREFERENCE not segmentation.
  Ask "do page N and N+1 belong to the same document instance?"
  rather than "where does this document end?"

Run standalone self-test:
  python3 src/segment.py
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

# ── Known fixed page lengths (heuristic for splitting back-to-back instances) ──
KNOWN_PAGE_LENGTHS: dict[str, int | str] = {
    "w2":                1,
    "paystub":           1,
    "form_1040":         2,
    "closing_disclosure":"variable",   # 3–5 pages
    "loan_estimate":     "variable",   # ~3 pages
    "bank_stmt_checking":"variable",   # 3–14 pages
    "bank_stmt_combo":   "variable",
}

# ── Distinguishing attribute regex patterns ────────────────────────────────────
# Used to detect when two same-type pages are actually different instances.
# e.g. bank stmt Feb 2024 vs Mar 2024 — same doc_type, different instance.
_ATTR_PATTERNS: dict[str, list[re.Pattern]] = {
    "bank_stmt_checking": [
        re.compile(r"statement\s+(?:period|date)[:\s]+(\w+\s+\d{4}|\d{4}-\d{2})", re.I),
        re.compile(r"account\s+(?:number|#|no\.?)[:\s#]+([X*\d\-]{4,})", re.I),
        re.compile(r"for\s+the\s+period\s+(\w+\s+\d+,?\s*\d{4})", re.I),
        # "Oct 01 – Oct 31, 2024" or "10/01/2024 – 10/31/2024" style
        re.compile(r"(\w{3,9}\s+\d{1,2}\s*[–\-]\s*\w{3,9}\s+\d{1,2},?\s*\d{4})", re.I),
        re.compile(r"(\d{1,2}/\d{1,2}/\d{4})\s*[–\-—]\s*\d{1,2}/\d{1,2}/\d{4}", re.I),
        # "through MM/DD/YYYY" or "as of MM/DD/YYYY"
        re.compile(r"(?:through|as\s+of|thru)\s+(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})", re.I),
        # Bank name on header page — Chase vs Wells Fargo vs CapitalOne etc.
        re.compile(r"^(chase|wells\s+fargo|bank\s+of\s+america|citibank|capital\s+one|us\s+bank|pnc|td\s+bank|truist|regions|suntrust|bb&t|citizens|fifth\s+third|keybank|huntington)", re.I),
    ],
    "bank_stmt_combo": [
        re.compile(r"statement\s+(?:period|date)[:\s]+(\w+\s+\d{4}|\d{4}-\d{2})", re.I),
        re.compile(r"account\s+(?:number|#|no\.?)[:\s#]+([X*\d\-]{4,})", re.I),
    ],
    "paystub": [
        re.compile(r"pay\s+(?:period|date)\s+end[:\s]+(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})", re.I),
        re.compile(r"check\s+date[:\s]+(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})", re.I),
        re.compile(r"period\s+ending[:\s]+(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})", re.I),
    ],
    "form_1040": [
        re.compile(r"(?:tax\s+year|year)\s+(\d{4})", re.I),
        re.compile(r"u\.?s\.?\s+individual.*?(\d{4})", re.I),
        re.compile(r"for\s+the\s+year\s+jan\.\s+1.*?(\d{4})", re.I),
    ],
    "w2": [
        re.compile(r"(?:tax\s+year|wages.*?|year)\s+(\d{4})", re.I),
        re.compile(r"employee.*?(\d{4})\s+w-?2", re.I),
    ],
    "schedule_1": [
        re.compile(r"(\d{4})", re.I),
    ],
    "schedule_c": [
        re.compile(r"(\d{4})", re.I),
    ],
}


# ── Data structure ─────────────────────────────────────────────────────────────

@dataclass
class DocInstance:
    """One logical document instance — matches labels.json documents[] format."""
    doc_instance_id:   str
    doc_type:          str
    doc_type_label_id: int
    start_page:        int
    end_page:          int
    page_count:        int
    instance_ordinal:  int   # 1-indexed within this doc_type in the file
    distinguishing_attr: Optional[str] = None


# ── Distinguishing attribute extraction ───────────────────────────────────────

def _extract_distinguishing_attr(doc_type: str, page_text: str) -> Optional[str]:
    """
    Try to extract the attribute that makes this page's instance unique
    (statement date, tax year, pay period end, account number).

    Returns a string like "2024-02", "2023", "2024-01-15", or None if not found.
    """
    patterns = _ATTR_PATTERNS.get(doc_type, [])
    for pat in patterns:
        m = pat.search(page_text)
        if m:
            return m.group(1).strip()
    return None


# ── Additional boundary signals ───────────────────────────────────────────────

_ENDING_BAL_RE   = re.compile(r"(?:ending|closing|final)\s+balance[\s:$]*([0-9,]+\.\d{2})", re.I)
_BEGINNING_BAL_RE = re.compile(r"(?:beginning|opening|starting)\s+balance[\s:$]*([0-9,]+\.\d{2})", re.I)

def _detect_balance_break(text_a: str, text_b: str) -> bool:
    """
    Returns True if the ending balance on page A does NOT match the beginning
    balance on page B. This is the strongest signal that two bank statement
    pages belong to different accounts — a valid continuation always has:
        ending_balance(page N) == beginning_balance(page N+1)

    Only fires when BOTH values are extractable (not None) and they differ.
    A None result (not found) is treated as no signal — not a boundary.
    """
    m_end = _ENDING_BAL_RE.search(text_a)
    m_beg = _BEGINNING_BAL_RE.search(text_b)
    if not m_end or not m_beg:
        return False
    # Normalise: strip commas, compare as strings
    end_val = m_end.group(1).replace(",", "")
    beg_val = m_beg.group(1).replace(",", "")
    return end_val != beg_val


_INSTITUTION_RE = re.compile(
    r"^(chase|wells\s+fargo|bank\s+of\s+america|citibank|capital\s+one"
    r"|us\s+bank|pnc|td\s+bank|truist|regions|suntrust|bb&t|citizens"
    r"|fifth\s+third|keybank|huntington|ally|discover|navy\s+federal"
    r"|usaa|charles\s+schwab|fidelity|vanguard)",
    re.I | re.MULTILINE,
)

def _detect_new_doc_header(doc_type: str, text_a: str, text_b: str) -> bool:
    """
    Returns True if page B appears to start a NEW bank statement from a
    DIFFERENT institution than page A.

    Logic: extract the institution name from the first ~200 chars of each page.
    If both are found and differ → new document.
    """
    if doc_type not in ("bank_stmt_checking", "bank_stmt_combo", "brokerage_stmt"):
        return False
    m_a = _INSTITUTION_RE.search(text_a[:300])
    m_b = _INSTITUTION_RE.search(text_b[:300])
    if not m_a or not m_b:
        return False
    name_a = m_a.group(1).lower().replace(" ", "")
    name_b = m_b.group(1).lower().replace(" ", "")
    return name_a != name_b


# ── Boundary feature extraction ───────────────────────────────────────────────

def _extract_boundary_features(
    page_a: dict,
    page_b: dict,
    text_a: str = "",
    text_b: str = "",
) -> dict:
    """
    Compute pairwise features for adjacent pages A and B.
    Used to decide: is there a document boundary between A and B?

    Args:
        page_a: classify.py output dict for page N
        page_b: classify.py output dict for page N+1
        text_a: raw page text for page N (optional, improves attr extraction)
        text_b: raw page text for page N+1 (optional)

    Returns dict of features (all float 0–1 or bool):
      - doc_type_changed:    bool — different doc_type → always a boundary
      - known_length_hit:    bool — page count reached known fixed length → new instance
      - attr_a:              str|None — distinguishing attr extracted from A
      - attr_b:              str|None — distinguishing attr extracted from B
      - attr_changed:        bool — both extracted and they differ → boundary
      - confidence_reset:    bool — page B is a high-confidence "first page" (llm/heuristic, not carry_forward)
    """
    type_a = page_a.get("doc_type")
    type_b = page_b.get("doc_type")

    doc_type_changed = (type_a != type_b)

    attr_a = _extract_distinguishing_attr(type_a or "", text_a) if text_a else None
    attr_b = _extract_distinguishing_attr(type_b or "", text_b) if text_b else None

    attr_changed = bool(attr_a and attr_b and attr_a != attr_b)

    # A page classified by llm/heuristic (not carry_forward) is a candidate first page
    method_b = page_b.get("method", "")
    confidence_reset = method_b in ("llm", "heuristic") and page_b.get("confidence", 0) >= 0.70

    # Balance-break signal: ending balance of page A ≠ beginning balance of page B
    # When two bank statements from different banks/accounts are concatenated, the
    # ending balance of statement 1 will NOT match the beginning balance of statement 2.
    balance_break = _detect_balance_break(text_a, text_b) if text_a and text_b else False

    # New-document header signal: page B looks like the first page of a new document
    # (has a bank name / institution header that differs from page A's institution)
    new_doc_header = _detect_new_doc_header(type_b or "", text_a, text_b)

    return {
        "doc_type_changed":  doc_type_changed,
        "attr_a":            attr_a,
        "attr_b":            attr_b,
        "attr_changed":      attr_changed,
        "confidence_reset":  confidence_reset,
        "balance_break":     balance_break,
        "new_doc_header":    new_doc_header,
    }


def _is_boundary(features: dict, current_span_len: int, doc_type: str) -> bool:
    """
    Given boundary features, decide: new document instance starts at page B?

    Rules (in order of priority):
      1. doc_type changed → always a boundary
      2. distinguishing_attr changed → boundary (new month, new year, new account number)
      3. known fixed length reached (w2=1, paystub=1, form_1040=2) → boundary
      4. balance_break — ending balance ≠ beginning balance → different accounts
      5. new_doc_header — different bank institution name detected on page B
      (confidence_reset alone is NOT sufficient — carry_forward gaps happen legitimately)
    """
    # Rule 1: type change always splits
    if features["doc_type_changed"]:
        return True

    # Rule 2: distinguishing attribute changed (account number, statement period, tax year)
    if features["attr_changed"]:
        return True

    # Rule 3: fixed-length doc type exhausted its expected page count
    expected = KNOWN_PAGE_LENGTHS.get(doc_type)
    if isinstance(expected, int) and current_span_len >= expected:
        return True

    # Rule 4: balance break — strongest signal for same-type different-account boundaries.
    # ending_balance(A) ≠ beginning_balance(B) means these are different accounts.
    if features.get("balance_break"):
        return True

    # Rule 5: different institution name found on page B header
    if features.get("new_doc_header"):
        return True

    return False


# ── Main segmentation function ────────────────────────────────────────────────

def segment_documents(
    classifications: list[dict],
    page_texts: Optional[list[str]] = None,
) -> list[DocInstance]:
    """
    Convert per-page classifications into document instances.

    Args:
        classifications: output of classify_pages() — list of dicts with
                         {page_index, doc_type, doc_type_label_id, confidence, method}
                         Must be sorted by page_index.
        page_texts:      optional list of raw page texts (same order as classifications).
                         Used to extract distinguishing_attr (statement dates, tax years).
                         Pass None to skip attribute extraction.

    Returns:
        list of DocInstance objects, in page order.
        Each maps to one row in labels.json documents[] array.
    """
    if not classifications:
        return []

    # Sort by page_index just in case
    pages = sorted(classifications, key=lambda p: p["page_index"])

    if page_texts is None:
        texts = [""] * len(pages)
    else:
        texts = list(page_texts)

    instances: list[DocInstance] = []
    ordinal_counter: dict[str, int] = {}  # doc_type → count so far

    # Start first span
    span_start = 0
    span_type  = pages[0]["doc_type"]
    span_lid   = pages[0]["doc_type_label_id"]
    span_len   = 1
    span_attrs: list[str] = []
    if texts[0]:
        a = _extract_distinguishing_attr(span_type, texts[0])
        if a:
            span_attrs.append(a)

    def _close_span(end_idx: int):
        """Finalize the current span and append a DocInstance."""
        ordinal_counter[span_type] = ordinal_counter.get(span_type, 0) + 1
        ord_n = ordinal_counter[span_type]
        attr = span_attrs[0] if span_attrs else None
        instances.append(DocInstance(
            doc_instance_id    = f"{span_type}#{ord_n}",
            doc_type           = span_type,
            doc_type_label_id  = span_lid,
            start_page         = pages[span_start]["page_index"],
            end_page           = pages[end_idx]["page_index"],
            page_count         = end_idx - span_start + 1,
            instance_ordinal   = ord_n,
            distinguishing_attr= attr,
        ))

    for i in range(1, len(pages)):
        feat = _extract_boundary_features(
            pages[i - 1], pages[i],
            text_a=texts[i - 1],
            text_b=texts[i],
        )

        if _is_boundary(feat, span_len, span_type):
            _close_span(i - 1)
            # Start new span
            span_start = i
            span_type  = pages[i]["doc_type"]
            span_lid   = pages[i]["doc_type_label_id"]
            span_len   = 1
            span_attrs = []
            if texts[i]:
                a = _extract_distinguishing_attr(span_type, texts[i])
                if a:
                    span_attrs.append(a)
        else:
            span_len += 1
            if texts[i]:
                a = _extract_distinguishing_attr(span_type, texts[i])
                if a and (not span_attrs or a != span_attrs[-1]):
                    span_attrs.append(a)
                    # If we just discovered two different attrs in the same span, force split
                    if len(span_attrs) >= 2 and span_attrs[0] != span_attrs[-1]:
                        # Retroactively split: close span at i-1, restart at i
                        _close_span(i - 1)
                        span_start = i
                        span_len   = 1
                        span_attrs = [a]

    # Close final span
    _close_span(len(pages) - 1)

    return instances


def merge_coreference_instances(instances: list[DocInstance]) -> list[DocInstance]:
    """
    Second-pass global coreference merge.

    Problem: segment_documents() is a single left-to-right pass — it can only
    merge ADJACENT pages. If page 1 and page 2000 belong to the same bank
    statement (same account number, same period) but are separated by 1998
    other pages, they come out as two separate instances.

    Fix: after the first pass, group instances by (doc_type, distinguishing_attr).
    Any two instances with the same key → same logical document → merge them.

    Example:
        bank_stmt_checking#1  p1–p3    [****1234, Feb 2024]   ← fragment 1
        bank_stmt_checking#5  p1998–p2000 [****1234, Feb 2024] ← fragment 2
        → merged into:
        bank_stmt_checking#1  p1–p2000  page_count=6  [****1234, Feb 2024]

    Only merges when BOTH instances have a distinguishing_attr (account number
    or statement period). Instances with no attr are left as-is to avoid
    false merges on generic types like filler.
    """
    if not instances:
        return instances

    # Separate instances that have a usable attr from those that don't
    keyed:   dict[tuple, list[DocInstance]] = {}
    no_attr: list[DocInstance] = []

    for inst in instances:
        if inst.distinguishing_attr:
            key = (inst.doc_type, inst.distinguishing_attr.lower().strip())
            keyed.setdefault(key, []).append(inst)
        else:
            no_attr.append(inst)

    merged: list[DocInstance] = []

    for (doc_type, _attr), group in keyed.items():
        if len(group) == 1:
            merged.append(group[0])
        else:
            # Merge: span from earliest start_page to latest end_page
            # page_count = sum of all fragment page counts (we know these pages belong here)
            merged.append(DocInstance(
                doc_instance_id    = group[0].doc_instance_id,   # renumbered below
                doc_type           = doc_type,
                doc_type_label_id  = group[0].doc_type_label_id,
                start_page         = min(i.start_page for i in group),
                end_page           = max(i.end_page   for i in group),
                page_count         = sum(i.page_count  for i in group),
                instance_ordinal   = group[0].instance_ordinal,  # renumbered below
                distinguishing_attr= group[0].distinguishing_attr,
            ))

    merged.extend(no_attr)

    # Re-sort by start_page and re-number ordinals cleanly
    merged.sort(key=lambda i: i.start_page)
    ordinal_counter: dict[str, int] = {}
    result: list[DocInstance] = []
    for inst in merged:
        ordinal_counter[inst.doc_type] = ordinal_counter.get(inst.doc_type, 0) + 1
        ord_n = ordinal_counter[inst.doc_type]
        result.append(DocInstance(
            doc_instance_id    = f"{inst.doc_type}#{ord_n}",
            doc_type           = inst.doc_type,
            doc_type_label_id  = inst.doc_type_label_id,
            start_page         = inst.start_page,
            end_page           = inst.end_page,
            page_count         = inst.page_count,
            instance_ordinal   = ord_n,
            distinguishing_attr= inst.distinguishing_attr,
        ))

    return result


def instances_to_dict(instances: list[DocInstance]) -> list[dict]:
    """Convert DocInstance list to JSON-serialisable dicts."""
    result = []
    for inst in instances:
        d = {
            "doc_instance_id":    inst.doc_instance_id,
            "doc_type":           inst.doc_type,
            "doc_type_label_id":  inst.doc_type_label_id,
            "start_page":         inst.start_page,
            "end_page":           inst.end_page,
            "page_count":         inst.page_count,
            "instance_ordinal":   inst.instance_ordinal,
        }
        if inst.distinguishing_attr is not None:
            d["distinguishing_attr"] = inst.distinguishing_attr
        result.append(d)
    return result


# ── Evaluation helper ─────────────────────────────────────────────────────────

def score_against_ground_truth(
    predicted: list[DocInstance],
    ground_truth: list[dict],  # labels.json documents[] format
) -> dict:
    """
    Measure boundary F1 against ground truth.
    Uses DocSplit error taxonomy: split_groups, merged_groups, correct.

    Returns: {precision, recall, f1, split_errors, merge_errors, exact_matches}
    """
    # Build sets of (start_page, end_page, doc_type) for comparison
    pred_set  = {(i.start_page, i.end_page, i.doc_type) for i in predicted}
    truth_set = {(d["start_page"], d["end_page"], d["doc_type"]) for d in ground_truth}

    correct = len(pred_set & truth_set)
    precision = correct / len(pred_set)  if pred_set  else 0.0
    recall    = correct / len(truth_set) if truth_set else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    return {
        "precision":     round(precision, 3),
        "recall":        round(recall, 3),
        "f1":            round(f1, 3),
        "exact_matches": correct,
        "predicted":     len(pred_set),
        "ground_truth":  len(truth_set),
    }


# ── Self-test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json

    # Use the demo_schema INPUT directly
    demo_input = [
        {"page_index": 0,  "doc_type": "bank_stmt_checking", "doc_type_label_id": 10, "confidence": 0.94, "method": "llm"},
        {"page_index": 1,  "doc_type": "bank_stmt_checking", "doc_type_label_id": 10, "confidence": 0.50, "method": "carry_forward"},
        {"page_index": 2,  "doc_type": "bank_stmt_checking", "doc_type_label_id": 10, "confidence": 0.50, "method": "carry_forward"},
        {"page_index": 3,  "doc_type": "bank_stmt_checking", "doc_type_label_id": 10, "confidence": 0.50, "method": "carry_forward"},
        {"page_index": 4,  "doc_type": "bank_stmt_checking", "doc_type_label_id": 10, "confidence": 0.50, "method": "carry_forward"},
        {"page_index": 5,  "doc_type": "bank_stmt_checking", "doc_type_label_id": 10, "confidence": 0.50, "method": "carry_forward"},
        {"page_index": 6,  "doc_type": "bank_stmt_checking", "doc_type_label_id": 10, "confidence": 0.91, "method": "llm"},
        {"page_index": 7,  "doc_type": "bank_stmt_checking", "doc_type_label_id": 10, "confidence": 0.50, "method": "carry_forward"},
        {"page_index": 8,  "doc_type": "bank_stmt_checking", "doc_type_label_id": 10, "confidence": 0.50, "method": "carry_forward"},
        {"page_index": 9,  "doc_type": "paystub",            "doc_type_label_id": 4,  "confidence": 0.87, "method": "heuristic"},
        {"page_index": 10, "doc_type": "paystub",            "doc_type_label_id": 4,  "confidence": 0.87, "method": "heuristic"},
        {"page_index": 11, "doc_type": "paystub",            "doc_type_label_id": 4,  "confidence": 0.87, "method": "heuristic"},
        {"page_index": 12, "doc_type": "w2",                 "doc_type_label_id": 5,  "confidence": 0.80, "method": "heuristic"},
        {"page_index": 13, "doc_type": "form_1040",          "doc_type_label_id": 7,  "confidence": 0.80, "method": "heuristic"},
        {"page_index": 14, "doc_type": "form_1040",          "doc_type_label_id": 7,  "confidence": 0.50, "method": "carry_forward"},
        {"page_index": 15, "doc_type": "form_1040",          "doc_type_label_id": 7,  "confidence": 0.78, "method": "llm"},
        {"page_index": 16, "doc_type": "form_1040",          "doc_type_label_id": 7,  "confidence": 0.50, "method": "carry_forward"},
    ]

    # Fake page texts — include statement dates so bank stmts split correctly
    demo_texts = [
        "Statement Period: February 2024  Account: ****4521",   # page 0
        "Date  Description  Amount  Balance",                   # page 1 continuation
        "02/02 ACH DEPOSIT  1200.00  45320.00",                 # page 2
        "02/05 DEBIT CARD TARGET  89.00  45231.00",             # page 3
        "02/10 CHECK 1042  500.00  44731.00",                   # page 4
        "02/28 CLOSING BALANCE  44731.00",                      # page 5
        "Statement Period: March 2024  Account: ****4521",      # page 6 — NEW INSTANCE
        "Date  Description  Amount  Balance",                   # page 7
        "03/31 CLOSING BALANCE  43100.00",                      # page 8
        "Pay Period End: 01/15/2024  Employee: John Smith",     # page 9
        "Pay Period End: 01/31/2024  Employee: John Smith",     # page 10
        "Pay Period End: 02/15/2024  Employee: John Smith",     # page 11
        "W-2 Wage and Tax Statement  Tax Year 2023",            # page 12
        "Form 1040 U.S. Individual Income Tax Return  2022",    # page 13
        "Schedule 1  Additional Income  2022",                  # page 14
        "Form 1040 U.S. Individual Income Tax Return  2023",    # page 15 — NEW INSTANCE
        "Schedule 1  Additional Income  2023",                  # page 16
    ]

    expected = [
        ("bank_stmt_checking", 0, 5),   # Feb 2024
        ("bank_stmt_checking", 6, 8),   # Mar 2024
        ("paystub",            9, 9),
        ("paystub",           10, 10),
        ("paystub",           11, 11),
        ("w2",                12, 12),
        ("form_1040",         13, 14),  # 2022
        ("form_1040",         15, 16),  # 2023
    ]

    print("=== segment.py self-test ===\n")
    results = segment_documents(demo_input, demo_texts)

    print(f"{'ID':<25} {'start':>5} {'end':>5}  {'attr':<12}  {'status'}")
    print("-" * 65)

    got = [(r.doc_type, r.start_page, r.end_page) for r in results]
    for inst in results:
        match = (inst.doc_type, inst.start_page, inst.end_page) in expected
        status = "PASS" if match else "FAIL"
        print(f"{inst.doc_instance_id:<25} {inst.start_page:>5} {inst.end_page:>5}  {str(inst.distinguishing_attr):<12}  {status}")

    correct = sum(1 for e in expected if e in got)
    print(f"\nScore: {correct}/{len(expected)} instances correct")

    missing = [e for e in expected if e not in got]
    extra   = [g for g in got   if g not in expected]
    if missing:
        print(f"Missing: {missing}")
    if extra:
        print(f"Extra:   {extra}")

    # ── Hard case test: same doc_type, same period, DIFFERENT banks ──────────────
    print("\n=== Hard case: Chase + Wells Fargo, same month, same doc_type ===\n")

    hard_input = [
        {"page_index": 0, "doc_type": "bank_stmt_checking", "doc_type_label_id": 10, "confidence": 0.94, "method": "llm"},
        {"page_index": 1, "doc_type": "bank_stmt_checking", "doc_type_label_id": 10, "confidence": 0.50, "method": "carry_forward"},
        {"page_index": 2, "doc_type": "bank_stmt_checking", "doc_type_label_id": 10, "confidence": 0.50, "method": "carry_forward"},
        {"page_index": 3, "doc_type": "bank_stmt_checking", "doc_type_label_id": 10, "confidence": 0.94, "method": "llm"},
        {"page_index": 4, "doc_type": "bank_stmt_checking", "doc_type_label_id": 10, "confidence": 0.50, "method": "carry_forward"},
    ]
    hard_texts = [
        # Chase, Feb 2024, account ****1234
        "Chase\nStatement Period: February 2024\nAccount Number: ****1234\nBeginning Balance: $12,450.00",
        "Date Description Amount Balance\n02/05 DEBIT TARGET 89.00 12361.00",
        "Ending Balance: 11,200.00\nThank you for banking with Chase",
        # Wells Fargo, Feb 2024, account ****5678 — same period, different bank
        "Wells Fargo\nStatement Period: February 2024\nAccount Number: ****5678\nBeginning Balance: $8,340.00",
        "Date Description Amount Balance\n02/10 DEBIT WALMART 55.00 8285.00",
    ]
    hard_expected = [
        ("bank_stmt_checking", 0, 2),   # Chase Feb 2024
        ("bank_stmt_checking", 3, 4),   # Wells Fargo Feb 2024
    ]

    hard_results = segment_documents(hard_input, hard_texts)
    hard_got = [(r.doc_type, r.start_page, r.end_page) for r in hard_results]

    print(f"{'ID':<25} {'start':>5} {'end':>5}  {'attr':<15}  {'status'}")
    print("-" * 70)
    for inst in hard_results:
        match = (inst.doc_type, inst.start_page, inst.end_page) in hard_expected
        status = "PASS" if match else "FAIL"
        print(f"{inst.doc_instance_id:<25} {inst.start_page:>5} {inst.end_page:>5}  {str(inst.distinguishing_attr):<15}  {status}")

    hard_correct = sum(1 for e in hard_expected if e in hard_got)
    print(f"\nScore: {hard_correct}/{len(hard_expected)} hard-case instances correct")
