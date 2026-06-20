"""
classify.py — Document-type classification per page.

Strategy:
  1. Keyword/regex heuristic (free, instant) — high-confidence shortcut only
  2. Carry-forward for table-continuation pages (no LLM needed)
  3. Claude Haiku for everything else — generalises to ANY PDF, parallel async

Why LLM-first for ambiguous cases:
  Keywords break on unseen bank templates, foreign lender formats, etc.
  Haiku is fast (~0.3s/call), cheap (~$0.0003/page), and reads the actual text.
  Parallel async means 50 pages take the same wall time as 1.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import time
from anthropic import Anthropic, AsyncAnthropic

# ── 0. Label ID map — matches labels.json doctype_label_id exactly ───────────

LABEL_ID = {
    "urla_1003":             0,
    "form_1008":             1,
    "loan_estimate":         2,
    "closing_disclosure":    3,
    "paystub":               4,
    "w2":                    5,
    "voe":                   6,
    "form_1040":             7,
    "schedule_1":            8,
    "schedule_c":            9,
    "bank_stmt_checking":    10,
    "bank_stmt_combo":       11,
    "brokerage_stmt":        12,
    "check_image":           13,
    "deposit_receipt":       14,
    "credit_report":         15,
    "du_findings":           16,
    "lpa_feedback":          17,
    "purchase_contract":     18,
    "purchase_addendum":     19,
    "options_addendum":      20,
    "email_correspondence":  21,
    "letter_of_explanation": 22,
    "gift_letter":           23,
    "insurance_declaration": 24,
    "loan_summary":          25,
    "filler":                26,
    "unknown":               -1,
}

KNOWN_TYPES = [k for k in LABEL_ID if k != "unknown"]

# ── 1. Keyword signals — HIGH-confidence shortcuts only ───────────────────────
#    These only fire when we're very sure. Everything else goes to Haiku.

DOC_SIGNALS = {
    # ── Tax / income ──────────────────────────────────────────────────────────
    "w2": [
        r"\bw-?2\b",
        r"wage and tax statement",
        r"employee'?s? social security",
        r"box 1\b.*wages",
        r"employer identification number",
    ],
    "form_1040": [
        r"\b1040\b",
        r"u\.?s\.? individual income tax return",
        r"adjusted gross income",
        r"filing status",
        r"standard deduction",
    ],
    "schedule_1": [
        r"schedule 1\b",
        r"additional income and adjustments",
        r"other income\b",
        r"alimony received",
    ],
    "schedule_c": [
        r"schedule c\b",
        r"profit or loss from business",
        r"sole proprietor",
        r"principal business",
    ],
    "paystub": [
        r"pay stub|paystub|pay slip",
        r"earnings statement",
        r"year.?to.?date",
        r"gross pay|net pay",
        r"pay period",
        r"employee id",
    ],
    "voe": [
        r"verification of employment",
        r"\bvoe\b",
        r"date of hire",
        r"current salary|current wage",
    ],
    # ── Bank / asset ──────────────────────────────────────────────────────────
    "bank_stmt_checking": [
        r"checking account|checking\b",
        r"bank statement|statement of account",
        r"beginning balance|ending balance|opening balance",
        r"total deposits|total withdrawals",
        r"daily account activity|account activity|transaction history",
        r"statement period",
        r"account (number|#).*\*{2,}",
    ],
    "bank_stmt_combo": [
        r"combined statement",
        r"savings.*checking|checking.*savings",
        r"multiple accounts",
        r"combined account summary",
        r"in all accounts",                 # "TOTAL ENDING BALANCE IN ALL ACCOUNTS"
        r"total ending balance",
        r"360\b",                           # Capital One 360
        r"savings account",
        r"money market",
    ],
    "brokerage_stmt": [
        r"brokerage (account|statement)",
        r"portfolio (summary|value)",
        r"shares?\b.*price\b",
        r"investment account",
        r"(vanguard|fidelity|schwab|merrill|etrade|robinhood)",   # common brokers
        r"monthly transaction statement",
        r"symbol\s+name\s+shares",                                # stock table header
        r"(ira|roth|401k|403b)\b",
    ],
    "deposit_receipt": [
        r"deposit receipt|deposit slip",
        r"teller receipt",
        r"amount deposited",
    ],
    "check_image": [
        r"pay to the order of",
        r"routing number.*account number",
        r"\bvoid\b.*check",
    ],
    # ── Loan documents ────────────────────────────────────────────────────────
    "closing_disclosure": [
        r"closing disclosure",
        r"closing cost details",
        r"cash to close",
        r"final\b.*loan terms",
    ],
    "loan_estimate": [
        r"loan estimate",
        r"save this loan estimate",
        r"before you close",
        r"good faith estimate",
    ],
    "loan_summary": [
        r"loan summary",
        r"loan overview",
        r"note rate|note amount",
    ],
    "urla_1003": [
        r"\b1003\b",
        r"uniform residential loan application",
        r"\burla\b",
        r"borrower information",
        r"property and loan information",
        r"lender loan no",                  # "Lender Loan No. 6061178222"
        r"application date\b",
        r"to be completed by the (lender|borrower)",
    ],
    "form_1008": [
        r"\b1008\b",
        r"underwriting transmittal summary",
        r"uniform underwriting",
    ],
    "du_findings": [
        r"desktop underwriter",
        r"approve[/\s]eligible",
        r"risk assessment.*fannie mae",
    ],
    "lpa_feedback": [
        r"loan product advisor",
        r"\blpa\b.*feedback|feedback.*\blpa\b",
        r"freddie mac",
        r"lp key\b",                        # "Freddie Mac® LP Key R850682055"
        r"aus casefile",
        r"accept\b.*risk class",
    ],
    # ── Property / purchase ───────────────────────────────────────────────────
    "purchase_contract": [
        r"purchase (and sale )?agreement",
        r"real estate purchase contract",
        r"earnest money",
        r"purchase price\b",
    ],
    "purchase_addendum": [
        r"addendum\b",
        r"amendment to (purchase|contract)",
        r"addendum to (purchase|contract)",
    ],
    "options_addendum": [
        r"options addendum",
        r"option(s)? to purchase",
        r"option fee\b",
        r"unrestricted right to terminate",
    ],
    "insurance_declaration": [
        r"declaration(s)? page",
        r"homeowner'?s? insurance",
        r"dwelling coverage",
        r"named insured",
    ],
    # ── Credit / correspondence ───────────────────────────────────────────────
    "credit_report": [
        r"credit report",
        r"credit score|fico score",
        r"equifax|experian|transunion",
        r"derogatory\b",
        r"tradeline",
    ],
    "letter_of_explanation": [
        r"letter of explanation",
        r"\bloe\b",
        r"to whom it may concern",
        r"i am writing to explain",
    ],
    "gift_letter": [
        r"gift letter",
        r"gift funds",
        r"no repayment (is )?required",
    ],
    "email_correspondence": [
        r"from:\s+\S+@\S+",
        r"to:\s+\S+@\S+",
        r"subject:\s+",
    ],
    # ── Other ─────────────────────────────────────────────────────────────────
    "filler": [
        r"this page (is )?intentionally left blank",
        r"intentionally blank",
        r"equal credit opportunity",
        r"fair lending disclosure",
        r"privacy (policy|notice)",
        r"important disclosures?",
        r"your rights under",               # common in legal notice filler pages
        r"adverse action notice",
        r"notice to (applicant|borrower)",
        r"right to receive",
    ],
}

# Only trust heuristic when it's very confident — everything else goes to Haiku
HEURISTIC_SHORTCUT_THRESHOLD = 0.6


# ── 2. Heuristic scorer ───────────────────────────────────────────────────────

def _score_page(text: str) -> dict[str, float]:
    text_lower = text.lower()
    scores = {}
    for doc_type, patterns in DOC_SIGNALS.items():
        hits = sum(1 for p in patterns if re.search(p, text_lower))
        scores[doc_type] = hits / len(patterns)
    return scores


def classify_heuristic(text: str) -> dict:
    scores = _score_page(text)
    best_type = max(scores, key=scores.get)
    best_score = scores[best_type]
    return {
        "doc_type": best_type if best_score > 0 else "unknown",
        "confidence": round(best_score, 3),
        "method": "heuristic",
        "all_scores": scores,
    }


# ── 3. LLM classification (sync + async) ─────────────────────────────────────

_LLM_MODEL = "claude-haiku-4-5-20251001"
_MAX_TEXT   = 600   # chars sent to LLM — enough for classification, cheap

_SYSTEM = (
    "You classify pages from US mortgage loan files. "
    "Reply with ONLY a JSON object — no markdown, no explanation."
)

_USER_TMPL = """\
Classify this page. Known types:
{types}

Page text:
---
{text}
---

JSON only: {{"doc_type": "<type or unknown>", "confidence": <0.0-1.0>}}"""


def _parse_llm_response(raw: str) -> dict:
    raw = raw.strip()
    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*?\}", raw, re.DOTALL)
        if not match:
            return {"doc_type": "unknown", "confidence": 0.0, "method": "llm_parse_error"}
        result = json.loads(match.group())
    result["method"] = "llm"
    result["doc_type_label_id"] = LABEL_ID.get(result.get("doc_type", "unknown"), -1)
    return result


def classify_llm_sync(text: str) -> dict:
    """Sync LLM call — use only for single-page classification."""
    client = Anthropic()
    msg = client.messages.create(
        model=_LLM_MODEL,
        max_tokens=64,
        system=_SYSTEM,
        messages=[{"role": "user", "content": _USER_TMPL.format(
            types=", ".join(KNOWN_TYPES),
            text=text[:_MAX_TEXT],
        )}],
    )
    return _parse_llm_response(msg.content[0].text)


async def _classify_llm_async(client: AsyncAnthropic, text: str, sem: asyncio.Semaphore) -> dict:
    """Single async LLM call, rate-limited by semaphore."""
    async with sem:
        msg = await client.messages.create(
            model=_LLM_MODEL,
            max_tokens=64,
            system=_SYSTEM,
            messages=[{"role": "user", "content": _USER_TMPL.format(
                types=", ".join(KNOWN_TYPES),
                text=text[:_MAX_TEXT],
            )}],
        )
    return _parse_llm_response(msg.content[0].text)


# ── 4. Single-page entry point ────────────────────────────────────────────────

def classify_page(text: str) -> dict:
    """
    Classify one page. Heuristic shortcut if very confident, else Haiku.
    Returns: { doc_type, doc_type_label_id, confidence, method }
    """
    if not text or not text.strip():
        return {"doc_type": "unknown", "doc_type_label_id": -1,
                "confidence": 0.0, "method": "heuristic"}

    h = classify_heuristic(text)
    if h["confidence"] >= HEURISTIC_SHORTCUT_THRESHOLD:
        h["doc_type_label_id"] = LABEL_ID.get(h["doc_type"], -1)
        return {k: v for k, v in h.items() if k != "all_scores"}

    result = classify_llm_sync(text)
    return result


# ── 5. Continuation-page detection ───────────────────────────────────────────

_CONTINUATION_RE = re.compile(
    r"^("
    r"date\s+description|transaction\s+date|posting\s+date|"   # bank txn tables
    r"check\s+number|description\s+amount|ref\s+#|"
    r"company\s+name\s+account|account\s+type\s+account|"      # URLA liability tables
    r"creditor\s+name|monthly\s+payment|unpaid\s+balance"      # URLA liability continuation
    r")",
    re.IGNORECASE,
)

def _is_continuation(text: str) -> bool:
    """True if page is a data-table continuation — no document-identifying header."""
    lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
    if not lines:
        return False
    # First line matches a known table header pattern
    if bool(_CONTINUATION_RE.match(lines[0])) and len(lines) > 2:
        return True
    # Very short pages with only numbers/amounts — likely a table tail
    if len(lines) <= 4 and all(re.match(r'^[\d\s\$\.,\*\-\/]+$', l) for l in lines):
        return True
    return False


# ── 6. Batch classify (PARALLEL async) — primary integration path ─────────────

def classify_pages(
    page_records,
    max_concurrent: int = 20,
) -> list[dict]:
    """
    Classify a list of PageRecord objects from extract.py.
    LLM calls run in parallel — 50 pages take ~same time as 1.

    Returns list of:
        { page_index, doc_type, doc_type_label_id, confidence, method }
    """
    return asyncio.run(_classify_pages_async(page_records, max_concurrent))


async def _classify_pages_async(page_records, max_concurrent: int) -> list[dict]:
    sem    = asyncio.Semaphore(max_concurrent)
    client = AsyncAnthropic()

    # --- Pass 1: resolve continuation pages and collect LLM tasks ---
    results        = [None] * len(page_records)
    llm_tasks      = {}   # index → asyncio task
    last_confident = None

    for idx, pr in enumerate(page_records):
        text = pr.text or ""

        if not text.strip():
            results[idx] = {
                "page_index": pr.page_index,
                "doc_type": "unknown", "doc_type_label_id": -1,
                "confidence": 0.0, "method": "heuristic",
            }
            continue

        if _is_continuation(text) and last_confident:
            results[idx] = {
                "page_index": pr.page_index,
                **last_confident,
                "method": "carry_forward",
                "confidence": 0.5,
            }
            continue

        h = classify_heuristic(text)
        if h["confidence"] >= HEURISTIC_SHORTCUT_THRESHOLD:
            entry = {
                "page_index": pr.page_index,
                "doc_type": h["doc_type"],
                "doc_type_label_id": LABEL_ID.get(h["doc_type"], -1),
                "confidence": h["confidence"],
                "method": "heuristic",
            }
            results[idx] = entry
            last_confident = {"doc_type": entry["doc_type"],
                               "doc_type_label_id": entry["doc_type_label_id"]}
        else:
            # Schedule async LLM call
            llm_tasks[idx] = asyncio.create_task(
                _classify_llm_async(client, text, sem)
            )

    # --- Pass 2: await all LLM tasks ---
    if llm_tasks:
        await asyncio.gather(*llm_tasks.values())

    for idx, task in llm_tasks.items():
        pr = page_records[idx]
        result = task.result()
        entry = {"page_index": pr.page_index, **result}
        results[idx] = entry
        if result.get("confidence", 0) >= 0.5 and result.get("doc_type") != "unknown":
            last_confident = {"doc_type": result["doc_type"],
                              "doc_type_label_id": result.get("doc_type_label_id", -1)}

    # Fill any remaining None (shouldn't happen)
    for idx, pr in enumerate(page_records):
        if results[idx] is None:
            results[idx] = {"page_index": pr.page_index, "doc_type": "unknown",
                            "doc_type_label_id": -1, "confidence": 0.0, "method": "heuristic"}

    heuristic_count  = sum(1 for r in results if r["method"] == "heuristic")
    llm_count        = sum(1 for r in results if r["method"] == "llm")
    carry_count      = sum(1 for r in results if r["method"] == "carry_forward")
    total            = len(results)

    print(f"[classify] {total} pages — "
          f"heuristic: {heuristic_count} | "
          f"carry_forward: {carry_count} | "
          f"llm (parallel): {llm_count} "
          f"({100*llm_count/total:.0f}% hit model)")

    await client.close()
    return results


# ── 7. Quick test ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    samples = [
        ("W-2",          "W-2 Wage and Tax Statement Employee social security Box 1 Wages 85000"),
        ("1040",         "Form 1040 U.S. Individual Income Tax Return Filing Status Adjusted Gross Income"),
        ("Closing disc", "Closing Disclosure Cash to Close Closing Cost Details"),
        ("Loan est",     "Loan Estimate Save this Loan Estimate"),
        ("Bank stmt",    "CHASE STATEMENT OF ACCOUNT Statement Period Checking Account Beginning Balance Ending Balance Total Deposits"),
        ("Continuation", "Date Description Withdrawals Deposits Balance\n02/08 ACH DEPOSIT 2957.02 805254.85"),
        ("Empty",        ""),
    ]

    t0 = time.time()
    for label, text in samples:
        r = classify_page(text)
        print(f"{label:15s} → {r['doc_type']:25s} id={r['doc_type_label_id']:>3}  "
              f"conf={r['confidence']:.2f}  method={r['method']}")
    print(f"\n{time.time()-t0:.2f}s")
