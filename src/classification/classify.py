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
import os as _os

# Support both Anthropic and OpenAI backends.
# OpenAI is used when OPENAI_API_KEY is set (Anthropic credits exhausted fallback).
_USE_OPENAI = bool(_os.environ.get("OPENAI_API_KEY"))

if _USE_OPENAI:
    import openai as _openai
    from openai import AsyncOpenAI as _AsyncOpenAI
else:
    from anthropic import Anthropic, AsyncAnthropic

# ── 0. Label ID map — auto-discovered from labels.json, with fallback ────────

# Default mapping (used when no labels.json is available — e.g., blind classification
# on a new PDF with no ground truth). If labels.json exists, we merge any new types
# discovered there so the system adapts to schema changes automatically.
_DEFAULT_LABEL_ID = {
    "urla_1003": 0, "form_1008": 1, "loan_estimate": 2, "closing_disclosure": 3,
    "paystub": 4, "w2": 5, "voe": 6, "form_1040": 7, "schedule_1": 8,
    "schedule_c": 9, "bank_stmt_checking": 10, "bank_stmt_combo": 11,
    "brokerage_stmt": 12, "check_image": 13, "deposit_receipt": 14,
    "credit_report": 15, "du_findings": 16, "lpa_feedback": 17,
    "purchase_contract": 18, "purchase_addendum": 19, "options_addendum": 20,
    "email_correspondence": 21, "letter_of_explanation": 22, "gift_letter": 23,
    "insurance_declaration": 24, "loan_summary": 25, "filler": 26,
    # ── General document types (non-mortgage) ─────────────────────────────────
    "utility_bill": 27, "phone_bill": 28, "insurance_policy": 29, "invoice": 30,
    "narrative_chapter": 31, "textbook_chapter": 32, "medical_record": 33,
    "legal_contract": 34,
    "unknown": -1,
}


def _discover_label_ids(dataset_root: str = "DataSet ") -> dict[str, int]:
    """
    Scan labels.json files in the dataset to auto-discover doc_type → label_id mapping.
    Falls back to _DEFAULT_LABEL_ID if no labels.json is found (blind run on new PDF).
    Merges any new types found in labels.json into the default map so the system
    adapts to schema changes automatically.
    """
    import json
    from pathlib import Path

    discovered = dict(_DEFAULT_LABEL_ID)
    root = Path(dataset_root)
    if not root.exists():
        return discovered

    # Scan up to a few labels.json files to discover all types
    for labels_path in list(root.glob("pkg_*/labels.json"))[:5]:
        try:
            data = json.loads(labels_path.read_text())
            for page in data.get("pages", []):
                dt = page.get("doc_type", "")
                lid = page.get("doc_type_label_id")
                if dt and lid is not None and dt not in discovered:
                    discovered[dt] = lid
        except (json.JSONDecodeError, OSError):
            continue

    return discovered


LABEL_ID = _discover_label_ids()

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
        r"in all accounts",                                # "TOTAL ENDING BALANCE IN ALL ACCOUNTS"
        r"total ending balance",                           # same phrase, split so both count
        r"thanks for saving with",                         # Capital One 360 greeting
        r"cashflow summary",                               # Capital One 360 section
        r"360 checking|360 performance|performance savings", # Capital One 360 account names
        r"combined statement|combined account summary",
        r"savings.*checking|checking.*savings|checking and savings",
        r"all accounts summary|relationship summary",
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
        r"this form is a statement of final loan terms",
        r"projected payments",                # LE page 2 header
        r"comparisons.*in 5 years",           # LE comparisons table
        r"use these measures",                # LE "Use these measures to compare" section
        r"other considerations",              # LE page 3 section heading
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
        r"tri.?merge|merged credit",        # "Tri-Merge Merged Credit Report"
        r"beacon\s+\d|fico\s+v\d",          # "Beacon 5.0" / "FICO V2" score model names
        r"repository sources",              # "Repository Sources: Equifax · Experian"
        r"revolving account utilization",   # credit report section header
        r"trade lines|trade line",          # "TRADE LINES — CREDIT GRANTOR HISTORY"
        r"credit grantor",                  # column header in credit tables
        r"representative score",            # tri-merge score label
        r"payment history|lmt\b",           # credit report column headers
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
    # ── General document types (non-mortgage) ────────────────────────────────
    "utility_bill": [
        r"utility bill|electricity bill|electric bill|power bill",
        r"(electric|gas|water|energy|utility)\s+(company|service|provider|corp)",
        r"meter reading|kilowatt.?hour|\bkwh\b",
        r"amount due|payment due",
        r"service address|account number",
        r"billing period|bill date",
    ],
    "phone_bill": [
        r"(mobile|wireless|cellular|telecom|telephone)\s+(bill|statement|account|service)",
        r"(at&t|verizon|t-mobile|sprint|comcast|at&t wireless|metro pcs|cricket)",
        r"data usage|minutes used|text messages",
        r"monthly service fee|plan charges|line access",
        r"phone number.*account|account.*phone number",
    ],
    "insurance_policy": [
        r"insurance policy|policy number|policy #",
        r"(health|auto|car|life|dental|vision|disability|renters?|liability)\s+insurance",
        r"premium|deductible|copay|coinsurance",
        r"effective date|expiration date|coverage period",
        r"insured|policyholder|beneficiary",
        r"explanation of benefits|eob\b",
        r"claims?\s+(number|id|form|history)",
    ],
    "invoice": [
        r"\binvoice\b",
        r"invoice (number|#|no\.?)",
        r"bill to|ship to",
        r"subtotal|tax\s+amount|total\s+due|amount\s+due",
        r"payment\s+terms|due\s+date",
        r"purchase\s+order|po\s+number",
        r"quantity.*unit\s+price|description.*amount",
    ],
    "narrative_chapter": [
        r"(?:^|\n)\s*chapter\s+\d+",
        r"(?:^|\n)\s*chapter\s+[ivxlcdm]+\b",  # roman numeral chapters
        r"(?:^|\n)\s*prologue|epilogue|afterword",
        r"(?:he|she|they|it)\s+(said|asked|replied|thought|whispered|shouted)",
        r"\"\s*[A-Z].*\"\s*(?:said|replied|asked)",  # dialogue pattern
    ],
    "textbook_chapter": [
        r"(?:^|\n)\s*chapter\s+\d+",
        r"learning objectives?|objectives?:",
        r"key\s+(concepts?|terms?|points?|takeaways?)",
        r"summary\s*$|\bexercises?\s*$|\bproblems?\s*$",
        r"figure\s+\d+\.\d+|table\s+\d+\.\d+",
        r"definition\s*:|theorem\s*:|lemma\s*:|proof\s*:",
        r"review\s+questions?|discussion\s+questions?",
    ],
    "medical_record": [
        r"patient\s+(name|id|dob|date of birth)",
        r"physician|doctor|dr\.\s+\w+",
        r"diagnosis|icd.?\d+|cpt\s+code",
        r"prescription|dosage|medication",
        r"lab\s+results?|blood\s+test|urinalysis",
        r"hospital|clinic|medical\s+center",
        r"discharge\s+summary|admission\s+date",
    ],
    "legal_contract": [
        r"agreement\s+between|this\s+agreement",
        r"terms\s+and\s+conditions|terms\s+of\s+service",
        r"whereas\b|now\s*,\s*therefore\b|in\s+witness\s+whereof",
        r"party\s+of\s+the\s+first\s+part|hereinafter\s+referred\s+to",
        r"governing\s+law|jurisdiction\b",
        r"indemnification|liability\s+limitation",
        r"signature\s+page|executed\s+(on|this)\s+day",
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
        r"nmlsr id\b",                      # NMLS licence page — common filler
        r"applicant'?s? acknowledgement",
        r"borrower acknowledgment",
        r"notice of right to copy",
        r"affiliated business arrangement",
        r"servicing disclosure",
        r"appraisal independence",
        r"we collect|we may share|we do not share",   # privacy notice boilerplate
        r"why\?.*what\?.*how\?",            # privacy notice table header
        r"page \d+ of \d+\s*$",            # standalone page-number-only page
        r"ecoa|equal credit opportunity",   # ECOA notice pages (multi-page legal boilerplate)
        r"right to receive a copy",
        r"home loan toolkit",
        r"your home loan toolkit",
        r"statement of specific reasons",   # ECOA continuation pages
        r"creditor'?s? standards",          # ECOA continuation
        r"the creditor must notify",        # adverse action boilerplate
    ],
}

# Only trust heuristic when it's very confident — everything else goes to Haiku
HEURISTIC_SHORTCUT_THRESHOLD = 0.6


# ── 1b. Table-header fingerprints — format-agnostic structural classification ──
# Column headers extracted by extract.py are consistent regardless of bank/lender.
# Matched BEFORE keyword heuristic — higher precision than text patterns.
# Each entry: (frozenset of normalised header tokens, doc_type, confidence)

_HEADER_FINGERPRINTS: list[tuple[frozenset, str, float]] = [
    # Bank transactions
    (frozenset({"date", "description", "withdrawals", "deposits", "balance"}),   "bank_stmt_checking", 0.98),
    (frozenset({"date", "description", "debit", "credit", "balance"}),           "bank_stmt_checking", 0.98),
    (frozenset({"date", "description", "amount", "balance"}),                    "bank_stmt_checking", 0.92),
    (frozenset({"posting date", "description", "amount", "balance"}),            "bank_stmt_checking", 0.95),
    (frozenset({"transaction date", "description", "debit", "credit"}),          "bank_stmt_checking", 0.95),
    # Combo statement
    (frozenset({"account name", "beginning balance", "ending balance"}),         "bank_stmt_combo",    0.97),
    (frozenset({"account name", "deposits", "withdrawals", "ending balance"}),   "bank_stmt_combo",    0.97),
    # Brokerage
    (frozenset({"symbol", "shares", "price", "value"}),                          "brokerage_stmt",     0.98),
    (frozenset({"symbol", "quantity", "price", "market value"}),                 "brokerage_stmt",     0.98),
    (frozenset({"description", "shares", "price per share", "market value"}),    "brokerage_stmt",     0.96),
    # Paystub
    (frozenset({"description", "hours", "rate", "current", "ytd"}),              "paystub",            0.98),
    (frozenset({"earnings", "hours", "rate", "amount", "ytd"}),                  "paystub",            0.97),
    (frozenset({"description", "current", "year to date"}),                      "paystub",            0.90),
    # URLA liability table
    (frozenset({"creditor", "account type", "unpaid balance", "monthly payment"}),"urla_1003",         0.98),
    (frozenset({"financial institution", "account type", "account number", "cash or market value"}), "urla_1003", 0.97),
    (frozenset({"company name", "account type", "account number", "monthly payment"}), "urla_1003",   0.97),
    # Credit report
    (frozenset({"credit grantor", "type", "account number", "balance"}),         "credit_report",      0.97),
    (frozenset({"creditor", "account number", "balance", "payment status"}),     "credit_report",      0.95),
    # DU / LPA findings
    (frozenset({"category", "finding", "condition"}),                            "du_findings",        0.92),
    (frozenset({"code", "verification message"}),                                "lpa_feedback",       0.92),
    # W-2 boxes
    (frozenset({"wages tips", "federal income tax", "social security wages"}),   "w2",                 0.98),
]

_CHAPTER_NUM_RE = re.compile(
    r"^[\s\n]*(?:chapter|ch\.?)\s*(?:\d{1,3}|[ivxlcdmIVXLCDM]{1,6})\b",
    re.IGNORECASE,
)

# Narrative prose signals in the second line: pronouns + speech/movement verbs.
# These confirm the page is fiction/story content, not a financial document header.
_NARRATIVE_PROSE_RE = re.compile(
    r"\b(he|she|they|we|i)\s+\w+|"
    r"\b(said|asked|replied|thought|walked|ran|looked|smiled|laughed|cried|whispered|shouted|sat|stood)\b|"
    r"['\u2018\u2019\u201C\u201D].{5,}['\u2018\u2019\u201C\u201D]",  # quoted dialogue
    re.IGNORECASE,
)

# Financial / legal document keywords — block narrative misclassification
_FINANCIAL_HEADER_RE = re.compile(
    r"\b(loan|account|statement|invoice|payment|balance|mortgage|credit|bank|"
    r"report|underwriting|disclosure|receipt|form|schedule|paystub|tax|federal|"
    r"insurance|policy|estimate|closing|purchase|contract|agreement|id\s*#|no\s*\.?\s*\d)\b",
    re.IGNORECASE,
)


def classify_by_chapter_header(text: str) -> dict | None:
    """
    Detect pages that begin a new chapter or story.

    Format 1 — Explicit: "Chapter N" in first 100 chars (textbooks, novels)
    Format 2 — Story title: short first line + the second line is narrative prose
                            AND the title does NOT contain financial keywords

    Returns high-confidence narrative_chapter, or None.
    """
    head = text.strip()
    if not head:
        return None

    # Format 1 — explicit "Chapter N" heading
    if _CHAPTER_NUM_RE.match(head):
        return {
            "doc_type":          "narrative_chapter",
            "doc_type_label_id": LABEL_ID["narrative_chapter"],
            "confidence":        0.92,
            "method":            "chapter_header",
        }

    # Format 2 — story title page
    # Condition: first line ≤7 words, no financial keywords,
    #            second line contains narrative prose signals
    lines = head.split("\n", 2)
    if len(lines) >= 2:
        first_line = lines[0].strip()
        second_line = lines[1].strip()
        word_count = len(first_line.split())
        if (
            3 <= word_count <= 7
            and not _FINANCIAL_HEADER_RE.search(first_line)
            and _NARRATIVE_PROSE_RE.search(second_line[:200])
        ):
            return {
                "doc_type":          "narrative_chapter",
                "doc_type_label_id": LABEL_ID["narrative_chapter"],
                "confidence":        0.85,
                "method":            "chapter_header",
            }

    return None


def classify_by_table_headers(fragment_headers_list: list[list[str]]) -> dict | None:
    """
    Classify a page using its extracted table column headers.
    More robust than text keywords — works across all banks and formats.

    Args:
        fragment_headers_list: list of header rows from this page's fragments
                               e.g. [["Date","Description","Amount","Balance"], [...]]

    Returns classification dict or None if no fingerprint matched.
    """
    for headers in fragment_headers_list:
        if not headers:
            continue
        # Normalise: lowercase, strip whitespace, remove empty
        norm = frozenset(h.lower().strip() for h in headers if str(h).strip())
        for fp_set, doc_type, conf in _HEADER_FINGERPRINTS:
            # Match if fingerprint is a subset of the actual headers (handles extra cols)
            if fp_set <= norm or norm <= fp_set and len(norm) >= 2:
                return {
                    "doc_type":           doc_type,
                    "doc_type_label_id":  LABEL_ID.get(doc_type, -1),
                    "confidence":         conf,
                    "method":             "table_header",
                }
    return None


# ── 1c. Structural combo detection — format-agnostic ─────────────────────────
# bank_stmt_combo has a summary table listing MULTIPLE accounts with balances.
# This appears regardless of bank name or template.
# Pattern: 2+ lines each containing an account-like identifier + two dollar amounts
#          (beginning and ending balance columns)

_COMBO_ACCOUNT_ROW = re.compile(
    r"(checking|savings|money\s+market|360\s+checking|360\s+performance|"
    r"performance\s+savings|savings\s+account|checking\s+account)"
    r".*(?:[.]{2,}|[*]{2,})\d{4}"      # masked account number: ...XXXX or ****XXXX
    r".*\$?[\d,]+\.\d{2}"              # plus a dollar amount
    r".*\$?[\d,]+\.\d{2}",             # and a second dollar amount
    re.IGNORECASE,
)

_COMBO_SUMMARY_TOTAL = re.compile(
    r"(total|all\s+accounts)\s+.*\$?[\d,]+\.\d{2}.*\$?[\d,]+\.\d{2}",
    re.IGNORECASE,
)

def _is_combo_statement(text: str) -> bool:
    """
    Returns True if page is a multi-account combo summary.
    Requires: 2+ named account rows (checking/savings + masked account number + two amounts)
    OR: account rows + a total row.
    This prevents false positives on single-account summaries (Chase ACCOUNT SUMMARY).
    """
    lines = text.splitlines()
    account_rows = sum(1 for line in lines if _COMBO_ACCOUNT_ROW.search(line))
    has_total_row = any(_COMBO_SUMMARY_TOTAL.search(line) for line in lines)
    # Need either 2+ typed account rows, or 1 account row + explicit total row
    return account_rows >= 2 or (account_rows >= 1 and has_total_row)


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

    # Priority override: more-specific type wins over its parent type
    # bank_stmt_combo IS a bank statement, so checking signals also fire on it.
    # When both score above threshold, prefer the more specific one.
    _SPECIFICITY_OVERRIDES = [
        ("bank_stmt_combo", "bank_stmt_checking"),  # combo > checking
    ]
    for specific, general in _SPECIFICITY_OVERRIDES:
        if scores.get(specific, 0) >= 0.5 and scores.get(general, 0) >= 0.5:
            scores[general] = scores[specific] - 0.01  # specific wins by a nose

    best_type = max(scores, key=scores.get)
    best_score = scores[best_type]
    return {
        "doc_type": best_type if best_score > 0 else "unknown",
        "confidence": round(best_score, 3),
        "method": "heuristic",
        "all_scores": scores,
    }


# ── 3. LLM classification (sync + async) ─────────────────────────────────────

_LLM_MODEL       = "gpt-4o-mini" if _USE_OPENAI else "claude-haiku-4-5-20251001"
_MAX_TEXT        = 1200  # increased from 600 — more context = better accuracy on ambiguous pages

_SYSTEM = """\
You are a document page classifier for US mortgage loan files.
Classify by CONTENT and STRUCTURE — not by specific keywords, because formatting varies by bank.
Reply with ONLY a JSON object. No markdown. No explanation."""

# Used when page has at least one mortgage signal (keyword score > 0 or known table header)
# Closed-set: always returns one of the 27 types
_USER_TMPL = """\
Classify this page into one of these US mortgage document types:

  APPLICATION:    urla_1003, form_1008
  DISCLOSURES:    loan_estimate, closing_disclosure
  INCOME:         paystub, w2, voe, form_1040, schedule_1, schedule_c
  ASSETS:         bank_stmt_checking, bank_stmt_combo, brokerage_stmt, check_image, deposit_receipt
  CREDIT:         credit_report
  UNDERWRITING:   du_findings, lpa_feedback, loan_summary
  PROPERTY:       purchase_contract, purchase_addendum, options_addendum, insurance_declaration
  MISC:           email_correspondence, letter_of_explanation, gift_letter, filler

Classify by STRUCTURE — these patterns hold across ALL banks and lenders:
- Transaction rows (date/desc/amount/balance columns) → bank_stmt_checking
- Multi-account summary table: 2+ rows each with [account type]...[last 4 digits] + two balance columns, plus "All Accounts" total row → bank_stmt_combo
- Liability table (creditor/account type/unpaid balance/monthly payment) → urla_1003
- Stock/portfolio rows (symbol/shares/price/value) → brokerage_stmt
- Earnings columns (gross/net/YTD/federal tax/state tax) → paystub
- Two-column tax form with numbered boxes → w2
- Loan terms table + projected payments → loan_estimate
- Final closing cost breakdown with cash-to-close → closing_disclosure
- Dense legal/regulatory boilerplate (privacy notices, ECOA, disclosures) → filler
- Blank or section title only → filler

Page text (first {max_chars} chars):
---
{text}
---

IMPORTANT: doc_type must be one of the exact type names listed above (e.g. "bank_stmt_checking", "urla_1003") — NOT a category name like "ASSETS" or "INCOME".

JSON only: {{"doc_type": "<type>", "confidence": <0.0–1.0>, "reasoning": "<one line>"}}"""

# Used when page has ZERO mortgage signals — open-set, can return "unknown"
_USER_TMPL_OPENSET = """\
Does this page belong to a US mortgage loan file?

If YES, classify it:
  APPLICATION: urla_1003, form_1008 | DISCLOSURES: loan_estimate, closing_disclosure
  INCOME: paystub, w2, voe, form_1040, schedule_1, schedule_c
  ASSETS: bank_stmt_checking, bank_stmt_combo, brokerage_stmt, check_image, deposit_receipt
  CREDIT: credit_report | UNDERWRITING: du_findings, lpa_feedback, loan_summary
  PROPERTY: purchase_contract, purchase_addendum, options_addendum, insurance_declaration
  MISC: email_correspondence, letter_of_explanation, gift_letter, filler

If NO (magazine article, recipe, medical record, news, fiction, product manual, etc.) → "unknown"

Page text (first {max_chars} chars):
---
{text}
---

JSON only: {{"doc_type": "<type or unknown>", "confidence": <0.0–1.0>, "reasoning": "<one line>"}}"""


def _parse_llm_response(raw: str) -> dict:
    raw = raw.strip()
    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*?\}", raw, re.DOTALL)
        if not match:
            return {"doc_type": "unknown", "confidence": 0.0, "method": "llm_parse_error"}
        result = json.loads(match.group())

    # Normalize doc_type: lowercase, strip whitespace, fix common LLM typos
    dt = result.get("doc_type", "unknown").lower().strip()

    # Fix common typos/variants from GPT-4o-mini
    _TYPO_MAP = {"voie": "voe", "voe_form": "voe", "w-2": "w2"}
    dt = _TYPO_MAP.get(dt, dt)

    # GPT-4o-mini sometimes returns category names instead of specific types.
    _CATEGORY_FALLBACK = {
        "application": "urla_1003", "disclosures": "loan_estimate",
        "income": "paystub", "assets": "bank_stmt_checking",
        "credit": "credit_report", "underwriting": "du_findings",
        "property": "purchase_contract", "misc": "filler",
    }
    if dt in _CATEGORY_FALLBACK:
        dt = _CATEGORY_FALLBACK[dt]

    result["doc_type"] = dt
    result["method"] = "llm"
    result["doc_type_label_id"] = LABEL_ID.get(dt, -1)
    return result


def _pick_prompt(text: str) -> str:
    """
    Use open-set prompt (allows unknown) only when text has zero mortgage signals.
    For pages with any financial/legal signal, use the closed-set prompt —
    it's more accurate on mortgage docs and won't confuse filler for unknown.
    """
    scores = _score_page(text)
    has_any_signal = any(v > 0 for v in scores.values())
    return _USER_TMPL if has_any_signal else _USER_TMPL_OPENSET


def classify_llm_sync(text: str) -> dict:
    """Sync LLM call — use only for single-page classification."""
    prompt = _pick_prompt(text).format(max_chars=_MAX_TEXT, text=text[:_MAX_TEXT])
    if _USE_OPENAI:
        client = _openai.OpenAI()
        resp = client.chat.completions.create(
            model=_LLM_MODEL, max_tokens=128,
            messages=[{"role": "system", "content": _SYSTEM},
                      {"role": "user",   "content": prompt}],
        )
        return _parse_llm_response(resp.choices[0].message.content)
    else:
        from anthropic import Anthropic
        client = Anthropic()
        msg = client.messages.create(
            model=_LLM_MODEL, max_tokens=128, system=_SYSTEM,
            messages=[{"role": "user", "content": prompt}],
        )
        return _parse_llm_response(msg.content[0].text)


async def _classify_llm_async(client, text: str, sem: asyncio.Semaphore) -> dict:
    """Single async LLM call, rate-limited by semaphore. Retries on 429."""
    prompt = _pick_prompt(text).format(max_chars=_MAX_TEXT, text=text[:_MAX_TEXT])
    for attempt in range(5):
        async with sem:
            try:
                if _USE_OPENAI:
                    resp = await client.chat.completions.create(
                        model=_LLM_MODEL, max_tokens=128,
                        messages=[{"role": "system", "content": _SYSTEM},
                                  {"role": "user",   "content": prompt}],
                    )
                    return _parse_llm_response(resp.choices[0].message.content)
                else:
                    msg = await client.messages.create(
                        model=_LLM_MODEL, max_tokens=128, system=_SYSTEM,
                        messages=[{"role": "user", "content": prompt}],
                    )
                    return _parse_llm_response(msg.content[0].text)
            except Exception as e:
                if "429" in str(e) or "rate" in str(e).lower():
                    await asyncio.sleep(2 ** attempt)
                    continue
                raise
    return {"doc_type": "unknown", "confidence": 0.0, "method": "llm_rate_limited"}


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
    # First line matches a known table header pattern AND page has enough data rows
    if bool(_CONTINUATION_RE.match(lines[0])) and len(lines) > 5:
        # Verify at least some lines look like table data (contain numbers/currency)
        data_lines = sum(1 for l in lines[1:] if re.search(r'\d+\.\d{2}|\$[\d,]+', l))
        if data_lines >= 2:
            return True
    return False


# ── 6. Batch classify (PARALLEL async) — primary integration path ─────────────

def classify_pages(
    page_records,
    max_concurrent: int = 20,
    api_key: str | None = None,
) -> list[dict]:
    """
    Classify a list of PageRecord objects from extract.py.
    LLM calls run in parallel — 50 pages take ~same time as 1.

    Returns list of:
        { page_index, doc_type, doc_type_label_id, confidence, method }
    """
    return asyncio.run(_classify_pages_async(page_records, max_concurrent, api_key))


async def _classify_pages_async(
    page_records, max_concurrent: int, api_key: str | None = None
) -> list[dict]:
    sem    = asyncio.Semaphore(max_concurrent)
    client = _AsyncOpenAI() if _USE_OPENAI else AsyncAnthropic(api_key=api_key) if api_key else AsyncAnthropic()

    # --- Pass 1: resolve continuation pages and collect LLM tasks ---
    results        = [None] * len(page_records)
    llm_tasks      = {}   # index → asyncio task
    last_confident = None
    carry_streak   = 0    # consecutive carry_forwards — cap to prevent runaway propagation
    _MAX_CARRY     = 8    # re-evaluate after this many consecutive carry_forwards

    for idx, pr in enumerate(page_records):
        text = pr.text or ""

        if not text.strip() or len(text.strip()) < 20:
            # Near-blank pages are almost always filler (cover pages, tab sheets, intentional blanks)
            results[idx] = {
                "page_index": pr.page_index,
                "doc_type": "filler", "doc_type_label_id": LABEL_ID["filler"],
                "confidence": 0.85, "method": "heuristic",
            }
            carry_streak = 0
            continue

        if _is_continuation(text) and last_confident and carry_streak < _MAX_CARRY:
            results[idx] = {
                "page_index": pr.page_index,
                **last_confident,
                "method": "carry_forward",
                "confidence": 0.5,
            }
            carry_streak += 1
            continue

        # Chapter-header shortcut — fires before any other check.
        # If a page opens with "Chapter N / Section N", classify immediately
        # and set carry-forward so body pages inherit the chapter type.
        ch = classify_by_chapter_header(text)
        if ch:
            entry = {"page_index": pr.page_index, **ch}
            results[idx] = entry
            last_confident = {"doc_type": ch["doc_type"],
                               "doc_type_label_id": ch["doc_type_label_id"]}
            carry_streak = 0
            continue

        # Table-header fingerprint — most robust signal, runs before text keywords
        fragment_headers = getattr(pr, "fragment_headers", None) or []
        th = classify_by_table_headers(fragment_headers)
        if th:
            entry = {"page_index": pr.page_index, **th}
            results[idx] = entry
            last_confident = {"doc_type": th["doc_type"],
                               "doc_type_label_id": th["doc_type_label_id"]}
            carry_streak = 0
            continue

        # Structural combo detection — format-agnostic, runs before keyword heuristic.
        # Detects multi-account summary tables regardless of bank name or template.
        if _is_combo_statement(text):
            entry = {
                "page_index": pr.page_index,
                "doc_type": "bank_stmt_combo",
                "doc_type_label_id": LABEL_ID["bank_stmt_combo"],
                "confidence": 0.92,
                "method": "heuristic",
            }
            results[idx] = entry
            last_confident = {"doc_type": "bank_stmt_combo",
                               "doc_type_label_id": LABEL_ID["bank_stmt_combo"]}
            carry_streak = 0
            continue

        # If the previous confident page was narrative/textbook and THIS page has
        # no competing high-confidence signal, carry-forward the chapter type.
        if last_confident and last_confident["doc_type"] in (
            "narrative_chapter", "textbook_chapter"
        ) and carry_streak < _MAX_CARRY:
            results[idx] = {
                "page_index": pr.page_index,
                **last_confident,
                "method": "carry_forward",
                "confidence": 0.60,
            }
            carry_streak += 1
            continue

        h = classify_heuristic(text)
        if (h["doc_type"] == "bank_stmt_combo" and h["confidence"] >= 0.45):
            entry = {
                "page_index": pr.page_index,
                "doc_type": "bank_stmt_combo",
                "doc_type_label_id": LABEL_ID["bank_stmt_combo"],
                "confidence": h["confidence"],
                "method": "heuristic",
            }
            results[idx] = entry
            last_confident = {"doc_type": "bank_stmt_combo",
                               "doc_type_label_id": LABEL_ID["bank_stmt_combo"]}
            continue
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
            carry_streak = 0
        elif client is not None:
            # Schedule async LLM call
            llm_tasks[idx] = asyncio.create_task(
                _classify_llm_async(client, text, sem)
            )
        else:
            # No API key — fall through to heuristic best-guess or unknown
            best_type, best_score = max(
                _score_page(text).items(), key=lambda kv: kv[1], default=("unknown", 0.0)
            )
            results[idx] = {
                "page_index": pr.page_index,
                "doc_type": best_type if best_score > 0 else "unknown",
                "doc_type_label_id": LABEL_ID.get(best_type if best_score > 0 else "unknown", -1),
                "confidence": min(best_score, 0.59),
                "method": "heuristic",
            }

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
            carry_streak = 0

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

    if client is not None:
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
