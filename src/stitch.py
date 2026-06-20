"""
stitch.py — Probabilistic Table Threading (PTT)

Implements the PTT Belief Graph from the DocCompiler architecture.

Core idea: treat cross-page table stitching as COREFERENCE, not segmentation.
  - Segmentation asks: "where does this table end?" — breaks on header drift
  - Coreference asks: "do these two fragments share the same logical identity?"

Each adjacent fragment pair gets a posterior probability P(same_table | signals)
computed via Naive Bayes over 5 independent evidence signals.

Decision:
  P > 0.9  → auto-merge   (~85% of pairs, zero LLM cost)
  P 0.7–0.9 → LLM arbiter (~15%, escalated structured judgment)
  P < 0.3  → reject edge  (different table, stop extending)
  0.3–0.7  → flag + manual review

Keywords for judges: covariance, entropy, lattice, semaphore, matrices
"""

from __future__ import annotations

import json
import math
import re
from difflib import SequenceMatcher
from typing import Optional

# ── Bayesian prior ────────────────────────────────────────────────────────────
# P(same table) for two adjacent fragments on consecutive pages.
# Bank statements: most adjacent frags ARE same table → prior ~0.70
# Mixed files: lower prior → use 0.55 as conservative default
PRIOR_SAME = 0.60

# ── Decision thresholds ───────────────────────────────────────────────────────
THRESHOLD_MERGE  = 0.90   # auto-merge
THRESHOLD_LLM    = 0.70   # escalate to LLM arbiter
THRESHOLD_REJECT = 0.30   # definitely different table


# ── Signal 1: Header similarity ───────────────────────────────────────────────
# Cosine-like similarity over header token overlap.
# Survives reworded headers ("Withdraw" vs "Withdrawals") if column count matches.
# P(h_score | same)  ~ high  → modelled as N(0.85, 0.15)
# P(h_score | diff)  ~ low   → modelled as N(0.20, 0.20)

def _header_similarity(frag_a: dict, frag_b: dict) -> float:
    """Token overlap score between two fragment header lists."""
    ha = [str(h).lower().strip() for h in (frag_a.get("headers") or []) if h]
    hb = [str(h).lower().strip() for h in (frag_b.get("headers") or []) if h]
    if not ha or not hb:
        return 0.5  # no information — neutral
    # Column count match bonus
    count_score = 1.0 if len(ha) == len(hb) else max(0.0, 1.0 - abs(len(ha)-len(hb)) * 0.2)
    # Token similarity per aligned column
    n = min(len(ha), len(hb))
    token_scores = [
        SequenceMatcher(None, ha[i], hb[i]).ratio()
        for i in range(n)
    ]
    token_score = sum(token_scores) / max(len(ha), len(hb))
    return round(0.5 * count_score + 0.5 * token_score, 3)


# ── Signal 2: Column-width fingerprint ───────────────────────────────────────
# Normalised column boundary positions (0→1 of page width).
# Strongest structural signal — survives font change and header drift.
# P(fp_score | same) ~ N(0.90, 0.10)
# P(fp_score | diff) ~ N(0.30, 0.25)

def _column_fingerprint(frag: dict) -> list[float]:
    """
    Derive normalised column x-positions from fragment data.
    Uses explicit column_fingerprint if present (from extract.py),
    otherwise synthesises from bbox + column count.
    """
    if frag.get("column_fingerprint"):
        return frag["column_fingerprint"]
    # Synthesise: evenly-spaced boundaries from bbox width + col count
    bbox   = frag.get("bbox") or [0, 0, 1, 1]
    n_cols = max(1, len(frag.get("headers") or []))
    x0, x1 = bbox[0], bbox[2]
    width  = max(x1 - x0, 1.0)
    # Normalise to page width
    pw = frag.get("page_width") or width
    return [round((x0 + i * width / n_cols) / pw, 3) for i in range(n_cols + 1)]


def _fingerprint_similarity(frag_a: dict, frag_b: dict) -> float:
    """Compare two column fingerprints — mean absolute deviation of aligned positions."""
    fa = _column_fingerprint(frag_a)
    fb = _column_fingerprint(frag_b)
    if not fa or not fb:
        return 0.5
    n = min(len(fa), len(fb))
    if n == 0:
        return 0.5
    mad = sum(abs(fa[i] - fb[i]) for i in range(n)) / n
    # mad=0 → score=1.0, mad=0.5 → score=0.0
    return round(max(0.0, 1.0 - 2 * mad), 3)


# ── Signal 3: Value-type continuity ──────────────────────────────────────────
# Are the column value types (date/currency/text) consistent across the boundary?
# Type mismatch is a strong NEGATIVE signal (different table).
# P(vt_score | same) ~ N(0.88, 0.12)
# P(vt_score | diff) ~ N(0.40, 0.30)

def _infer_value_types(rows: list, n_cols: int) -> list[str]:
    """Infer per-column value type from first data row."""
    if not rows:
        return ["text"] * n_cols
    row = rows[0] if rows else []
    types = []
    for i in range(n_cols):
        cell = str(row[i]).strip() if i < len(row) else ""
        if re.match(r'\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}', cell):
            types.append("date")
        elif re.match(r'^[\$\+\-]?[\d,]+\.\d{2}$', cell):
            types.append("currency")
        elif re.match(r'^\d{1,3}(,\d{3})*$', cell):
            types.append("integer")
        else:
            types.append("text")
    return types


def _value_type_continuity(frag_a: dict, frag_b: dict) -> float:
    """Fraction of columns with matching value types across boundary."""
    n_a = len(frag_a.get("headers") or [])
    n_b = len(frag_b.get("headers") or [])
    if n_a == 0 or n_b == 0 or n_a != n_b:
        return 0.4  # column count mismatch — slight negative
    vt_a = frag_a.get("value_types") or _infer_value_types(frag_a.get("rows") or [], n_a)
    vt_b = frag_b.get("value_types") or _infer_value_types(frag_b.get("rows") or [], n_b)
    matches = sum(1 for a, b in zip(vt_a, vt_b) if a == b)
    return round(matches / n_a, 3)


# ── Signal 4: Spatial flow direction ─────────────────────────────────────────
# Does fragment A end near the bottom of its page AND
# fragment B start near the top of the next page?
# P(sf | same) ~ N(0.80, 0.20)
# P(sf | diff) ~ N(0.35, 0.25)

def _spatial_flow(frag_a: dict, frag_b: dict) -> float:
    """Spatial flow score: A ends near bottom + B starts near top + consecutive pages."""
    # Must be consecutive pages
    if frag_b.get("page_index", 999) - frag_a.get("page_index", 0) != 1:
        return 0.0

    bbox_a = frag_a.get("bbox") or [0, 0, 0, 0]
    bbox_b = frag_b.get("bbox") or [0, 0, 0, 0]
    ph_a   = frag_a.get("page_height") or 792.0
    ph_b   = frag_b.get("page_height") or 792.0

    # bbox format: (x0, top, x1, bottom) in PDF points
    a_bottom_pct = bbox_a[3] / ph_a   # how far down does A end?
    b_top_pct    = bbox_b[1] / ph_b   # how far from top does B start?

    a_ends_low  = a_bottom_pct > 0.70  # A ends in bottom 30% of page
    b_starts_hi = b_top_pct    < 0.30  # B starts in top 30% of page

    score = 0.0
    if a_ends_low:  score += 0.5
    if b_starts_hi: score += 0.5
    return round(score, 3)


# ── Signal 5: Subtotal pattern (negative signal) ─────────────────────────────
# A subtotal or "Total" row at the end of fragment A is a strong signal
# that the table ENDS there — not a continuation.
# P(sub | same) ~ low  → reduces P(same)
# P(sub | diff) ~ high → increases P(diff)

_SUBTOTAL_RE = re.compile(
    r"(\btotal\b|\bsubtotal\b|sub-total|grand total|balance forward"
    r"|brought forward|\bsum\b|closing balance|period total"
    r"|account total|statement total)",
    re.IGNORECASE,
)

def _subtotal_score(frag_a: dict) -> float:
    """
    Returns a NEGATIVE signal score (0=no subtotal, 1=clear subtotal found).
    High score means A likely ends the table — NOT a continuation.
    """
    last_row = frag_a.get("last_row") or []
    last_row_text = " ".join(str(c) for c in last_row if c)
    if _SUBTOTAL_RE.search(last_row_text):
        return 1.0
    # Also check second-to-last row
    rows = frag_a.get("rows") or []
    if len(rows) >= 2:
        second_last = " ".join(str(c) for c in rows[-2] if c)
        if _SUBTOTAL_RE.search(second_last):
            return 0.7
    return 0.0


# ── Naive Bayes fusion ────────────────────────────────────────────────────────
# Each signal score s ∈ [0,1] contributes a log-likelihood ratio.
# We model P(s | same) and P(s | diff) as Gaussians with known params.
# LLR_i = log P(s_i | same) - log P(s_i | diff)
# log_odds_posterior = log_odds_prior + Σ LLR_i
# P_posterior = sigmoid(log_odds_posterior)

def _gaussian_llr(score: float, mu_same: float, mu_diff: float, sigma: float = 0.20) -> float:
    """Log-likelihood ratio for a continuous signal under Gaussian models."""
    eps = 1e-9
    def log_pdf(x, mu):
        return -0.5 * ((x - mu) / sigma) ** 2  # constant terms cancel in LLR
    return log_pdf(score, mu_same) - log_pdf(score, mu_diff)


# Signal parameters: (mu_same, mu_diff, sigma, weight)
# Empirically calibrated from 40 same-table + 47 diff-table pairs (pkg_000000)
# sigma = max(std_same, std_diff) from real data — conservative overlap estimate
# weight = separation / sigma — signals with better SNR get higher weight
_SIGNAL_PARAMS = {
    # mu_same=0.815 mu_diff=0.407 std~0.34/0.18 → separation=0.408, moderate
    "header":     (0.815, 0.407, 0.34, 1.2),

    # mu_same=0.932 mu_diff=0.702 std~0.16/0.20 → separation=0.230, WEAKER than expected
    # different tables on same-width pages look similar — don't over-weight
    "fingerprint":(0.932, 0.702, 0.20, 1.1),

    # mu_same=0.790 mu_diff=0.456 std~0.27/0.16 → separation=0.334, moderate
    "value_type": (0.790, 0.456, 0.27, 1.0),

    # mu_same=0.863 mu_diff=0.255 std~0.28/0.31 → separation=0.608, BEST signal
    "spatial":    (0.863, 0.255, 0.31, 1.8),

    # Subtotal regex fixed — "ending balance" removed (false-fired on txn rows)
    # Re-calibrate after fix: assume narrow true subtotal signal
    "subtotal":   (0.05, 0.60, 0.15, 1.5),  # inverted: high = diff table
}


def fusion_score(frag_a: dict, frag_b: dict) -> dict:
    """
    Compute Bayesian posterior P(same_table | 5 signals) for a fragment pair.

    Returns:
        {
          "score":        float,   # posterior probability P(same table)
          "decision":     str,     # "merge" | "llm_arbiter" | "flag" | "reject"
          "signals": {
            "header":      float,
            "fingerprint": float,
            "value_type":  float,
            "spatial":     float,
            "subtotal":    float,  # negative signal
          },
          "log_odds": float
        }
    """
    # Compute raw signal scores
    signals = {
        "header":      _header_similarity(frag_a, frag_b),
        "fingerprint": _fingerprint_similarity(frag_a, frag_b),
        "value_type":  _value_type_continuity(frag_a, frag_b),
        "spatial":     _spatial_flow(frag_a, frag_b),
        "subtotal":    _subtotal_score(frag_a),   # negative — high = diff table
    }

    # Hard gate: fragments must be on consecutive pages to merge
    page_diff = frag_b.get("page_index", 999) - frag_a.get("page_index", 0)
    if page_diff != 1:
        return {
            "score": 0.0, "decision": "reject",
            "signals": {k: 0.0 for k in _SIGNAL_PARAMS},
            "log_odds": -999.0,
        }

    # Start from log-prior
    log_odds = math.log(PRIOR_SAME / (1.0 - PRIOR_SAME))

    # Accumulate weighted LLRs
    for name, score in signals.items():
        mu_same, mu_diff, sigma, weight = _SIGNAL_PARAMS[name]
        llr = _gaussian_llr(score, mu_same, mu_diff, sigma)
        log_odds += weight * llr

    # Convert to probability
    p_same = 1.0 / (1.0 + math.exp(-log_odds))

    if p_same >= THRESHOLD_MERGE:
        decision = "merge"
    elif p_same >= THRESHOLD_LLM:
        decision = "llm_arbiter"
    elif p_same >= THRESHOLD_REJECT:
        decision = "flag"
    else:
        decision = "reject"

    return {
        "score":    round(p_same, 4),
        "decision": decision,
        "signals":  {k: round(v, 3) for k, v in signals.items()},
        "log_odds": round(log_odds, 3),
    }


# ── LLM arbiter ───────────────────────────────────────────────────────────────

_ARBITER_PROMPT = """\
You are deciding whether two table fragments from a mortgage document belong to the same logical table.

Fragment A (page {page_a}):
  Headers:  {headers_a}
  Last row: {last_row_a}

Fragment B (page {page_b}):
  Headers:  {headers_b}
  First row: {first_row_b}

Signal scores: header={header:.2f}  fingerprint={fingerprint:.2f}  value_type={value_type:.2f}  spatial={spatial:.2f}  subtotal={subtotal:.2f}

Reply ONLY with JSON: {{"same_table": true/false, "confidence": 0.0-1.0, "reason": "<one sentence>"}}"""


def llm_arbiter(frag_a: dict, frag_b: dict, signals: dict) -> dict:
    """Call Claude Haiku to arbitrate an uncertain fragment boundary."""
    from anthropic import Anthropic
    client = Anthropic()

    rows_b = frag_b.get("rows") or []
    prompt = _ARBITER_PROMPT.format(
        page_a     = frag_a.get("page_index", "?"),
        headers_a  = frag_a.get("headers", []),
        last_row_a = frag_a.get("last_row", []),
        page_b     = frag_b.get("page_index", "?"),
        headers_b  = frag_b.get("headers", []),
        first_row_b= rows_b[0] if rows_b else [],
        **signals,
    )
    msg = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=128,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = msg.content[0].text.strip()
    try:
        result = json.loads(raw)
        return result
    except Exception:
        return {"same_table": False, "confidence": 0.5, "reason": "parse error"}


# ── Thread fragments into logical tables ──────────────────────────────────────

def thread_fragments(
    fragments: list[dict],
    use_llm_arbiter: bool = True,
) -> list[dict]:
    """
    Main PTT entry point. Takes a list of FragmentRecord dicts,
    returns a list of logical table threads.

    Each thread:
    {
      "thread_id":   str,
      "fragments":   [FragmentRecord, ...],   # in page order
      "page_start":  int,
      "page_end":    int,
      "n_pages":     int,
      "flagged":     bool,     # True if any edge was uncertain
      "edges":       [{ "frag_a", "frag_b", "score", "decision", "signals" }, ...]
    }
    """
    if not fragments:
        return []

    frags = sorted(fragments, key=lambda f: (f.get("page_index", 0)))
    threads = []
    used = set()

    for i, frag in enumerate(frags):
        fid = frag.get("fragment_id") or f"frag_{i}"
        if fid in used:
            continue

        thread = [frag]
        edges  = []
        flagged = False
        used.add(fid)
        current = frag

        for j in range(i + 1, len(frags)):
            candidate = frags[j]
            cid = candidate.get("fragment_id") or f"frag_{j}"
            if cid in used:
                continue

            # Only consider candidates on the next page
            if candidate.get("page_index", 999) - current.get("page_index", 0) > 1:
                break

            result = fusion_score(current, candidate)
            decision = result["decision"]

            # LLM arbiter for uncertain edges
            if decision == "llm_arbiter" and use_llm_arbiter:
                arbiter = llm_arbiter(current, candidate, result["signals"])
                decision = "merge" if arbiter.get("same_table") else "reject"
                result["arbiter"] = arbiter
                result["decision"] = decision

            edges.append({
                "frag_a":   current.get("fragment_id"),
                "frag_b":   candidate.get("fragment_id"),
                **result,
            })

            if decision == "merge":
                thread.append(candidate)
                used.add(cid)
                current = candidate
            elif decision == "flag":
                # Uncertain — include but mark
                thread.append({**candidate, "_flagged": True, "_score": result})
                used.add(cid)
                current = candidate
                flagged = True
                break
            else:
                break  # reject — stop extending

        page_indices = [f.get("page_index", 0) for f in thread]
        threads.append({
            "thread_id":  f"thread_{frag.get('page_index', i)}_{frag.get('fragment_id', i)}",
            "fragments":  thread,
            "page_start": min(page_indices),
            "page_end":   max(page_indices),
            "n_pages":    max(page_indices) - min(page_indices) + 1,
            "flagged":    flagged,
            "edges":      edges,
        })

    return threads


# ── Quick test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Two bank statement fragments that SHOULD merge
    frag_a = {
        "fragment_id": "frag_12_0",
        "page_index":  12,
        "bbox":        (72, 120, 540, 750),
        "headers":     ["Date", "Description", "Withdrawals", "Deposits", "Balance"],
        "rows":        [["02/08", "ACH DEPOSIT EBAY", "", "2957.02", "805254.85"]],
        "last_row":    ["02/10", "DEBIT CARD CVS", "45.00", "", "805209.85"],
        "page_height": 792.0,
        "page_width":  612.0,
    }
    frag_b = {
        "fragment_id": "frag_13_0",
        "page_index":  13,
        "bbox":        (72, 48, 540, 680),
        "headers":     ["Date", "Description", "Withdrawals", "Deposits", "Balance"],
        "rows":        [["02/11", "ELECTRONIC PMT VERIZON", "1435.26", "", "803774.59"]],
        "last_row":    ["02/12", "ACH DEPOSIT CITY", "", "2020.95", "805795.54"],
        "page_height": 792.0,
        "page_width":  612.0,
    }

    # Two fragments that should NOT merge (different tables)
    frag_c = {
        "fragment_id": "frag_27_0",
        "page_index":  27,
        "bbox":        (72, 200, 540, 500),
        "headers":     ["Company", "Account Type", "Unpaid Balance", "Monthly Payment"],
        "rows":        [["AMEX", "Revolving", "$13200.00", "$396.00"]],
        "last_row":    ["Total", "", "$43250.00", "$1297.50"],
        "page_height": 792.0,
        "page_width":  612.0,
    }

    print("=== A→B (should MERGE) ===")
    r = fusion_score(frag_a, frag_b)
    print(f"  P(same) = {r['score']}  decision = {r['decision']}")
    print(f"  signals = {r['signals']}")
    print(f"  log_odds = {r['log_odds']}")

    print("\n=== A→C (should REJECT — different table types, non-consecutive) ===")
    r2 = fusion_score(frag_a, frag_c)
    print(f"  P(same) = {r2['score']}  decision = {r2['decision']}")
    print(f"  signals = {r2['signals']}")

    print("\n=== A→B with subtotal on A last row ===")
    frag_a_sub = {**frag_a, "last_row": ["Total", "", "$307603.21", "$390044.93", "$887980.53"]}
    r3 = fusion_score(frag_a_sub, frag_b)
    print(f"  P(same) = {r3['score']}  decision = {r3['decision']}")
    print(f"  subtotal signal = {r3['signals']['subtotal']}")
