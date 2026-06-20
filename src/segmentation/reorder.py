"""
reorder.py — General Page Sequence Reconstruction

Works for ANY PDF without page numbers:
    textbook, story, comic, legal doc, bank statements, notes.

Algorithm — 3 paths, cheapest first:

  PATH A  Chapter-structured documents  (textbooks, notes, manuals)
          Detect pages with "Chapter N / Section N" headers.
          Sort by chapter number. Assign orphan pages to nearest
          chapter by keyword overlap. O(n) after detection, zero LLM.

  PATH B  Flowing-text documents  (stories, reports, research papers)
          Build pairwise score matrix from 5 structural signals:
            • Sentence continuation  (A ends mid-sentence, B continues)
            • Numbered list flow     (A ends "3.", B starts "4.")
            • Keyword overlap        (Jaccard of last/first paragraphs)
            • Pronoun reference      (B starts "It/This/These...")
            • Transition phrases     (B starts "However/Therefore...")
          Greedy Hamiltonian path through score matrix. O(n²), zero LLM.

  PATH C  LLM semantic boost  (any document, called only on hard pairs)
          Claude Haiku: "Does page B directly follow page A?" → 0..1
          Async parallel. Called only when structural score < threshold.
"""

from __future__ import annotations

import asyncio
import json
import re
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _tail(text: str, n: int = 400) -> str:
    return text[-n:].strip() if text else ""

def _head(text: str, n: int = 400) -> str:
    return text[:n].strip() if text else ""

def _keywords(text: str) -> set[str]:
    STOP = {
        "the","and","that","this","with","from","they","have","been","were",
        "will","also","which","when","into","for","are","was","can","its",
        "not","but","all","one","their","more","such","each","both","than",
    }
    return {w.lower() for w in re.findall(r"\b[a-zA-Z]{4,}\b", text)} - STOP


# ─────────────────────────────────────────────────────────────────────────────
# PATH A — Chapter / Section number detection
# ─────────────────────────────────────────────────────────────────────────────

_CHAPTER_RE = re.compile(
    r"(?:chapter|section|unit|module|part|lecture|ch\.?)\s*(\d{1,3})\b",
    re.IGNORECASE,
)
_FIRST_PAGE_RE = re.compile(
    r"\b(table\s+of\s+contents|preface|introduction|foreword|index|contents)\b",
    re.IGNORECASE,
)
_LAST_PAGE_RE = re.compile(
    r"\b(bibliography|references|appendix|conclusion|summary|glossary|index)\b",
    re.IGNORECASE,
)


def _extract_chapter_num(text: str) -> Optional[int]:
    """Extract the first chapter/section number from the top of a page."""
    # Only look in the first 300 chars (header area) for chapter labels
    m = _CHAPTER_RE.search(text[:300])
    return int(m.group(1)) if m else None


def _is_first_page(text: str) -> bool:
    return bool(_FIRST_PAGE_RE.search(text[:500]))


def _is_last_page(text: str) -> bool:
    return bool(_LAST_PAGE_RE.search(text[-300:]))


def _path_a_chapter_order(pages: list[dict]) -> Optional[list[dict]]:
    """
    PATH A: Order pages using chapter/section numbers.

    1. Classify each page:
       - chapter_start: has 'Chapter N' in first 300 chars → anchor
       - first_page:    Table of Contents / Preface → goes first
       - last_page:     Bibliography / Appendix → goes last
       - orphan:        no chapter number → must be assigned to a chapter

    2. Sort chapter_start pages by chapter number.

    3. For each orphan page, find the chapter-start page with the
       highest keyword overlap → assign it there.

    4. Within each chapter group, order orphan pages by keyword
       similarity to the chapter-start (closest = immediately after start).

    Returns ordered list, or None if not enough chapter structure found.
    """
    chapter_pages: dict[int, dict] = {}   # chapter_num → page
    first_pages:   list[dict]      = []
    last_pages:    list[dict]      = []
    orphans:       list[dict]      = []

    for p in pages:
        text = p.get("text") or ""
        cn   = _extract_chapter_num(text)

        if cn is not None:
            # If two pages have the same chapter number, keep the one
            # that actually starts the chapter (has the heading near the top)
            if cn not in chapter_pages:
                chapter_pages[cn] = p
            else:
                # Both claim same chapter — the one with heading in first 100 chars wins
                existing_pos = (chapter_pages[cn].get("text") or "").find(f"{cn}")
                new_pos      = text.find(str(cn))
                if new_pos < existing_pos:
                    orphans.append(chapter_pages[cn])   # demote old to orphan
                    chapter_pages[cn] = p
                else:
                    orphans.append(p)
        elif _is_first_page(text):
            first_pages.append(p)
        else:
            orphans.append(p)

    # Need at least 3 distinct chapter numbers AND they must cover ≥30% of pages
    if len(chapter_pages) < 3:
        return None
    total_pages = len(pages)
    chapter_coverage = len(chapter_pages) / total_pages
    if chapter_coverage < 0.25:
        return None   # chapter structure only exists in a small minority → wrong path

    # Sort chapter anchor pages by chapter number
    sorted_chapters = sorted(chapter_pages.items(), key=lambda x: x[0])
    chapter_order   = [p for _, p in sorted_chapters]

    # Assign each orphan to the chapter with highest keyword overlap
    def _best_chapter(orphan: dict) -> int:
        """Return index into chapter_order for best matching chapter."""
        ok = _keywords((orphan.get("text") or "")[:600])
        best_idx   = 0
        best_score = -1.0
        for idx, cp in enumerate(chapter_order):
            ck    = _keywords((cp.get("text") or "")[:600])
            score = len(ok & ck) / len(ok | ck) if (ok | ck) else 0.0
            if score > best_score:
                best_score = score
                best_idx   = idx
        return best_idx

    # Build groups: {chapter_idx: [chapter_start_page, orphan1, orphan2, ...]}
    groups: dict[int, list[dict]] = {i: [cp] for i, cp in enumerate(chapter_order)}
    for orphan in orphans:
        if _is_last_page(orphan.get("text") or ""):
            last_pages.append(orphan)
        else:
            ci = _best_chapter(orphan)
            groups[ci].append(orphan)

    # Within each group, order the non-start pages by overlap with chapter start
    # (highest overlap = most directly follows the chapter header)
    for ci, group in groups.items():
        if len(group) <= 1:
            continue
        anchor   = group[0]
        ak       = _keywords((anchor.get("text") or "")[:600])
        rest     = group[1:]
        rest_scored = []
        for p in rest:
            pk = _keywords((p.get("text") or "")[:600])
            s  = len(ak & pk) / len(ak | pk) if (ak | pk) else 0.0
            rest_scored.append((s, p))
        rest_scored.sort(key=lambda x: -x[0])   # highest overlap first
        groups[ci] = [anchor] + [p for _, p in rest_scored]

    # Flatten: first_pages → chapter groups in order → last_pages
    ordered = first_pages[:]
    for ci in sorted(groups.keys()):
        ordered.extend(groups[ci])
    ordered.extend(last_pages)

    return ordered


# ─────────────────────────────────────────────────────────────────────────────
# PATH B — Structural pairwise signals (flowing text)
# ─────────────────────────────────────────────────────────────────────────────

def _sentence_cut(tail_a: str, head_b: str) -> float:
    """A ends mid-sentence; B continues it."""
    if not tail_a or not head_b:
        return 0.0
    last_char  = tail_a.rstrip()[-1] if tail_a.rstrip() else ""
    first_char = head_b.lstrip()[0]  if head_b.lstrip() else ""
    fw_match   = re.match(r"\s*(\w+)", head_b)
    first_w    = fw_match.group(1).lower() if fw_match else ""
    CONNECTORS = {"and","but","or","so","yet","which","that","who","where",
                  "when","while","although","however","therefore","thus",
                  "moreover","furthermore","consequently","additionally"}
    score = 0.0
    if last_char not in ".!?\"'":   score += 0.45
    if first_char.islower():        score += 0.35
    if first_w in CONNECTORS:       score += 0.20
    return min(score, 1.0)


def _numbered_list(tail_a: str, head_b: str) -> float:
    """A ends at list item N; B starts with item N+1."""
    nums = re.findall(r"(?:^|\n)\s*(\d+)[.)]\s+\S", tail_a)
    if not nums:
        return 0.0
    next_re = re.match(r"^\s*" + str(int(nums[-1]) + 1) + r"[.)]\s+", head_b)
    return 0.95 if next_re else 0.0


def _keyword_overlap(tail_a: str, head_b: str) -> float:
    """Jaccard similarity of keywords at page boundary."""
    wa, wb = _keywords(tail_a), _keywords(head_b)
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / len(wa | wb)


def _balance_chain(text_a: str, text_b: str) -> float:
    """
    Bank statement signal: ending balance of page A == beginning balance of page B.
    When it matches → very high confidence B follows A.
    """
    _AMT = re.compile(r"\$?\s*([\d,]+\.\d{2})")
    _END = re.compile(
        r"(?:ending|closing|final|new)\s*balance[:\s]+\$?\s*([\d,]+\.\d{2})",
        re.IGNORECASE,
    )
    _BEG = re.compile(
        r"(?:beginning|opening|starting|previous|prior)\s*balance[:\s]+\$?\s*([\d,]+\.\d{2})",
        re.IGNORECASE,
    )

    def _amt(m): return float(m.group(1).replace(",","")) if m else None

    end_a = _amt(_END.search(text_a))
    beg_b = _amt(_BEG.search(text_b))

    if end_a is not None and beg_b is not None:
        return 0.98 if abs(end_a - beg_b) < 0.02 else 0.0

    # Fallback: last amount on page A == first amount on page B
    amounts_a = [float(m.group(1).replace(",","")) for m in _AMT.finditer(_tail(text_a, 200))]
    amounts_b = [float(m.group(1).replace(",","")) for m in _AMT.finditer(_head(text_b, 200))]
    if amounts_a and amounts_b and abs(amounts_a[-1] - amounts_b[0]) < 0.02:
        return 0.75
    return 0.0


def _date_sequence(text_a: str, text_b: str) -> float:
    """
    Date on last transaction of A ≤ date on first transaction of B.
    Works for bank stmts, invoices, emails — any date-ordered document.
    """
    _DATE = re.compile(r"\b(\d{1,2})[/\-](\d{1,2})[/\-](\d{2,4})\b")

    def _dates(text):
        results = []
        for m in _DATE.finditer(text):
            month, day, year = int(m.group(1)), int(m.group(2)), int(m.group(3))
            if year < 100: year += 2000
            if 1 <= month <= 12 and 1 <= day <= 31:
                results.append((year, month, day))
        return results

    dates_a = _dates(_tail(text_a, 400))
    dates_b = _dates(_head(text_b, 400))
    if not dates_a or not dates_b:
        return 0.0
    last_a  = dates_a[-1]
    first_b = dates_b[0]
    if last_a <= first_b:
        return 0.55   # dates flow forward
    if last_a > first_b:
        return 0.0    # dates go backward → B does NOT follow A
    return 0.0


def _pronoun_ref(head_b: str) -> float:
    """B starts with a back-reference pronoun/phrase."""
    REF_RE = re.compile(
        r"^\s*(?:it|this|these|those|they|such|the\s+above|as\s+(?:mentioned"
        r"|shown|described)|furthermore|moreover|consequently|therefore|thus)\b",
        re.IGNORECASE,
    )
    return 0.35 if REF_RE.match(head_b) else 0.0


def structural_score(page_a: dict, page_b: dict) -> float:
    """P(page B directly follows page A) — zero LLM, structural only."""
    text_a = page_a.get("text") or ""
    text_b = page_b.get("text") or ""
    if not text_a or not text_b:
        return 0.0

    tail_a = _tail(text_a)
    head_b = _head(text_b)

    # Balance chaining dominates for financial docs (bank stmts, invoices)
    bc = _balance_chain(text_a, text_b)
    if bc > 0.7:
        return bc

    # Numbered list dominates for structured lists
    nl = _numbered_list(tail_a, head_b)
    if nl > 0.5:
        return 0.95 * nl + 0.05 * _keyword_overlap(tail_a, head_b)

    sc  = _sentence_cut(tail_a, head_b)
    wo  = _keyword_overlap(tail_a, head_b)
    pr  = _pronoun_ref(head_b)
    ds  = _date_sequence(text_a, text_b)

    return round(min(0.35*sc + 0.25*wo + 0.20*nl + 0.10*pr + 0.10*ds, 1.0), 4)


def _greedy_hamiltonian(pages: list[dict], matrix: list[list[float]]) -> list[int]:
    """
    Greedy O(n²) Hamiltonian path.
    Start from the page with the lowest 'best incoming score'
    (nothing points strongly to it → likely the first page).
    """
    n = len(pages)
    best_in = [max((matrix[i][j] for i in range(n) if i != j), default=0.0)
               for j in range(n)]
    start   = min(range(n), key=lambda j: best_in[j])

    visited = [False] * n
    order   = [start]
    visited[start] = True

    for _ in range(n - 1):
        cur = order[-1]
        best_j, best_s = -1, -1.0
        for j in range(n):
            if not visited[j] and matrix[cur][j] > best_s:
                best_s, best_j = matrix[cur][j], j
        if best_j == -1:
            best_j = next(
                j for j in sorted(range(n), key=lambda k: pages[k]["page_index"])
                if not visited[j]
            )
        visited[best_j] = True
        order.append(best_j)

    return order


# ─────────────────────────────────────────────────────────────────────────────
# PATH C — LLM semantic scoring (Claude Haiku, async parallel)
# ─────────────────────────────────────────────────────────────────────────────

_LLM_PROMPT = """\
You are checking whether two pages from a document are in the correct order.

END of Page A (last ~300 chars):
---
{tail_a}
---

START of Page B (first ~300 chars):
---
{head_b}
---

Does page B directly and naturally follow page A in the same document?
Reply with JSON only:
{{"follows": true/false, "score": 0.0-1.0, "reason": "one line"}}
score=1.0 → B definitely follows A. score=0.0 → definitely does not."""


async def _llm_one(client, sem, page_a: dict, page_b: dict) -> float:
    async with sem:
        try:
            msg = await client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=80,
                messages=[{"role": "user", "content": _LLM_PROMPT.format(
                    tail_a=_tail(page_a.get("text") or "", 300),
                    head_b=_head(page_b.get("text") or "", 300),
                )}],
            )
            d = json.loads(msg.content[0].text.strip())
            s = float(d.get("score", 0.5))
            return s if d.get("follows") else 1.0 - s
        except Exception:
            return 0.0


async def _llm_score_pairs(pairs: list[tuple[dict, dict]], api_key: str,
                           concurrency: int = 15) -> list[float]:
    from anthropic import AsyncAnthropic
    sem = asyncio.Semaphore(concurrency)
    async with AsyncAnthropic(api_key=api_key) as client:
        tasks = [_llm_one(client, sem, a, b) for a, b in pairs]
        results = await asyncio.gather(*tasks, return_exceptions=True)
    return [float(r) if not isinstance(r, Exception) else 0.0 for r in results]


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def reorder_pages(
    pages: list[dict],
    api_key: Optional[str]  = None,
    llm_threshold: float    = 0.20,
    max_llm_pairs: int      = 300,
) -> list[dict]:
    """
    Reorder pages of ANY PDF into correct sequence — no page numbers needed.

    Args:
        pages:         list of dicts, each needs 'page_index' + 'text'
        api_key:       Anthropic key — enables LLM on hard pairs (optional)
        llm_threshold: escalate to LLM when best structural score < this
        max_llm_pairs: cap on LLM calls (cost control)

    Returns:
        Same pages list. Each page gets:
          sorted_page_index  — 0-based correct position
          reorder_method     — which path resolved it
    """
    n = len(pages)
    if n == 0:
        return pages
    if n == 1:
        pages[0].update(sorted_page_index=0, reorder_method="single")
        return pages

    # ── PATH A: Chapter-structured document ──────────────────────────────────
    print(f"[reorder] {n} pages — trying PATH A (chapter structure)...")
    ordered = _path_a_chapter_order(pages)

    if ordered:
        print(f"[reorder] PATH A succeeded — chapter/section ordering applied.")
        for i, p in enumerate(ordered):
            p["sorted_page_index"] = i
            p["reorder_method"]    = "chapter_structure"
        return ordered

    # ── PATH B: Structural pairwise (flowing text) ────────────────────────────
    print(f"[reorder] PATH A insufficient — trying PATH B (structural pairwise)...")
    matrix    = [[0.0]*n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j:
                matrix[i][j] = structural_score(pages[i], pages[j])

    best_out   = [max(matrix[i][j] for j in range(n) if j != i) for i in range(n)]
    weak       = [i for i, s in enumerate(best_out) if s < llm_threshold]
    avg_score  = sum(best_out) / n

    print(f"[reorder] PATH B: avg best-score={avg_score:.3f}, "
          f"weak={len(weak)}/{n} (threshold={llm_threshold})")

    method = "structural"

    # ── PATH C: LLM on weak pairs ─────────────────────────────────────────────
    if api_key and weak:
        pairs_idx: list[tuple[int,int]] = []
        for wi in weak:
            for j in range(n):
                if j != wi:
                    pairs_idx.append((wi, j))
                    pairs_idx.append((j, wi))

        if len(pairs_idx) > max_llm_pairs:
            pairs_idx = pairs_idx[:max_llm_pairs]

        print(f"[reorder] PATH C: {len(pairs_idx)} LLM calls (Haiku, parallel)...")
        pair_objs = [(pages[i], pages[j]) for i,j in pairs_idx]
        llm_vals  = asyncio.run(_llm_score_pairs(pair_objs, api_key))

        for (i, j), v in zip(pairs_idx, llm_vals):
            matrix[i][j] = 0.30 * matrix[i][j] + 0.70 * v

        method = "structural+llm"
        print(f"[reorder] PATH C done.")
    elif weak and not api_key:
        print(f"[reorder] {len(weak)} weak pages — add --api-key for LLM boost")

    order = _greedy_hamiltonian(pages, matrix)
    for rank, pi in enumerate(order):
        pages[pi]["sorted_page_index"] = rank
        pages[pi]["reorder_method"]    = method

    print(f"[reorder] Done. Method={method}")
    return pages
