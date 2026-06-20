# Pipeline Results — Pagination Test

**17 PDFs | 2,295 pages | 742 document instances detected**

## Structure

```
results/
├── README.md                  ← this file
├── summary.json               ← aggregated results for all 17 PDFs
├── doc_000/
│   └── pipeline_output.json   ← full pipeline output (pages, documents, tables)
├── doc_001/
│   └── pipeline_output.json
├── ...
└── doc_016/
    └── pipeline_output.json
```

## How to verify

Each `pipeline_output.json` contains:

```json
{
  "documents": [
    {
      "doc_instance_id": "bank_stmt_checking#1",
      "doc_type": "bank_stmt_checking",
      "start_page": 15,
      "end_page": 24,
      "page_count": 10,
      "distinguishing_attr": "06/08/2023"
    }
  ],
  "pages": [
    {
      "page_index": 0,
      "doc_type": "form_1008",
      "doc_type_label_id": 1,
      "doc_instance_id": "form_1008#1",
      "is_first_page_of_doc": true,
      "page_in_doc": 1,
      "total_pages_in_doc": 1,
      "render_mode": "digital | scanned",
      "text": "...",
      "has_table": true,
      "fragment_ids": ["frag_0_0"]
    }
  ],
  "tables": [
    {
      "table_id": "table_0000",
      "pages": [15, 16, 17],
      "fragment_count": 3
    }
  ]
}
```

## Results Summary

| PDF | Pages | Documents | Top Types |
|-----|-------|-----------|-----------|
| doc_000 | 161 | 60 | bank_stmt_checking(78), filler(21), narrative_chapter(9) |
| doc_001 | 151 | 58 | bank_stmt_checking(58), unknown(23), filler(18) |
| doc_002 | 152 | 60 | bank_stmt_checking(76), filler(22), narrative_chapter(13) |
| doc_003 | 155 | 49 | filler(76), bank_stmt_checking(31), narrative_chapter(18) |
| doc_004 | 24 | 1 | unknown(24) — non-mortgage document (company brochure) |
| doc_005 | 137 | 54 | bank_stmt_checking(53), filler(47), closing_disclosure(4) |
| doc_006 | 138 | 49 | bank_stmt_checking(65), filler(15), narrative_chapter(13) |
| doc_007 | 154 | 37 | filler(71), bank_stmt_checking(47), unknown(11) |
| doc_008 | 149 | 34 | filler(73), bank_stmt_checking(48), closing_disclosure(4) |
| doc_009 | 166 | 50 | filler(72), bank_stmt_checking(55), narrative_chapter(9) |
| doc_010 | 160 | 49 | bank_stmt_checking(74), filler(25), narrative_chapter(22) |
| doc_011 | 145 | 34 | filler(64), bank_stmt_checking(38), narrative_chapter(16) |
| doc_012 | 130 | 51 | bank_stmt_checking(64), filler(20), unknown(13) |
| doc_013 | 150 | 37 | filler(95), bank_stmt_checking(23), unknown(4) |
| doc_014 | 18 | 5 | unknown(16), filler(2) — non-mortgage document |
| doc_015 | 147 | 57 | bank_stmt_checking(78), filler(25), closing_disclosure(5) |
| doc_016 | 158 | 57 | bank_stmt_checking(74), filler(35), paystub(9) |

## How to reproduce

```bash
# Place PDFs in pagination-test/ folder
# Run pipeline on all:
export OPENAI_API_KEY=sk-...
for d in results/doc_*; do
    python3 src/pipeline/run_pipeline.py --pkg "$d"
done
```

## Pipeline stages per PDF

1. **Extract** — pdfplumber (digital) + GPT-4o-mini VLM (scanned, 50 concurrent)
2. **Classify** — 27 doc types, FrugalGPT cascade (heuristic → carry_forward → LLM)
3. **Segment** — Pairwise boundary detection → document instances
4. **Stitch** — Probabilistic Table Threading (PTT), Naive Bayes fusion
5. **Render** — Final JSON output
