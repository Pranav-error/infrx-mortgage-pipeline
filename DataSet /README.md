# US Mortgage Loan-Package Dataset

40 US residential mortgage loan-origination packages with machine-readable ground truth.
Each package is one multi-document, multi-page PDF — the bundle a lender assembles for a
single loan file (application forms, disclosures, income and asset documents, credit
report, underwriting output, property documents, correspondence, and boilerplate) — paired
with annotations describing its structure and contents. Packages vary in length (roughly
110–190 pages).

It supports document-AI work such as pagination and document classification, multi-page
table extraction, chart/plot understanding, and document question answering. Pages are a
mix of clean digital renders and scan-style images (grayscale, noise, skew), with a few
heavily rotated (90/180/270°).

## Layout

```
loan-packages-dataset/
├── README.md
└── pkg_000000 … pkg_000039     (40 packages)
```

Each `pkg_*/` directory contains three files:

- `package.pdf` — the loan package (the input).
- `labels.json` — structural ground truth.
- `qa.json` — question-answering ground truth.

All annotations reference `package.pdf` by 0-based page index and PDF-point coordinates
(top-left origin, `[x0, y0, x1, y1]`); see `coord_system` in `labels.json` for the
specifics, including how the `_px` fields relate to scanned/rotated pages.

## `labels.json`

- `documents[]` — the constituent documents and their page ranges.
- `pages[]` — per-page document type, boundary flags, and render/rotation info.
- `tables[]` — logical tables (which may span pages) with per-cell boxes.
- `charts[]` — embedded charts with their underlying data series and per-element boxes.
- plus `coord_system`, `total_pages`, and other top-level metadata.

Page-level document types are integer-coded; the key ↔ id mapping is at the end of this
file.

## `qa.json`

`qa[]` is a list of questions about the package. Each item carries the `question` (and a
meaning-preserving `question_rephrased`), the `answer` and its `answer_type`, a `kind` and
`difficulty`, an `answerable` flag, and `evidence[]` locating the support (page, box, and —
for chart questions — the cited chart element). Unanswerable items have `answer` =
`"Not in file"` and no evidence.

## Document type ids

| id | key | | id | key | | id | key |
|----|-----|---|----|-----|---|----|-----|
| 0 | urla_1003 | | 9 | schedule_c | | 18 | purchase_contract |
| 1 | form_1008 | | 10 | bank_stmt_checking | | 19 | purchase_addendum |
| 2 | loan_estimate | | 11 | bank_stmt_combo | | 20 | options_addendum |
| 3 | closing_disclosure | | 12 | brokerage_stmt | | 21 | email_correspondence |
| 4 | paystub | | 13 | check_image | | 22 | letter_of_explanation |
| 5 | w2 | | 14 | deposit_receipt | | 23 | gift_letter |
| 6 | voe | | 15 | credit_report | | 24 | insurance_declaration |
| 7 | form_1040 | | 16 | du_findings | | 25 | loan_summary |
| 8 | schedule_1 | | 17 | lpa_feedback | | 26 | filler |
