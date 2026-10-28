# Financial Graph RAG

Answers questions about company annual reports (Apple and Microsoft, fiscal
years 2023–2025) using retrieved evidence and a knowledge graph. Agent rules
are in [CLAUDE.md](CLAUDE.md).

## Status

| Component | State |
|---|---|
| PDF extraction | Done (`fin_graph_rag/pdf_extractor.py`). Run on all 6 reports: 682 pages, 1 empty. |
| Preprocessing / chunking | Done - phase 1 (`fin_graph_rag/preprocess.py`). 617 non-empty pages → 1,411 chunks, all ≤ 500 tokens (bert-base-uncased; median 288); 375 tables (365 whole, 10 split into labelled parts). |
| Embeddings / index | Not started |
| Entity & relation extraction | Not started |
| Knowledge graph (Neo4j) | Not started |
| Retrieval / answer generation | Not started |
| Evaluation | Unit tests + real-corpus checks; 9 eval cases defined in `eval/cases.jsonl` (not yet runnable - no retrieval). |

## Setup

```bash
py -3.11 -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
cp .env.example .env            # optional; defaults work without it
```

## Data

Source PDFs are **not** committed. Place them in `data/raw/` named
`<company>_<fiscal_year>.pdf`:

```
apple_2023.pdf  apple_2024.pdf  apple_2025.pdf
microsoft_2023.pdf  microsoft_2024.pdf  microsoft_2025.pdf
```

The year is the company's **fiscal** year (Apple: ends late September;
Microsoft: ends June 30). Files with other names are skipped and reported.

## Run

```bash
python -m fin_graph_rag.pdf_extractor        # reads data/raw, writes data/extracted
python -m fin_graph_rag.preprocess           # reads data/extracted, writes data/processed
python -m pytest -q                          # real-corpus tests skip if data is absent
```

Extraction outputs:

- `data/extracted/pages.jsonl` — one record per page: `doc_id`, `company`,
  `fiscal_year`, `source_file`, `page_number` (1-based), `text`,
  `char_count`, `is_empty`.
- `data/extracted/extraction_report.json` — per-document page counts, empty
  pages, page-level errors, SHA-256 of each source file, and status
  (`ok` / `partial` / `failed`).

The command exits non-zero if no document was extracted or any document failed.

## Chunking (phase 1)

`data/processed/chunks.jsonl` - one record per chunk with `chunk_id`,
`doc_id`, `company`, `fiscal_year` (of the **report**), `page_start`/`page_end`
(PDF pages), `printed_pages`, `section` (e.g. `Item 8`), `in_exhibits`,
`tables`, `qualifiers`, `oversized`, `char_count`, `token_count`, `text`.
`chunking_report.json` holds the config (including the tokenizer actually
used), per-document counts and the list of qualifier-flagged tables.

**Size limits are in tokens**, counted with the `bert-base-uncased` WordPiece
tokenizer (the vocabulary of common 512-token embedding models: all-MiniLM,
BGE-small/base, E5-small/base; override with `FGR_TOKENIZER`). Every chunk is
≤ `MAX_TOKENS` = 500 content tokens, so with `[CLS]`/`[SEP]` it fits a
512-token model with headroom for a prefix like E5's `passage: `. Prose
targets `TARGET_TOKENS` = 350. Character budgets were dropped because numbers
tokenize poorly (`383,285` → 4 tokens): the earlier 2,500-char limit let 73
table chunks reach up to 825 tokens. If the tokenizer can't be loaded
(offline), a conservative chars/2 estimate is used and recorded in the report.

Measured (current run): median 288 tokens, p90 420, max 500; table chunks
median 422.

Rules:

- **Cleaning:** removes the PDFCrowd watermark (Microsoft FY24/FY25 PDFs were
  printed from HTML), page-number footers and `PART II / Item 8` running
  headers (used to set `section`). Pages after the SIGNATURES page are
  flagged `in_exhibits` (kept, not dropped).
- **Tables are kept together.** A table = run of numeric rows, including its
  title, unit line (`(In millions…)`), intro sentence ending in `:`, and
  footnotes. Documents are processed as one line stream, so tables split by a
  PDF page break stay together.
- **Large tables** (over 500 tokens; currently 10: cash-flow statements, a
  balance sheet, debt tables, Apple FY25 segment table) are split after
  complete rows, preferring sub-section labels such as `Financing activities:`.
  Every part repeats the title / unit / year-header lines and begins with a
  **generated** label, e.g.
  `[Table part 2/3 | FY2024, FY2023, FY2022 | CONSOLIDATED STATEMENTS OF CASH FLOWS]`.
  The label is not report text. Parts share a `table_id` so the full table
  can be reassembled.
- **Stacked tables** (one block per year under one title, e.g. Apple's segment
  table "2025 … 2024 … 2023 …") are split between year blocks where possible.
  A part starting inside a block repeats *that* block's year heading and
  column headers, and its label lists only the years it contains.
- **Prose** is packed by paragraph up to `TARGET_TOKENS`; over-long text with
  no sentence ends (e.g. exhibit index lists) is cut at line boundaries into
  evenly sized pieces. A short trailing unit (e.g. a risk-factor heading) is
  moved into the next chunk so it stays with the text it introduces; leftover
  fragments under `MIN_TOKENS` are merged into a same-section neighbour.
  Chunks never cross a section or exhibit boundary.
- **Each table records** `table_id`, `part`/`n_parts`, `title`,
  `column_years` (from its year-header row) and `qualifiers`
  (`pro_forma`, `unaudited`, `non_gaap`).
- Chunking is lossless: removing generated labels and repeated headers gives
  back the cleaned text exactly (tested on the real corpus).

### Misleading figures found (see `eval/cases.jsonl`)

| Where | Figure | Why it misleads | Reported figure |
|---|---|---|---|
| MSFT FY24 10-K p108 | Net income 88,308; revenue 247,442; diluted EPS 11.82 | Unaudited **pro forma**, as if Activision closed July 1, 2022 | 88,136; 245,122; 11.80 (p80) |
| MSFT FY25 10-K p94 | Same pro forma table | In the FY**2025** report but columns are FY2024/FY2023 | FY25 net income 101,832 (p69) |
| MSFT FY23 10-K p43, p49 | Adjusted net income 73,307 | **Non-GAAP** (excludes severance/impairment charges) | 72,361 (p58) |
| MSFT FY24 10-K p108 | Revenue 5,729 | Activision's contribution since acquisition, not total revenue | 245,122 |
| Segment tables (both companies) | e.g. Apple Americas operating income | Segment-level, not company total | - |

A chunk's `fiscal_year` is the report's year; use a table's `column_years`
to know which year a figure belongs to.

Known limitations: titles are heuristic (33 non-exhibit tables have no
detected title; a few include a stray wrapped prose line); prose cut at line
boundaries may break mid-sentence (no overlap between chunks); 46 chunks are
under 50 tokens, all one-line Items such as "Item 1B … None."; token limits
assume a BERT-WordPiece embedding model - re-run with `FGR_TOKENIZER` set if a
different model is chosen; chunk size not yet tuned against retrieval
results; no OCR.

## Decisions

- **pdfplumber** for extraction — MIT-licensed, page-level, keeps table row
  layout reasonably well. Alternatives: pypdf (weaker layout), PyMuPDF
  (faster but AGPL). Revisit if extraction quality on the real reports is poor.
- Company/year come from the filename, not parsed from PDF text, so
  provenance is deterministic.
- **tokenizers** (Apache-2.0, HF) to count chunk tokens exactly. Downloads the
  `bert-base-uncased` vocabulary once (~700 KB, cached). Alternative: a
  character estimate - rejected because it under-counted numeric tables.
