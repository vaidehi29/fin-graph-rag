# Financial Graph RAG

Answers questions about company annual reports (Apple and Microsoft, fiscal
years 2023–2025) using retrieved evidence and a knowledge graph. Agent rules
are in [CLAUDE.md](CLAUDE.md).

## Status

| Component | State |
|---|---|
| PDF extraction | Done (`fin_graph_rag/pdf_extractor.py`). Run on all 6 reports: 682 pages, 1 empty. |
| Preprocessing / chunking | Done - phase 1 (`fin_graph_rag/preprocess.py`). 617 non-empty pages → 1,432 chunks, all ≤ 500 tokens incl. overlap (bert-base-uncased); 375 tables (365 whole, 10 split into labelled parts); 20-30 token overlap between prose chunks. |
| Embeddings / index | Local (`fin_graph_rag/embeddings.py`): `BAAI/bge-small-en-v1.5` on CPU, 1,432 × 384-d flat cosine index, chunks embedded with a context header. Gemini / Hugging Face free tiers not yet compared. |
| Entity & relation extraction | Not started |
| Knowledge graph (Neo4j) | Not started |
| Retrieval | Hybrid (vector + BM25, RRF) recommended, in `fin_graph_rag/retrieval.py`. On 39 eval cases: support@5 = 27/39, support@10 = 31/39 (vector + header: 24, 28). |
| Answer generation | Not started |
| Evaluation | Unit tests + real-corpus checks; 42 eval cases in `eval/cases.jsonl` (every citation verified against page text), retrieval scored in `eval/results/` (one file per run, never overwritten). |

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
python -m fin_graph_rag.embeddings build     # embeds chunks (with context header) -> data/index/<model>-hdr/
python -m fin_graph_rag.embeddings search "Microsoft net income fiscal 2024"
python -m fin_graph_rag.embeddings evaluate -k 5   # writes eval/results/
python -m fin_graph_rag.retrieval search "..." --mode hybrid   # or bm25 / vector
python -m fin_graph_rag.retrieval evaluate -k 5 --mode hybrid
python -m pytest -q                          # real-corpus tests skip if data is absent
```

## Embeddings (local baseline)

**Free only:** the model is downloaded once from Hugging Face (~130 MB) and
then runs offline on CPU - no API key, no per-call cost, no report text sent
anywhere. `build` refuses to run if any chunk exceeds the model's 512-token
limit (checked with the model's own tokenizer: max 501 incl. `[CLS]`/`[SEP]`),
so nothing is silently truncated. Queries get BGE's recommended instruction
prefix; passages don't. The index records the SHA-256 of `chunks.jsonl`, and
search refuses a stale index.

**Context header (default).** Before embedding, each chunk is prefixed with
e.g. `Microsoft Corporation | FY2024 Form 10-K | Item 8 Financial Statements
and Supplementary Data | Table: INCOME STATEMENTS (FY2024, FY2023, FY2022)`
(qualifiers such as `[pro forma, unaudited]` included). Only the embedded
input changes - stored chunk text and citations do not. If the header would
push a chunk past 512 tokens it is shortened (1,380 full, 42 without table,
9 company/year only, 1 without header). `--no-header` uses the plain index.

**Retrieval results** (measured 2026-10-28, `eval/results/`; retrieval only,
no answers generated; first 8 cases with evidence - superseded by the 42-case
results below, kept for the record; indicative, not statistically
meaningful):

| | plain top-5 | plain top-10 | **header top-5** | **header top-10** |
|---|---|---|---|---|
| support@k - a chunk with the metric label + correct figure is retrieved | 5/8 | 6/8 | **6/8** | **8/8** |
| hit@k (strict) - the specific cited line is retrieved | 2/8 | 3/8 | 2/8 | 6/8 |
| a pro forma trap ranks above the cited evidence | 2 | 2 | **0** | **0** |

The Microsoft income-statement lookups (E01, E02) moved from rank 47-60 to
rank 7-8. Caveats: the header was designed after seeing those failures, so
these 9 cases may flatter it; on 3 unscored held-out questions it was better
(R&D expense) or equal (share repurchases, supply-chain risks), never worse.
An earlier header version that named "ITEM 8. FINANCIAL STATEMENTS" instead
of the table scored support@5 = 7/8; it was fixed for correctness, not tuned
to the eval. Encoding all chunks takes ~220 s on CPU.

## Keyword and hybrid search (experiment)

`fin_graph_rag/retrieval.py` adds BM25 keyword search (own ~40-line
implementation, standard k1 = 1.5, b = 0.75, no new dependency) over the same
context header + chunk text, and hybrid search that merges the vector and BM25
rankings with Reciprocal Rank Fusion (k = 60, the published default; not
tuned). `--mode vector|bm25|hybrid` on `search` and `evaluate`.

**Results on 42 cases** (measured 2026-10-28; 39 with evidence, 3 missing-
evidence; retrieval only, no answers generated; `eval/results/*_n42_*`).
`support@k` = a chunk with the metric label + correct figure is in the top k;
MRR = mean of 1/rank of that chunk.

| | support@1 | support@5 | support@10 | MRR | E10-E42 only (31): support@5 / @10 / MRR |
|---|---|---|---|---|---|
| plain vector | 9 | 18 | 22 | 0.347 | 13 / 16 / 0.324 |
| vector + header | 11 | 24 | 28 | 0.430 | 18 / 20 / 0.403 |
| BM25 only | 12 | 21 | 24 | 0.426 | 16 / 18 / 0.392 |
| **hybrid (vector + header + BM25, RRF)** | **16** | **27** | **31** | **0.538** | **21 / 24 / 0.502** |

**Hybrid is now the recommended retrieval** (`fin_graph_rag.retrieval`, default
`--mode hybrid`). It is best on every measure, including the 31 cases written
after the header and hybrid were designed. Per case vs vector + header it is
better on 19, worse on 10, equal on 10 - a consistent direction but not
statistically conclusive (sign test p ≈ 0.14). Pro forma traps never rank above
the evidence in either header-based config (plain vector: 2).

This **reverses the earlier 8-case result** ("hybrid no better"): one case
moving changed that score by 12.5 points, so it was noise. Those earlier
numbers are reproduced exactly by the E01-E09 subset of these runs.

Remaining misses at top-10 (hybrid): Apple Greater China / Japan FY2025 (the
adjacent prose chunk ranks #1, the segment table #29-40 - passing neighbouring
chunks to the answer step may help, untested); Apple FY2025 buybacks (question
ambiguous: retrieval finds the $89.3B share-repurchase text, the case expects
the $90,711M cash-flow line); E01, E18, E36 (ranks 11-12); cross-year and
cross-company comparisons E38/E39, which need two documents for one query.
Untested idea: graph lookups for exact metrics (next phase).

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
`tables`, `qualifiers`, `oversized`, `char_count`, `token_count`,
`overlap_chars`, `overlap_from`, `text`.
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

**Overlap:** a chunk that starts with prose begins with the last 20-30 tokens
(`OVERLAP_MIN_TOKENS`/`OVERLAP_MAX_TOKENS`) of the previous chunk in the same
section, starting at a sentence start when one falls in that window, so text
cut at a boundary stays retrievable. `text[:overlap_chars]` is the repeated
part (copied from `overlap_from`); `text[overlap_chars:]` is the chunk's own
text - use that for citations and de-duplication. Chunks starting with a table
get no overlap (they already start with the table title). Packing reserves
30 tokens so overlap never pushes a chunk past 500.

Measured (current run): 1,087 of 1,432 chunks carry overlap (median 30
tokens); max chunk 499 tokens.

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
detected title; a few include a stray wrapped prose line); overlap often
starts mid-sentence when no sentence start falls in the 20-30 token window
(298 of 1,087 start at a sentence); 46 chunks are
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
