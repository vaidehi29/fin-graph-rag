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
| Entity & relation extraction | Done - phase 3 (`fin_graph_rag/entity_extractor.py`). 214 financial facts parsed from tables (no LLM); 247 risks, 78 executive/board entries, 73 competitor mentions via Gemini free tier (46 calls), every LLM item validated against a verbatim quote. |
| Knowledge graph (Neo4j) | Done - phase 3 (`fin_graph_rag/graph_builder.py`, `run_ingestion.py`). AuraDB Free; MERGE-only, idempotent. Phase 4: + 1,432 `TextChunk` nodes with bge-small vectors and the `chunk_embeddings` vector index (ONLINE). Totals: 2,007 nodes, 4,612 relationships. |
| Retrieval | Done - phase 5 (`fin_graph_rag/hybrid.py`): graph facts (`cypher_retriever.py`) + text chunks (Neo4j vector index fused with BM25). On 39 eval cases: answer supported by graph or top-5 text = 37/39 (text alone 27/39); top-10 = 38/39. |
| Answer generation | Done - phase 6 (`fin_graph_rag/answer.py`): Gemini free tier, pipeline.txt system prompt verbatim, citations and numbers checked against the retrieved context. 44 questions: 41/41 answers with a valid citation, 0 invalid citations, 0 unverified numbers, 3/3 missing-evidence questions declined; manual review 38/39 correct + 1 partial. |
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
python -m fin_graph_rag.run_ingestion --dry-run   # extract facts/entities, print, no DB writes
python -m fin_graph_rag.run_ingestion        # load into Neo4j (needs NEO4J_* and GEMINI_API_KEY in .env)
python -m pytest -q                          # real-corpus tests skip if data is absent
python -m fin_graph_rag.hybrid "Compare Apple and Microsoft revenue"   # graph facts + text + sources
python -m fin_graph_rag.hybrid --evaluate -k 5    # writes eval/results/*_hybrid-graph_*
python -m fin_graph_rag.answer "What was Microsoft's net income in fiscal 2024?"   # cited answer
python -m fin_graph_rag.answer --evaluate         # 42 eval cases + 2 pipeline samples (~44 Gemini calls)
FGR_RUN_INTEGRATION=1 python -m pytest -q -k integration   # read-only check against live Neo4j
```

## Knowledge graph (phase 3)

**Numbers from tables, text facts from Gemini.** Financial figures are read
directly from the parsed statement and MD&A tables, so each value is exactly
the figure on its cited page - no LLM reads numbers. Gemini (free tier,
`gemini-3.5-flash-lite`) extracts only risks, executives and competitors from
Item 1A, Item 1 / Item 10 and the signature page - 46 calls, inside one day's
free quota, cached so re-runs cost nothing.

Schema (all writes `MERGE`, uniqueness constraints per `pipeline.txt`):

```
(:Company)-[:FILED]->(:Report {company, year})
(:Company)-[:HAS_METRIC]->(:FinancialMetric {metric, fiscal_year, basis, unit, value, segment, product, conflict})
(:Company)-[:HAS_SEGMENT]->(:BusinessSegment)-[:HAS_METRIC]->(:FinancialMetric)
(:Company)-[:SELLS]->(:Product)-[:HAS_METRIC]->(:FinancialMetric)
(:FinancialMetric)-[:REPORTED_IN {page, table_id, chunk_id, row_text, value}]->(:Report)
(:Company)-[:FACES_RISK]->(:Risk)-[:IN_CATEGORY]->(:RiskCategory);  (:Risk)-[:DISCLOSED_IN {page, chunk_id, quote}]->(:Report)
(:Company)-[:HAS_EXECUTIVE | BOARD_MEMBER {fiscal_year, role, page, quote}]->(:Person)
(:Company)-[:COMPETES_WITH {fiscal_year, context, page, quote}]->(:Company | :Offering)
```

**Table facts** (214): revenue, gross margin, operating income, net income,
R&D, SG&A, tax provision, diluted EPS, operating cash flow, buybacks,
dividends, total assets - company-level, per segment (Apple geographic,
Microsoft P&BP / IC / MPC) and per Apple product category. Rules: only an
allow-list of statement/MD&A tables (statement titles matched in CAPITALS, so
Microsoft's Activision "net impact" table - "...consolidated income statements
since..." - is excluded); values must line up one-to-one with the table's year
columns; units only from a stated "in millions" line; "Diluted" share counts
are not mistaken for EPS. GAAP, non-GAAP and pro forma are separate facts.
Checked: 37/37 hand-verified eval figures extracted with the exact value.

**Conflicts are kept, not resolved silently.** 12 facts are flagged
`conflict=true` - all Microsoft segment figures for FY2023/FY2024 that the
FY2025 report recast after a segment reorganisation (e.g. FY2023 Intelligent
Cloud revenue: 87,907 in the FY2023 report, 72,944 in the FY2025 report).
Each report's value stays on its `REPORTED_IN` edge; the node's `value` is the
figure from that fiscal year's own report.

**LLM output is validated, not trusted.** Each item must cite a chunk from its
batch and quote it verbatim (risks >= 4 words, competitors >= 3, executives
>= 2 - the person's name must also be in that chunk); risk categories come
from a fixed list of 12. Rejected in the final run: 6 risks, 10 executive
entries, 1 competitor. Competitors are classified as company / product /
open-source project; only companies become `Company` nodes (Linux, MySQL, PHP
are `Offering`s). Microsoft's FY2025 10-K names no competitor companies (it
mentions only Linux and Unix), so that year has none - verified against the text.

Example queries that search could not answer well:

```cypher
// E39: which company had higher FY2025 net income?  (search: evidence at rank 43)
MATCH (c:Company)-[:HAS_METRIC]->(m:FinancialMetric {metric:'net_income', fiscal_year:2025, basis:'gaap'})
WHERE m.segment IS NULL RETURN c.name, m.value ORDER BY m.value DESC
// Apple CFO by year, with source page
MATCH (:Company {id:'apple'})-[h:HAS_EXECUTIVE]->(p) WHERE h.role CONTAINS 'Chief Financial'
RETURN h.fiscal_year, p.name, h.page ORDER BY h.fiscal_year
```

Known limitations: "Cisco" and "Cisco Systems" are separate nodes (not merged
on a guess); LLM categories and titles for risks are model judgements (the
quote is verified, the label is not); Apple's 10-K lists only signing officers,
not its full executive team (that is in its proxy statement, not indexed).

## Text chunks and vector index in Neo4j (phase 4)

`run_ingestion` also loads every chunk as a `TextChunk` node with its
embedding and builds the `chunk_embeddings` vector index (cosine).
`--skip-embed` skips this step.

```
(:TextChunk {chunk_id, text, company, year, page_num, page_end, chunk_index, section,
             in_exhibits, has_table, qualifiers, overlap_chars, token_count, embedding_model, embedding})
(:TextChunk)-[:SOURCED_FROM]->(:Company)      (:TextChunk)-[:PART_OF]->(:Report)
```

**Free, no re-embedding.** `pipeline.txt` specifies paid OpenRouter
`text-embedding-3-small` (1536-d). Instead the vectors are the ones already
built and evaluated locally - `BAAI/bge-small-en-v1.5` with the context header
(384-d) - read via the same staleness checks as local search. The index
dimension is taken from those vectors; an existing index with a different
dimension is reported, not overwritten.

Measured 2026-10-28: 1,432 `TextChunk` nodes, index `ONLINE`, 100% populated,
384 dimensions (`pipeline.txt` acceptance: >= 500 nodes, ONLINE - checked by
the integration test). Neo4j vector search vs the local index on all 42 eval
questions: identical top-1 chunk 42/42, top-10 overlap 9.86/10 on average
(min 8; Neo4j's index is approximate). Neo4j reports scores as
`(1 + cosine) / 2` (verified to within 0.00003), so 0.85 there = 0.70 cosine.
Re-running is idempotent (same node and relationship counts).

## Hybrid graph + text retrieval (phase 5)

`hybrid_search(driver, query)` returns
`{"graph_facts": [...], "text_chunks": [...], "sources": [...]}`:

- **Graph facts** (`cypher_retriever.cypher_search`): keyword detection of the
  company (Apple / Microsoft, plus the alias "iPhone maker"), fiscal years,
  metric, segment/product, basis and intent - metric, segment, risk, person
  (CEO / CFO / board), competitor - then parameterised Cypher. Two companies
  named = side-by-side results. GAAP by default; pro forma / non-GAAP only when
  the question asks. Conflicting reports are shown in the fact text. An
  unrecognised company (e.g. NVIDIA) returns no facts.
- **Text chunks** (`vector_retriever.vector_search` + BM25, fused with RRF):
  Neo4j vector index candidates fused with keyword ranking - the combination
  that measured best. The query model is read from the stored chunks, so a
  model mismatch fails loudly. If the question names one company, other
  companies' chunks are dropped; if it names two, results alternate between them.
- **Sources**: `"<Company> Annual Report FY<year>, p.<page>"`, graph facts first,
  de-duplicated - always with a page (`pipeline.txt` acceptance).

Measured 2026-10-28 on 42 cases (39 with an expected figure; `eval/results/*hybrid-graph*`).
"Supported" = a graph fact equals the expected figure, or a returned chunk has
a line with the metric label and figure:

| | top-5 | top-10 |
|---|---|---|
| text only (previous best: vector + header + BM25) | 27/39 | 31/39 |
| graph facts alone | 32/39 | 32/39 |
| **graph + text (hybrid_search)** | **37/39** | **38/39** |

The 3 missing-evidence questions (NVIDIA, FY2026) get no graph facts. Still
missing at top-5: Apple's FY2024 share count bought back (E24 - text only; the
graph returns the $94,949M cash outflow) and Microsoft's FY2024 gaming growth
(E36 - text only, chunk ranks 6-10). Graph misses are text facts by design
(employees, Microsoft Cloud revenue, tax rate) plus Activision's 5,729
contribution, deliberately kept out of company revenue.

Options measured, no difference on this set, so the simpler defaults stay:
de-duplicating chunks by (company, year, page) - `pipeline.txt` says by page,
but page 32 exists in every report - is off; filtering text to the named
company is on. A question-parsing bug found by the eval was fixed: "the
iPhone maker" also triggered the iPhone *product* filter.

## Answer generation (phase 6)

`ask(question, driver)` = `hybrid_search` -> `build_context` -> `generate_answer`,
returning `{answer, sources, model, confidence, insufficient_evidence,
invalid_citations, unverified_numbers}`.

- **Context**: graph facts first (exact figures, higher confidence), then text
  excerpts, each labelled `[Source: <Company> Annual Report FY<year>, p.<page>]`.
- **Model**: Gemini free tier (`gemini-3.5-flash-lite`, `FGR_ANSWER_MODEL`)
  instead of pipeline.txt's OpenRouter; temperature 0; responses cached, so
  re-asking costs no quota. The **pipeline.txt system prompt is used verbatim**;
  extra rules (state figures as written with company / fiscal year / unit, flag
  pro forma / non-GAAP / disagreeing reports, mark calculations "(calculated)",
  say when the reports do not answer) go in the user message.
- **Verification after generation** (a fluent answer is not evidence):
  every `[Source: ...]` must exactly match a retrieved source - anything else
  is returned in `invalid_citations`, never shown as a valid source; the
  shorthand "p.15, 17" is expanded and checked page by page. Numbers in the
  answer that appear nowhere in the context are returned in
  `unverified_numbers` (fiscal years and "(calculated)" sentences excepted).
- **Confidence** (pipeline.txt rule): `high` if graph facts were retrieved,
  otherwise `medium`.

**Measured 2026-10-28** (`eval/results/2026-10-28_195606_answers_n44.json`;
42 eval cases + pipeline.txt's "Compare Apple and Microsoft revenue" and
"What risks does Apple mention?"):

| check | result |
|---|---|
| answers with >= 1 valid citation (answerable + samples) | **41/41** |
| citations not in the retrieved context | **0** |
| answers with numbers not found in the context | **0** |
| missing-evidence questions (NVIDIA, FY2026) declined, no figures | **3/3** |
| expected figure present (automatic string check) | 36/39 |
| **manual review of the answers** | **38/39 correct, 1 partial** |
| "What risks does Apple mention?" risk categories (acceptance >= 2) | 6 |

Manual review of the 4 automatically flagged answers: E03 gives GAAP $72,361M
and labels $73,307M as non-GAAP (allowed by the case - the trap check is
over-strict); E36 computes gaming growth from $21,503M / $15,466M = +$6,037M,
39% "(calculated)" - correct; E37 gives 18% (FY2024 report) and 18.2% (FY2025
report's tax note) - both correct and cited; E24 says the share count is not
in the context and gives the $94,949M cash outflow - honest but partial
(known retrieval miss).

Known limitations: text citations use the chunk's first page, so a figure on
the second page of a two-page chunk is cited one page early (E36 cites p.130;
the figures are on p.131) - TODO: cite chunk page ranges (needs a full re-run,
~44 calls). Gemini free tier is slow at times (HTTP 503 "overloaded", retried):
the 44-question run took about 25 minutes. Answers are only checked
automatically for citations and numbers; wording and completeness need review.

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

**Larger local model: `BAAI/bge-base-en-v1.5`** (768-d, same vocabulary and
512-token limit, same inputs; measured 2026-10-28, `eval/results/*bge-base*`):

| support (39 cases) | @1 | @5 | @10 | MRR | E10-E42: @5 / @10 / MRR |
|---|---|---|---|---|---|
| bge-small vector + header | 11 | 24 | 28 | 0.430 | 18 / 20 / 0.403 |
| bge-base vector + header | 9 | 25 | 30 | 0.395 | 20 / 22 / 0.412 |
| **bge-small hybrid** | **16** | **27** | 31 | **0.538** | 21 / 24 / **0.502** |
| bge-base hybrid | 13 | 26 | **33** | 0.489 | 21 / **25** / 0.491 |

**No meaningful gain - bge-small stays the default.** Per case, bge-base
hybrid vs bge-small hybrid is better on 11, worse on 9, equal on 19 (noise).
bge-base finds 2 more supporting chunks in the top 10 but ranks fewer first
(13 vs 16) - and rank 1-5 matters most when only a few chunks go to the
answer step. It costs 2x: 457 s vs 217 s to embed on CPU, 768-d vs 384-d index.

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
