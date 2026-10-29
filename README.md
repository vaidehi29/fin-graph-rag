# Financial Graph RAG

Answers questions about company annual reports (Apple and Microsoft, fiscal
years 2023–2025; IBM and NVIDIA added through phases 1-4 on 2026-10-29 - see
[Adding IBM and NVIDIA](#adding-ibm-and-nvidia-phases-1-4-2026-10-29)) using
retrieved evidence and a knowledge graph. Agent rules are in [CLAUDE.md](CLAUDE.md).

## Status

| Component | State |
|---|---|
| IBM + NVIDIA | Phases 1-8 done 2026-10-29 (12 reports in total; 109 unit + 8 integration tests pass, golden set 6/6) - see [Adding IBM and NVIDIA](#adding-ibm-and-nvidia-phases-1-4-2026-10-29). The rows below describe the Apple/Microsoft baseline. |
| PDF extraction | Done (`fin_graph_rag/pdf_extractor.py`). Run on all 6 reports: 682 pages, 1 empty. |
| Preprocessing / chunking | Done - phase 1 (`fin_graph_rag/preprocess.py`). 617 non-empty pages → 1,432 chunks, all ≤ 500 tokens incl. overlap (bert-base-uncased); 375 tables (365 whole, 10 split into labelled parts); 20-30 token overlap between prose chunks. |
| Embeddings / index | Local (`fin_graph_rag/embeddings.py`): `BAAI/bge-small-en-v1.5` on CPU, 1,432 × 384-d flat cosine index, chunks embedded with a context header. Gemini / Hugging Face free tiers not yet compared. |
| Entity & relation extraction | Done - phase 3 (`fin_graph_rag/entity_extractor.py`). 286 financial facts parsed from tables (no LLM); 247 risks, 78 executive/board entries, 73 competitor mentions via Gemini free tier (46 calls), every LLM item validated against a verbatim quote. |
| Knowledge graph (Neo4j) | Done - phase 3 (`fin_graph_rag/graph_builder.py`, `run_ingestion.py`). AuraDB Free; MERGE-only, idempotent. Phase 4: + 1,432 `TextChunk` nodes with bge-small vectors and the `chunk_embeddings` vector index (ONLINE). Totals: 2,007 nodes, 4,612 relationships. |
| Retrieval | Done - phase 5 (`fin_graph_rag/hybrid.py`): graph facts (`cypher_retriever.py`) + text chunks (Neo4j vector index fused with BM25). On 95 eval cases (89 answerable): answer supported by graph or top-5 text = 83/89 (93%), top-10 = 86/89 (text alone 60/89, 74/89). |
| Answer generation | Done - phase 6 (`fin_graph_rag/answer.py`): Gemini free tier, pipeline.txt system prompt verbatim, citations and numbers checked against the retrieved context. 95 eval questions + 2 samples: 0 invalid citations, 0 unverified numbers, 0 wrong figures found; manual review of the 89 answerable: 85 correct, 1 partial, 3 declined although the reports contain the answer; 6/6 missing-evidence questions handled correctly. |
| Web UI | Done - phase 7 (`app.py`, `fin_graph_rag/graph_view.py`): Streamlit Q&A with confidence badge, sources and history; Knowledge Graph tab rendered with pyvis. Checked end to end headless (Streamlit AppTest, live Neo4j). |
| Testing & validation | Done - phase 8: 90 unit tests (offline) + 8 integration tests (`pytest -m integration`, live Neo4j / Gemini / local model) all pass; golden set 4/4; 95 eval cases in `eval/cases.jsonl` (every citation verified against page text), results in `eval/results/` (one file per run, never overwritten). |

## Adding IBM and NVIDIA (phases 1-4, 2026-10-29)

Only the new companies were processed; Apple and Microsoft outputs were not
re-run. Their lines in `pages.jsonl`, `chunks.jsonl`, `facts.jsonl` and
`text_entities.jsonl` and their 1,432 index vectors are byte-identical to before
(checked against a backup), and every chunking/fact rule change below was
re-run on Apple/Microsoft in memory to confirm it reproduces their existing
chunks exactly. Measured results:

| Step | IBM (FY2023-25) | NVIDIA (FY2023-25) |
|---|---|---|
| Source check | full 10-K filings (cover, Items 1A/7/8, signatures p.26) + Exhibit 13 | standalone 10-Ks, cover p.1, signed p.91 / 84 / 86 |
| Pages extracted (empty) | 345 / 270 / 272 (3 blank) | 169 / 96 / 130 (0) |
| Chunks (max tokens) | 871 / 771 / 615 (500) | 436 / 267 / 371 (500) |
| Table facts (company level) | 80 | 94 |
| Gemini entities (45 calls, free tier) | 78 risks, 24 executive entries, 51 competitors | 173 risks, 15 executive entries, 49 competitors |
| Neo4j | 80 HAS_METRIC, 2,257 TextChunks | 94 HAS_METRIC, 1,074 TextChunks |

Graph after loading: 5,886 nodes, 12,852 relationships; `chunk_embeddings` index
ONLINE, 100% populated, 4,763 vectors (384-d). Embedding the 3,331 new chunks
took 341 s on CPU (`embeddings append`).

**What had to change** (each rule is commented in the code with the case that
needed it):

- *Filing layout.* IBM's 10-K incorporates Items 7 and 8 from **Exhibit 13**
  (the annual report), which follows the signatures page. Those pages are now
  report content, not exhibits; their running header ("Consolidated Balance
  Sheet", "Management Discussion") becomes the chunk `section`, e.g.
  `Exhibit 13 Consolidated Income Statement`. The signatures page must now
  contain "Pursuant to the requirements of Section 13 or 15(d)" (IBM's table of
  contents has a bare "SIGNATURES" line; NVIDIA writes "Signatures"). The
  "Table of Contents" link line on EDGAR-printed pages is removed.
- *Oversized lines.* Some IBM benefit-plan exhibits extract a whole page as one
  line (up to ~3,000 tokens); such lines are split at sentence ends (else words).
- *Tables.* Year rows like "For the year ended December 31: Notes 2025 2024";
  IBM Notes-column letters ("Total revenue C 67,535"); long-label balance-sheet
  rows with a parenthetical; IBM's equity block split from the balance sheet by
  three heading lines; NVIDIA rows whose label wraps above the figures.
- *Metric labels.* "Gross profit" -> `gross_margin` (same measure);
  IBM "Research, development and engineering" -> R&D (FY2025 report renames it,
  same figures); IBM diluted EPS = "Assuming dilution / Total"; IBM equity =
  "Total IBM stockholders' equity" (before noncontrolling interests). NVIDIA's
  capex line includes intangibles, so it is a separate metric
  (`purchases_of_property_equipment_and_intangibles`), as with Microsoft's
  "depreciation, amortization, and other". Mixed-case statement titles must
  equal a title part exactly, so NVIDIA's note "Consolidated Statements of
  Income include stock-based compensation..." is not read as the statement.
- *People.* Person keys fold accents ("Nicolás"/"Nicolas A. Fehring" = one node).

Spot-checked against the statements: NVIDIA revenue 26,974 / 60,922 / 130,497
and net income 4,368 / 29,760 / 72,880 (FY2023-25); IBM revenue 61,860 / 62,753 /
67,535, net income 7,502 / 6,023 / 10,593, diluted EPS 8.14 / 6.43 / 11.17. Two
new conflicts, both genuine: NVIDIA's FY2025 report restates FY2023/FY2024
diluted EPS for the June 2024 10-for-1 split (1.74 -> 0.17, 11.93 -> 1.19);
both values are kept on their `REPORTED_IN` edges.

**Known limitations / not done:**

- No segment or product facts for IBM/NVIDIA: their segment tables put segments
  in columns (and IBM recast its revenue categories in 2025), which the
  year-column parser does not read. Segment questions rely on text chunks.
- Competitor names are not merged on a guess: "AMD" / "Advanced Micro Devices"
  and "Huawei" / "Huawei Technologies Co." are separate nodes, and Microsoft's
  existing competitor node "IBM" is separate from the reporting company
  "International Business Machines Corporation" (merging would edit Microsoft's
  edges).
- IBM's income statement has no "operating income" line, so there is no IBM
  `operating_income` fact.
- Found while testing: the wrapped-row rule would also recover Microsoft's
  FY2023 income-statement R&D row (label and figures on separate lines; FY2021
  R&D 20,716 is currently missing from the graph). Not applied - Apple/Microsoft
  were deliberately not re-run.
- Phases 5-7 were then updated for IBM/NVIDIA - see below.

### Phase 5 (hybrid retrieval) for IBM and NVIDIA

- `cypher_retriever`: IBM ("IBM", "International Business Machines") and NVIDIA
  ("NVIDIA", "NVDA") are recognised - legal name and ticker only, no guessed
  descriptions. Question words added for the new labels: "gross profit",
  "sales, general and administrative", "research, development", "income before
  income tax" (the old pattern `taxes?` never matched "tax" - fixed). A capex
  question also returns NVIDIA's separate capex-incl.-intangibles metric.
- `hybrid.text_search`: text sources now use the graph's caption ("IBM Annual
  Report FY2024, p.175"), not `company.title()` ("Ibm ...").
- Eval: 16 new cases E96-E111 (IBM and NVIDIA: lookups, cross-year, the NVIDIA
  split-adjusted EPS conflict, IBM's "Research, development and engineering"
  label, an IBM-vs-NVIDIA comparison whose fiscal years end ~11 months apart, a
  missing-evidence FY2026 question, and the first two person questions). E40
  (NVIDIA data center revenue) was a missing-evidence case; it is now answerable
  (115,186, p.80) and was revised. Every case's evidence is checked on its page.
  `graph_supports` now also counts a non-numeric answer (a person and role) when
  a graph fact's text contains both; the 95 earlier cases are all numeric, so
  their scoring is unchanged.

Measured 2026-10-29 (`eval/results/2026-10-29_14*_n111_*`; retrieval only):

| | combined support@5 | combined support@10 |
|---|---|---|
| original 89 answerable cases, before IBM/NVIDIA | 83 | 86 |
| same 89 cases, with IBM/NVIDIA indexed | **83** | **86** |
| 16 new / revised cases | 14 | 15 |
| all 105 answerable cases | **97** | **101** |

Graph facts alone support 14 of the 16 new cases. Text-only support on the
original cases moved 60 -> 61 (@5) and 74 -> 71 (@10): the corpus is three times
larger, so BM25 statistics and the vector candidates change; graph facts cover
those cases, so combined support is unchanged. A company pre-filter before
taking candidates (`PREFILTER_BY_COMPANY`) was tried and did not help (96 vs 97
@5, 101 vs 101 @10), so it is off. Remaining misses on the new cases: E107 (IBM
Software segment revenue - text only, chunk not retrieved; no IBM segment facts
in the graph) and E40 at top-5 (found at rank 6-10; the graph offers NVIDIA
*total* revenue, which the answer step must not use for data center revenue).
`pipeline.txt`'s "AI chips demand" vector test now returns NVIDIA (integration
test updated); "Compare IBM and NVIDIA revenue" returns both companies' facts
with paged sources.

### Phases 6-7 (answers, UI) for IBM and NVIDIA

- `answer.cited_sources`: a combined citation "[Source: IBM Annual Report ...;
  NVIDIA Annual Report ...]" is now split (the split only recognised mixed-case
  names such as "Apple"), so it is not flagged as invalid.
- `answer.build_context`: when the evidence spans two or more companies, the
  context gets each company's fiscal-year convention, quoted from its FY2025
  report with a source (Apple p.4, Microsoft p.69, IBM p.175, NVIDIA p.57),
  plus an instruction to state each year-end. Reason: asked to compare IBM and
  NVIDIA revenue "in fiscal 2024", the model called both "the same fiscal year",
  though NVIDIA's ended January 28, 2024 and IBM's December 31, 2024. Single-
  company prompts are unchanged (5 of 111 eval cases have multi-company
  evidence: E39, E90, E91, E92, E109).
- `answer --evaluate --ids E40,E96-E111 [--samples ...]` answers a subset only,
  to save free-tier quota.
- UI: `config.COMPANIES` = Apple, Microsoft, IBM, NVIDIA (sidebar scope and graph
  tab); captions no longer say "Apple and Microsoft". The end-to-end app test
  now also asks an NVIDIA question and opens NVIDIA's graph.

Measured 2026-10-29 (`eval/results/2026-10-29_152604_answers_n19.json`,
`..._153139_answers_n7.json`, `..._153251_answers_n4.json`; 33 Gemini calls):

| check (17 new/revised cases + 2 samples) | result |
|---|---|
| invalid citations / answers with unverified numbers | **0 / 0** |
| expected figure present (automatic) | 14/16 |
| missing-evidence question declined without figures | 1/1 |
| trap (IBM basic EPS 6.53) used | 0/1 |
| **manual review, 16 answerable** | **13 correct, 1 partial, 2 wrongly declined, 0 wrong figures** |

Manual review: correct - E96-E97, E99-E106, E108, E109 (after the fiscal-year
note: both figures, each year-end cited), E111; partial - E98 (gives 11.93 and
says the FY2025 report states 1.19, but not that the 10-for-1 split explains it);
declined although the reports contain it - E40 (data center 115,186) and E107
(IBM Software 29,962), both the known retrieval misses. E100 gives the cash
outflow 33,706 and the text's "$34.0 billion" repurchase amount, both cited.
Apple/Microsoft comparisons re-answered with the note (E39, E90, E91 and
"Compare Apple and Microsoft revenue"): same figures, 3/3 present, now with
each year-end. E92 (no company named) still declines, as before.
UI verified 2026-10-29: `pytest -m integration` 8/8 (headless app: Q&A with a
cited NVIDIA answer, NVIDIA graph tab); `streamlit run app.py` passes the health
check. Not re-checked in a real browser this time.

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
ibm_2023.pdf  ibm_2024.pdf  ibm_2025.pdf
nvidia_2023.pdf  nvidia_2024.pdf  nvidia_2025.pdf
```

The year is the company's **fiscal** year (Apple: ends late September;
Microsoft: ends June 30; IBM: December 31; NVIDIA: last Sunday of January, so
`nvidia_2025.pdf` is the year ended January 26, 2025). Files with other names
are skipped and reported. Each file must be the Form 10-K itself (checked for
the IBM and NVIDIA files: SEC cover page, Items 1A/7/8, signatures page); IBM's
files are the full filing including Exhibit 13, which holds its MD&A and
financial statements.

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
# Adding companies without touching the others (outputs merged; other companies' lines kept byte for byte):
python -m fin_graph_rag.pdf_extractor --companies ibm nvidia
python -m fin_graph_rag.preprocess --companies ibm nvidia
python -m fin_graph_rag.embeddings append    # embeds only chunks not yet in the index
python -m fin_graph_rag.run_ingestion --companies ibm nvidia
python -m pytest -q                          # offline unit tests; integration tests are skipped
python -m fin_graph_rag.hybrid "Compare Apple and Microsoft revenue"   # graph facts + text + sources
python -m fin_graph_rag.hybrid --evaluate -k 5    # writes eval/results/*_hybrid-graph_*
python -m fin_graph_rag.answer "What was Microsoft's net income in fiscal 2024?"   # cited answer
python -m fin_graph_rag.answer --evaluate         # 42 eval cases + 2 pipeline samples (~44 Gemini calls)
streamlit run app.py                              # web UI at http://localhost:8501
python -m pytest tests/ -v --ignore-glob="*integration*"   # pipeline.txt unit command
python -m pytest -m integration -v           # live Neo4j + Gemini + local model (read-only)
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
(:Company)-[:COMPETES_WITH {fiscal_year, context, page, quote}]->(:Company:Competitor | :Offering)
// :Competitor (added 2026-10-29) = a competitor Company that is not one of the four reporting companies (no `id`)
```

**Table facts** (286): revenue, gross margin, operating income, net income,
R&D, SG&A, tax provision, diluted EPS, operating cash flow, buybacks,
dividends, total assets, and (added after the 95-case eval) income before
taxes, cash and cash equivalents, capital expenditures, depreciation and
amortization (Microsoft's "depreciation, amortization, and other" kept as a
separate metric - not the same measure), share-based compensation, total
current assets, total liabilities, shareholders' equity - company-level, per segment (Apple geographic,
Microsoft P&BP / IC / MPC) and per Apple product category. Rules: only an
allow-list of statement/MD&A tables (statement titles matched in CAPITALS, so
Microsoft's Activision "net impact" table - "...consolidated income statements
since..." - is excluded); values must line up one-to-one with the table's year
columns; units only from a stated "in millions" line; "Diluted" share counts
are not mistaken for EPS. GAAP, non-GAAP and pro forma are separate facts.
Checked: 37/37 hand-verified eval figures extracted with the exact value, plus
14/14 for the added metrics. The balance sheet's equity rows follow
"Commitments and contingencies / Shareholders' equity:" and arrive as an
untitled table without year columns; such a table inherits the previous
table's header only when on the same page, separated by <= 5 prose lines that
include a "...:" sub-header (affects exactly the 5 balance-sheet equity
sections; never guessed otherwise).

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

**95 cases** (89 with an expected figure; 53 added after phase 8, written
before any retrieval ran on them; `eval/results/*hybrid-graph_n95*`):

| | top-5 | top-10 |
|---|---|---|
| text only | 60/89 | 74/89 |
| graph facts alone (14 metrics) | 57/89 | 57/89 |
| graph + text, 14 metrics | 77/89 (87%) | 84/89 |
| graph facts alone (+9 metrics) | 70/89 | 70/89 |
| **graph + text, +9 metrics (current)** | **83/89 (93%)** | **86/89 (97%)** |

The new questions were harder (40/50 at top-5 before the added metrics, 46/50
after; the original 39 stay at 37/39), so 93% is the more representative
figure. Remaining top-5 misses: five text-only facts that rank too low (Apple
FY2024 buyback share count E24, Microsoft FY2024 gaming growth E36, Apple
employees FY2023/FY2025 E63/E64, Azure FY2025 growth E86) and E92 ("net income
in fiscal 2025?", no company named - the evaluation calls retrieval without the
UI's company scope). Two missing-evidence questions retrieve related facts
(iPhone units -> iPhone revenue; quarterly revenue -> annual revenue); only the
answer evaluation can show whether the answer declines correctly.

Earlier, on the first 42 cases (39 with an expected figure):
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

**Measured again on all 95 questions** (2026-10-28,
`eval/results/2026-10-28_233526_answers_n97.json`; ~55 new Gemini calls, no
rate-limit errors):

| check | result |
|---|---|
| answerable answers with >= 1 valid citation | 88/91 (the 3 without are "not enough information" answers) |
| citations not in the retrieved context | **0** |
| answers with numbers not found in the context | **0** |
| expected figure present (automatic string check) | 81/89 |
| missing-evidence questions declined without figures (automatic) | 5/6 |
| **manual review, 89 answerable** | **85 correct, 1 partial, 3 wrongly declined, 0 wrong figures** |

Manual review of the 8 automatically flagged answerable cases: correct - E36
(gaming +$6,037M / 39% calculated), E37 (18% and 18.2%, both cited), E86 (Azure
+34%), E88 (Activision $75.4 billion = $75,408 million); partial - E24 (share
count not retrieved; gives the $94,949M cash outflow); declined although the
reports contain it - E63 / E64 (Apple employee counts: text chunk not
retrieved) and E92 ("net income in fiscal 2025?" with no company - the
evaluation calls `ask()` without the UI's company scope). E93 (iPhone units)
declines and gives iPhone *net sales*, labelled as such - allowed by the case.
Spot-checked trap/conflict answers were right: recast segment figures give both
reports' values (E82, E83), non-GAAP EPS is labelled (E89), GAAP EPS is not
confused with it (E71). Every failure is a "not enough information" answer,
not a wrong number. Scoring bug fixed during this review: a decline that
repeats the question's fiscal year ("fiscal 2024") was counted as giving a figure.

Known limitations: text citations use the chunk's first page, so a figure on
the second page of a two-page chunk is cited one page early (seen in 2 of 97
answers: E36 cites p.130 and E61 p.25 for figures on p.131 and p.26) - TODO:
cite chunk page ranges (needs a full answer re-run, ~97 calls). Gemini free tier is slow at times (HTTP 503 "overloaded", retried):
the 44-question run took about 25 minutes. Answers are only checked
automatically for citations and numbers; wording and completeness need review.

## Web UI (phase 7)

`streamlit run app.py` (needs `NEO4J_*` and `GEMINI_API_KEY` in `.env`).

- **Sidebar**: company multiselect (Apple, Microsoft - used as the scope when
  a question names no company), answer-model dropdown (free-tier Gemini models
  from `fin_graph_rag/config.py`), and a toggle that hides answers not backed
  by graph facts.
- **Q&A tab**: question box + Search -> `fin_graph_rag.answer.ask` (the same
  code the evaluation measured). Shows the answer, a confidence badge (green =
  high: graph facts; yellow = medium: text only), any invalid citations or
  unverified numbers as warnings, an expandable source list (company - fiscal
  year - page), and the last 5 questions in a collapsed history.
- **Knowledge Graph tab**: company, relationship groups, fiscal year and a
  node budget -> read-only, parameterised 2-hop Cypher -> pyvis network
  (Company blue, FinancialMetric green, Risk red, BusinessSegment orange per
  pipeline.txt; others distinct). The budget is split across relationship
  types so 109 metrics cannot crowd out risks and people. The page's JS is
  inlined (works without CDN access). Labels and tooltips are HTML-escaped,
  because some stored text was written by the LLM - nothing from the database
  can inject markup or script (tested).
- **Errors**: Neo4j unreachable -> `st.error` with the connection string and
  the password masked; model failures (quota, network, API errors) ->
  `st.warning` with the raw error. Nothing is substituted.

Verified 2026-10-28: Streamlit `AppTest` runs the real app against live Neo4j
(`FGR_RUN_INTEGRATION=1`): it starts without exceptions, the graph tab renders
(Apple: 108 nodes, 156 relationships), and a search returns the cited answer
"$88,136 million (GAAP) [Source: Microsoft Annual Report FY2024, p.61]" with a
high-confidence badge, a source list and one history entry (uses a cached
answer - no quota). `streamlit run app.py` serves and passes the health check.

Checked in Chrome (`streamlit run app.py`, http://localhost:8501): sidebar,
Q&A with pipeline.txt's "Compare Apple and Microsoft revenue" (green
high-confidence badge, both companies FY2021-FY2025, every figure cited, 7
sources listed as company - fiscal year - page) and the Knowledge Graph tab
(108 nodes / 156 relationships for Apple, colour legend, interactive network).
The browser check found two issues the headless test could not see, both
fixed: (1) markdown rendered text between two "$" signs as a LaTeX formula -
dollar signs are now escaped (asserted in the end-to-end test); (2) the graph
layout spread most nodes outside the frame - it now settles and zooms to fit,
with on-screen zoom buttons.

## Testing and validation (phase 8)

| suite | command | what it covers | result (2026-10-28) | result (2026-10-29, 4 companies) |
|---|---|---|---|---|
| unit | `pytest tests/ -v --ignore-glob="*integration*"` | every component with fakes / synthetic data, plus real-corpus checks when `data/` is present | **88 passed** | **109 passed** (29 s) |
| integration | `pytest -m integration -v` | live Neo4j, local model, Gemini: retrieval, golden set, acceptance checks of phases 3-7, the Streamlit app end to end | **8 passed** (1 min 52 s) | **8 passed** (46 s) |

Added for IBM/NVIDIA (2026-10-29): 12 hand-verified IBM/NVIDIA figures in the
real-corpus fact check (`EXPECTED`), a real-corpus layout test (IBM signatures
p.26 and Exhibit 13 kept as report content; NVIDIA signatures pages 91/84/86;
titled income statements with FY2025-2023 columns), unit tests for every new
parsing rule, `--companies` merges, `embeddings append`, citations with all-caps
names and the fiscal-year note, and the integration checks below.

A plain `pytest` runs the unit tests and skips the integration tests
(`tests/conftest.py`), so it never touches the database or spends API quota.

**pipeline.txt phase 8, mapped to this project** (module names follow the
`fin_graph_rag/` layout; Tesla is not indexed, so that check uses the closest
equivalent; NVIDIA has been indexed since 2026-10-29, so its check now runs as
written):

| pipeline.txt test | here | result |
|---|---|---|
| `parse_pdf` on a mocked 2-page PDF - dict shape | `test_pipeline.py::test_parse_two_page_pdf_returns_expected_shape` | pass |
| `extract_entities` JSON parsing with extra whitespace | `test_llm_json_with_extra_whitespace_is_parsed` (+ thinking parts skipped, malformed JSON logged and skipped) | pass |
| `build_context`: graph facts before text chunks | `test_build_context_graph_facts_before_text` | pass |
| `cypher_search("Apple revenue")` has a revenue result | `test_integration.py::test_cypher_search_apple_revenue` | pass |
| `vector_search("AI chips demand")` -> NVIDIA | as written: top hit is NVIDIA (all top 5 are NVIDIA chunks); plus a Microsoft-only topic -> Microsoft | pass |
| `hybrid_search("Compare Tesla and Microsoft margins")` -> both in sources | "Compare Apple and Microsoft gross margin" and "Compare IBM and NVIDIA gross margin" -> both companies in sources and facts | pass |
| golden set (`tests/golden_questions.py`): answer > 50 chars, >= 1 citation, no "I don't know"; >= 3/4 | same checks + no invalid citation + the verified figure/name present; 6 questions since 2026-10-29, threshold 75% (>= 5) | **4/4**, then **6/6** |

Golden answers (cached, cited): Apple vs Microsoft revenue FY2021-FY2025 (7
sources); Apple risks across 6 categories (8 sources); Apple FY2025 net income
$112,010 million (p.32); Microsoft FY2025 CEO Satya Nadella (p.19). Added
2026-10-29: NVIDIA FY2025 revenue $130,497 million (p.52); IBM CEO Arvind
Krishna (FY2025 report). The Apple/Microsoft revenue answer now also states both
fiscal-year ends (phase 6 change).

**Failure handling** (CLAUDE.md 5.1 #10, `tests/test_pipeline.py`): malformed
model JSON is logged and that extraction batch skipped (not cached, retried
next run) - this was a crash before phase 8 wrote the test; a malformed answer
is reported as a model error, not shown as an answer; missing Gemini key ->
setup message; HTTP 503 is retried; unreachable Neo4j -> clear error without
the password; missing vector index -> the build command to run.

**End-to-end measurements** (earlier phases, `eval/results/`): retrieval
37/39 supported in the top 5 (graph + text); answers 41/41 with valid
citations, 0 invalid citations, 0 unverified numbers, 3/3 missing-evidence
questions declined, manual review 38/39 correct + 1 partial.

Open items: text citations use a chunk's first page (one answer cites p.130
for a figure on p.131); the eval set
(42 questions) is small - differences of a few cases are not significant.

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

**Gemini embeddings** (`gemini:gemini-embedding-001`, free tier, 768-d,
RETRIEVAL_DOCUMENT / RETRIEVAL_QUERY task types; same inputs and questions;
measured 2026-10-28, `eval/results/*gemini*`):

| support (39 cases) | @1 | @5 | @10 | MRR | E10-E42: @5 / @10 / MRR |
|---|---|---|---|---|---|
| bge-small hybrid (current default) | 16 | 27 | 31 | 0.538 | 21 / 24 / 0.502 |
| **Gemini vector + header** | **29** | **37** | **39** | **0.833** | **29 / 31 / 0.814** |
| Gemini hybrid (+ BM25, RRF) | 26 | 31 | 38 | 0.732 | 23 / 30 / 0.679 |

Gemini vector search is clearly better: per case vs bge-small vector it is
better on 25, worse on 3, equal on 11 (vs bge-small hybrid: 19 / 3 / 17), and
the gain holds on the 31 cases written after the design choices. Adding BM25
*hurts* Gemini (support@5 37 -> 31): keyword fusion helped the weaker local
model, not this one. Remaining misses rank 8-9 (E15 Apple Japan, E32 Microsoft
dividends). **Decision (2026-10-28): not adopted - the local bge-small model
stays the default.** Estimated end-to-end gain is small (graph + text support@5
37/39 measured with bge-small vs ~39/39 estimated with Gemini), while Gemini
would need a call per question (free
tier: per-minute and daily limits, network required, questions sent to Google)
and re-loading the 1,432 chunk vectors and the 768-d vector index in Neo4j.
Building the index took 47 batch calls over two days (daily quota hit once).

**bge-base: no meaningful gain over bge-small.** Per case, bge-base
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
