"""
COMPONENT : Evaluation / tests - entity extraction and graph rows
PURPOSE   : Check table-fact parsing rules (amounts, labels, table allow-list,
            EPS vs share counts, segment context, conflicts) and, on the real
            corpus, that extracted figures equal the hand-verified eval answers.
INPUTS    : Synthetic table units; data/extracted + data/processed when present.
OUTPUTS   : pytest results.
DEPENDS ON: pytest
PANEL NOTE: MOCK: unit tests use synthetic Unit objects. The real-corpus test
            compares against figures verified on the PDF pages (eval/cases.jsonl).
            Checks against the live graph are in tests/test_integration.py
            (pytest -m integration), so this file never touches the database.
"""

import os
from pathlib import Path

import pytest

from fin_graph_rag import entity_extractor as ex
from fin_graph_rag.graph_builder import fact_rows
from fin_graph_rag.preprocess import Line, Unit

ROOT = Path(__file__).resolve().parent.parent


def unit(title_lines, rows, years, company_doc="microsoft_2024", table_no=0):
    lines = [Line(t, 10, None, "Item 7", False) for t in title_lines + rows]
    u = Unit("table", lines, title=" / ".join(title_lines), n_header=len(title_lines), table_no=table_no)
    u.years = years
    return u, {"doc_id": company_doc, "company": company_doc.split("_")[0]}


def facts_from(u, doc):
    return [f for f, _ in ex.table_facts(u, doc, {}, ex.Counter())]


def test_parse_amounts_skips_percent_columns_and_keeps_negatives():
    assert ex.parse_amounts("$ 162,560 (4) % $ 169,658 11 % $ 153,306".split()) == [162560, 169658, 153306]
    assert ex.parse_amounts("(90,711) (94,949) (77,550)".split()) == [-90711, -94949, -77550]
    assert ex.parse_amounts("88,136 72,361 22%".split()) == [88136, 72361]
    assert ex.parse_amounts("— 5,228 5,465".split()) == [None, 5228, 5465]


def test_split_row_strips_footnote_marker_from_label():
    label, toks = ex.split_row("iPhone (1) $ 200,583 (2) % $ 205,489")
    assert ex.clean_label(label) == "iPhone" and toks[0] == "$"


def test_allow_list_is_case_sensitive_for_statement_titles():
    assert ex.table_allowed("ITEM 8. FINANCIAL STATEMENTS / INCOME STATEMENTS / (In millions)")
    # Activision net-impact table: "income statements" in lower case must NOT qualify.
    assert not ex.table_allowed("Following is the net impact of the Activision Blizzard acquisition on our "
                                "consolidated income statements since the date / of acquisition: / (In millions)")


def test_eps_is_not_confused_with_diluted_share_count():
    u, doc = unit(["CONSOLIDATED STATEMENTS OF OPERATIONS", "(In millions, except per-share amounts)"],
                  ["2025 2024", "Net income $ 112,010 $ 93,736", "Earnings per share:", "Diluted $ 7.46 $ 6.08",
                   "Shares used in computing earnings per share:", "Diluted 15,004,697 15,408,095"],
                  [2025, 2024], "apple_2025")
    eps = [f for f in facts_from(u, doc) if f.metric == "diluted_eps"]
    assert [(f.fiscal_year, f.value, f.unit) for f in eps] == [(2025, 7.46, "USD per share"), (2024, 6.08, "USD per share")]


def test_segment_rows_take_metric_from_context_and_segment_from_title():
    u, doc = unit(["SEGMENT RESULTS OF OPERATIONS", "(In millions, except percentages) 2025 2024 Change",
                   "Productivity and Business Processes"],
                  ["Revenue $ 120,810 $ 106,820 13%", "Operating Income $ 69,773 $ 59,661 17%", "Intelligent Cloud",
                   "Revenue $ 106,265 $ 87,464 21%"], [2025, 2024], "microsoft_2025")
    got = {(f.segment, f.metric, f.fiscal_year): f.value for f in facts_from(u, doc)}
    assert got[("Productivity and Business Processes", "revenue", 2025)] == 120810
    assert got[("Productivity and Business Processes", "operating_income", 2024)] == 59661
    assert got[("Intelligent Cloud", "revenue", 2025)] == 106265
    assert not any(seg is None for seg, _, _ in got)          # no company-level revenue from a segment table


def test_rows_under_cost_of_sales_are_not_read_as_product_sales():
    u, doc = unit(["CONSOLIDATED STATEMENTS OF OPERATIONS", "(In millions)"],
                  ["2023 2022", "Net sales:", "Services 85,200 78,129", "Cost of sales:", "Services 24,855 22,075"],
                  [2023, 2022], "apple_2023")
    services = {f.fiscal_year: f.value for f in facts_from(u, doc) if f.product == "Services"}
    assert services == {2023: 85200, 2022: 78129}


def test_pro_forma_table_gives_pro_forma_basis_and_unmatched_rows_are_skipped():
    u, doc = unit(["Following are the supplemental consolidated financial results of Microsoft Corporation on an "
                   "unaudited pro forma", "(In millions, except per share amounts)"],
                  ["Year Ended June 30, 2024 2023", "Net income 88,308 71,383", "Revenue $ 247,442"],
                  [2024, 2023])
    skipped = ex.Counter()
    facts = [f for f, _ in ex.table_facts(u, doc, {}, skipped)]
    assert {(f.metric, f.fiscal_year, f.basis, f.value) for f in facts} == {
        ("net_income", 2024, "pro_forma", 88308), ("net_income", 2023, "pro_forma", 71383)}
    assert skipped["values_do_not_match_year_columns"] == 1    # one value for two year columns


def test_merge_flags_conflicts_and_prefers_own_year_report():
    f = lambda: ex.Fact("microsoft", "revenue", 2023, "gaap", "USD millions", 0, segment="Intelligent Cloud")
    ev = lambda doc, v: ex.Evidence(doc, 1, None, "t", "title", "row", v)
    merged = ex.merge_facts([(f(), ev("microsoft_2023", 87907)), (f(), ev("microsoft_2024", 87907)),
                             (f(), ev("microsoft_2025", 72944))])
    assert len(merged) == 1 and merged[0].conflict and merged[0].value == 87907
    row = fact_rows(merged)[0]
    assert row["company_name"] == "Microsoft Corporation" and len(row["evidence"]) == 3
    assert {e["report_year"] for e in row["evidence"]} == {2023, 2024, 2025}


# Hand-verified on the PDF pages (see eval/cases.jsonl).
EXPECTED = [
    ("microsoft|net_income|||2024|gaap", 88136), ("microsoft|net_income|||2024|pro_forma", 88308),
    ("microsoft|net_income|||2023|non_gaap", 73307), ("microsoft|revenue|||2024|pro_forma", 247442),
    ("microsoft|revenue|||2025|gaap", 281724), ("microsoft|revenue|Intelligent Cloud||2025|gaap", 106265),
    ("microsoft|diluted_eps|||2025|gaap", 13.64), ("microsoft|total_assets|||2025|gaap", 619003),
    ("apple|revenue|||2024|gaap", 391035), ("apple|revenue|Greater China||2025|gaap", 64377),
    ("apple|revenue||iPhone|2023|gaap", 200583), ("apple|revenue||Services|2025|gaap", 109158),
    ("apple|diluted_eps|||2025|gaap", 7.46), ("apple|share_repurchases|||2025|gaap", -90711),
    ("apple|net_income|||2025|gaap", 112010), ("apple|operating_cash_flow|||2023|gaap", 110543),
]


def test_real_corpus_facts_match_verified_figures():
    if not (ROOT / "data/extracted/pages.jsonl").exists() or not (ROOT / "data/processed/chunks.jsonl").exists():
        pytest.skip("pipeline outputs not present")
    old = os.getcwd()
    os.chdir(ROOT)
    try:
        facts, stats = ex.extract_table_facts()
    finally:
        os.chdir(old)
    by_key = {f.key: f for f in facts}
    for key, value in EXPECTED:
        assert key in by_key and by_key[key].value == value, key
    # Every fact keeps its source row, page and chunk.
    assert all(e.row_text and e.page and e.chunk_id for f in facts for e in f.evidence)
    # The only disagreements are Microsoft's FY2025 segment recast.
    assert all(f.company == "microsoft" and f.segment and f.fiscal_year in (2023, 2024)
               for f in facts if f.conflict)


# --- LLM text entities: validation (MOCK: hand-written "model outputs") -------

def chunk(cid, text, company="microsoft", fy=2024):
    return {"chunk_id": cid, "doc_id": f"{company}_{fy}", "company": company, "fiscal_year": fy,
            "page_start": 12, "text": text, "overlap_chars": 0, "token_count": len(text.split()), "section": "Item 1"}


def test_validate_keeps_only_items_grounded_in_the_cited_chunk():
    c = chunk("m_c1", "Our competitors include Apple Inc. and Alphabet in devices. Satya Nadella, Chief Executive Officer")
    rejected = ex.Counter()
    items = [
        {"name": "Alphabet", "context": "devices", "chunk_id": "m_c1", "evidence_quote": "competitors include Apple Inc. and Alphabet"},
        {"name": "Apple Inc.", "context": "devices", "chunk_id": "m_c1", "evidence_quote": "competitors include Apple Inc. and Alphabet"},
        {"name": "Amazon", "context": "cloud", "chunk_id": "m_c1", "evidence_quote": "competitors include Apple Inc. and Alphabet"},
        {"name": "Alphabet", "context": "x", "chunk_id": "nope", "evidence_quote": "competitors include Apple Inc. and Alphabet"},
        {"name": "Alphabet", "context": "x", "chunk_id": "m_c1", "evidence_quote": "competitors are many and varied"},
    ]
    got = ex.validate("competitors", items, [c], rejected)
    assert [e.fields["name"] for e in got] == ["Alphabet", "Apple Inc."]   # "Apple Inc." maps to the canonical node
    assert rejected == {"competitors:name_not_in_chunk": 1, "competitors:unknown_chunk": 1,
                        "competitors:quote_not_in_chunk": 1}


def test_validate_rejects_self_reference_and_bad_risk_category():
    c = chunk("m_c2", "Microsoft Corporation competes broadly. Cyberattacks could disrupt our cloud services badly.")
    rejected = ex.Counter()
    ex.validate("competitors", [{"name": "Microsoft Corp.", "context": "x", "chunk_id": "m_c2",
                                 "evidence_quote": "Microsoft Corporation competes broadly"}], [c], rejected)
    risks = ex.validate("risks", [
        {"category": "cybersecurity_privacy", "title": "Cyberattacks", "description": "d", "chunk_id": "m_c2",
         "evidence_quote": "Cyberattacks could disrupt our cloud"},
        {"category": "made_up", "title": "X", "description": "d", "chunk_id": "m_c2",
         "evidence_quote": "Cyberattacks could disrupt our cloud"}], [c], rejected)
    assert [r.fields["category"] for r in risks] == ["cybersecurity_privacy"]
    assert rejected == {"competitors:self_reference": 1, "risks:bad_category_or_title": 1}


def test_quote_match_ignores_curly_quotes_case_and_line_breaks():
    c = chunk("a_c1", "The Company’s operations and\nperformance depend significantly on global conditions.", "apple")
    got = ex.validate("risks", [{"category": "macroeconomic", "title": "Global conditions", "description": "d",
                                 "chunk_id": "a_c1", "evidence_quote": "the company's operations and performance depend"}],
                      [c], ex.Counter())
    assert len(got) == 1


def test_people_rows_split_board_members_from_executives():
    from fin_graph_rag.graph_builder import entity_rows
    mk = lambda name, role: ex.TextEntity("executive", "apple", 2023, "apple_2023", "a_c9", 60, "q", {"name": name, "role": role})
    rows = entity_rows([mk("Timothy D. Cook", "Chief Executive Officer and Director"), mk("James A. Bell", "Director"),
                        mk("Arthur D. Levinson", "Director and Chair of the Board"),
                        mk("Chris Kondo", "Senior Director of Corporate Accounting")])["executive"]
    assert {r["name"]: r["board_only"] for r in rows} == {
        "Timothy D. Cook": False, "James A. Bell": True, "Arthur D. Levinson": True, "Chris Kondo": False}
    assert rows[0]["key"] == "timothy d cook"


def test_batches_never_mix_reports_and_respect_token_budget():
    cs = [dict(chunk(f"a{i}", "w " * 1200, "apple", 2023), token_count=1200) for i in range(3)] + \
         [dict(chunk("m0", "w", "microsoft", 2023), token_count=10)]
    b = ex.batches(cs, max_tokens=3000)
    assert [[c["chunk_id"] for c in x] for x in b] == [["a0", "a1"], ["a2"], ["m0"]]


def test_competitor_rows_carry_kind_so_only_companies_become_company_nodes():
    from fin_graph_rag.graph_builder import entity_rows
    mk = lambda name, kind: ex.TextEntity("competitor", "microsoft", 2023, "microsoft_2023", "m_c1", 14, "q",
                                          {"name": name, "kind": kind, "context": "x"})
    rows = entity_rows([mk("Linux", "open_source_project"), mk("Apple Inc.", "company")])["competitor"]
    assert [(r["name"], r["kind"]) for r in rows] == [("Linux", "open_source_project"), ("Apple Inc.", "company")]


def test_report_nodes_get_a_readable_name():
    from fin_graph_rag.graph_builder import entity_rows, report_name
    assert report_name("apple", 2024) == "Apple 10-K FY2024"
    f = ex.Fact("microsoft", "net_income", 2024, "gaap", "USD millions", 88136)
    f.evidence.append(ex.Evidence("microsoft_2025", 69, "c", "t", "title", "row", 88136))
    assert fact_rows([f])[0]["evidence"][0]["report_name"] == "Microsoft 10-K FY2025"   # the report it came from
    risk = ex.TextEntity("risk", "apple", 2023, "apple_2023", "a_c1", 8, "q", {"category": "other", "title": "T", "description": "d"})
    assert entity_rows([risk])["risk"][0]["report_name"] == "Apple 10-K FY2023"


def test_chunk_rows_carry_provenance_and_the_vector():
    from fin_graph_rag.graph_builder import chunk_rows
    c = {"chunk_id": "apple_2024_c0102", "text": "INCOME STATEMENTS ...", "company": "apple", "fiscal_year": 2024,
         "page_start": 32, "page_end": 33, "section": "Item 8", "in_exhibits": False, "has_table": True,
         "qualifiers": [], "overlap_chars": 0, "token_count": 450}
    row = chunk_rows([c], [[0.5, -0.25, 0.0]], "BAAI/bge-small-en-v1.5")[0]
    assert row["chunk_index"] == 102 and row["page_num"] == 32 and row["year"] == 2024
    assert row["report_name"] == "Apple 10-K FY2024" and row["company_name"] == "Apple Inc."
    assert row["vector"] == [0.5, -0.25, 0.0] and row["embedding_model"] == "BAAI/bge-small-en-v1.5"
