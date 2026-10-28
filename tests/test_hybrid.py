"""
COMPONENT : Evaluation / tests - graph and hybrid retrieval
PURPOSE   : Check question parsing (companies, years, metrics, segments, basis,
            intents), fact text and source labels, graph-support scoring, and
            hybrid text ranking (company filter, comparison interleaving).
INPUTS    : Synthetic questions; fake vector search and keyword index.
OUTPUTS   : pytest results.
DEPENDS ON: pytest, numpy
PANEL NOTE: MOCK: text ranking tests replace Neo4j vector search and the BM25
            index with fixed rankings. The live acceptance check (compare query
            returns both companies, every source has a page) runs only with
            FGR_RUN_INTEGRATION=1.
"""

import os

import numpy as np
import pytest

from fin_graph_rag import cypher_retriever as cr
from fin_graph_rag import hybrid


def test_detects_companies_including_aliases():
    assert cr.detect_companies("Compare Apple and Microsoft revenue") == ["apple", "microsoft"]
    assert cr.detect_companies("What net income did the iPhone maker report?") == ["apple"]
    assert cr.detect_companies("MSFT cloud growth") == ["microsoft"]
    assert cr.detect_companies("NVIDIA data center revenue") == []


def test_detects_fiscal_years():
    assert cr.detect_years("from fiscal 2023 to fiscal 2025") == [2023, 2025]
    assert cr.detect_years("FY24 results") == [2024]
    assert cr.detect_years("total revenue") == []


def test_eps_question_is_not_also_a_net_income_question():
    assert cr.detect_metrics("What were Apple's diluted earnings per share for fiscal 2025?") == ["diluted_eps"]
    assert cr.detect_metrics("What was Microsoft's net income?") == ["net_income"]


def test_iphone_maker_alias_does_not_trigger_the_iphone_product_filter():
    assert cr.detect_segments("What net income did the iPhone maker report?", "apple") == []
    assert cr.detect_segments("How much revenue did iPhone generate for Apple?", "apple") == ["iPhone"]
    # "Microsoft Cloud" revenue is not the Intelligent Cloud segment.
    assert cr.detect_segments("How large was Microsoft Cloud revenue?", "microsoft") == []


def test_pro_forma_and_non_gaap_only_when_asked():
    assert cr.detect_basis("What was Microsoft's net income in fiscal 2024?") == ["gaap"]
    assert cr.detect_basis("pro forma revenue if the acquisition had closed") == ["pro_forma"]
    assert cr.detect_basis("adjusted (non-GAAP) net income") == ["gaap", "non_gaap"]


def test_fact_text_shows_conflicting_reports_and_sources_have_pages():
    r = {"company_name": "Microsoft Corporation", "metric": "revenue", "fiscal_year": 2023, "value": 87907.0,
         "unit": "USD millions", "basis": "gaap", "segment": "Intelligent Cloud", "product": None, "conflict": True,
         "evidence": [{"report": "Microsoft 10-K FY2023", "value": 87907.0},
                      {"report": "Microsoft 10-K FY2025", "value": 72944.0}]}
    text = cr.metric_text(r)
    assert "$87,907 million (GAAP)" in text and "Microsoft 10-K FY2025 states $72,944 million" in text
    assert cr.source_label("Apple 10-K FY2024", 32) == "Apple Annual Report FY2024, p.32"


def test_graph_support_matches_figures_ignoring_outflow_sign():
    facts = [{"type": "metric", "value": -90711.0}, {"type": "metric", "value": 7.46}]
    assert hybrid.graph_supports(facts, {"label": "x", "figure": "90,711"})
    assert hybrid.graph_supports(facts, {"label": "x", "figure": "7.46"})
    assert not hybrid.graph_supports(facts, {"label": "x", "figure": "137.4 billion"})


def _fake_retrieval(monkeypatch, chunks, vector_order):
    class FakeBM25:
        def scores(self, toks):
            return np.zeros(len(chunks))
    monkeypatch.setattr(hybrid, "_keyword_index", lambda: (FakeBM25(), chunks))
    by_id = {c["chunk_id"]: c for c in chunks}
    monkeypatch.setattr(hybrid, "vector_search", lambda d, q, k, db, e: [
        {"chunk_id": cid, "text": by_id[cid]["text"], "company": by_id[cid]["company"],
         "year": by_id[cid]["fiscal_year"], "page_num": by_id[cid]["page_start"], "overlap_chars": 0,
         "qualifiers": []} for cid in vector_order])


def _c(cid, company, page):
    return {"chunk_id": cid, "company": company, "fiscal_year": 2024, "page_start": page, "text": f"text {cid}",
            "overlap_chars": 0, "qualifiers": []}


def test_text_search_filters_to_named_company_and_interleaves_comparisons(monkeypatch):
    chunks = [_c("m1", "microsoft", 1), _c("m2", "microsoft", 2), _c("m3", "microsoft", 3), _c("a1", "apple", 4)]
    _fake_retrieval(monkeypatch, chunks, ["m1", "m2", "m3", "a1"])
    only_apple = hybrid.text_search(None, "Apple revenue", 3)
    assert [c["chunk_id"] for c in only_apple] == ["a1"]
    both = hybrid.text_search(None, "Compare Apple and Microsoft revenue", 4)
    assert [c["chunk_id"] for c in both] == ["a1", "m1", "m2", "m3"]     # alternates, keeps the longer list
    assert both[0]["source"] == "Apple Annual Report FY2024, p.4"


def test_dedupe_keeps_one_chunk_per_company_year_page(monkeypatch):
    chunks = [_c("m1", "microsoft", 7), _c("m2", "microsoft", 7), _c("m3", "microsoft", 8)]
    _fake_retrieval(monkeypatch, chunks, ["m1", "m2", "m3"])
    assert [c["chunk_id"] for c in hybrid.text_search(None, "Microsoft", 3, dedupe=True)] == ["m1", "m3"]


@pytest.mark.skipif(os.getenv("FGR_RUN_INTEGRATION") != "1", reason="needs Neo4j; set FGR_RUN_INTEGRATION=1")
def test_integration_compare_query_returns_both_companies_with_pages():
    # pipeline.txt phase 5 acceptance.
    from fin_graph_rag.graph_builder import driver_session
    with driver_session() as (driver, db):
        res = hybrid.hybrid_search(driver, "Compare Apple and Microsoft revenue", 5, db)
    assert {f["company"] for f in res["graph_facts"]} == {"apple", "microsoft"}
    assert {c["company"] for c in res["text_chunks"]} == {"apple", "microsoft"}
    assert res["sources"] and all(", p." in s for s in res["sources"])
