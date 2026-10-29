"""
COMPONENT : Evaluation / tests - integration (live services)
PURPOSE   : pipeline.txt phase 8 integration tests and golden-set evaluation,
            plus the acceptance checks of phases 3-7, against the live Neo4j
            graph, the local embedding model and Gemini (free tier).
INPUTS    : Neo4j + GEMINI_API_KEY from .env; tests/golden_questions.py
OUTPUTS   : pytest results
DEPENDS ON: fin_graph_rag (graph_builder, cypher_retriever, vector_retriever,
            hybrid, answer), streamlit (AppTest)
PANEL NOTE: Marked `integration`: skipped in the default run, selected with
            `pytest -m integration`. Read-only against Neo4j. Gemini answers are
            cached, so re-runs cost no quota (a new question costs one call).
            pipeline.txt's NVIDIA / Tesla checks are replaced by the closest
            Apple / Microsoft equivalents (only those reports are indexed).
"""

import os
import re
from pathlib import Path

import pytest

from golden_questions import GOLDEN_QUESTIONS

pytestmark = pytest.mark.integration
ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def neo4j():
    from fin_graph_rag.graph_builder import driver_session
    with driver_session() as (driver, db):
        yield driver, db


# --- pipeline.txt phase 8: integration tests ---------------------------------

def test_cypher_search_apple_revenue(neo4j):
    # pipeline.txt: cypher_search("Apple revenue") -> at least one revenue result.
    from fin_graph_rag.cypher_retriever import cypher_search
    driver, db = neo4j
    facts = cypher_search(driver, "Apple revenue", db)
    revenue = [f for f in facts if f["type"] == "metric" and f["metric"] == "revenue" and f["company"] == "apple"]
    assert revenue and all(f["sources"] for f in revenue)
    assert any(f["fiscal_year"] == 2025 and f["value"] == 416161 for f in revenue)


def test_vector_search_finds_the_right_company(neo4j):
    # pipeline.txt: vector_search("AI chips demand") -> NVIDIA (NVIDIA indexed since
    # 2026-10-29), plus the earlier stand-in: a Microsoft-only topic -> Microsoft.
    from fin_graph_rag.vector_retriever import vector_search
    driver, db = neo4j
    hits = vector_search(driver, "AI chips demand", 5, db)
    assert hits and hits[0]["company"] == "nvidia"
    hits = vector_search(driver, "Azure and Microsoft Cloud revenue growth", 5, db)
    assert hits and hits[0]["company"] == "microsoft"
    assert all(h["page_num"] and h["report"] for h in hits)


def test_hybrid_search_compare_cites_both_companies(neo4j):
    # pipeline.txt: hybrid_search("Compare Tesla and Microsoft margins") -> both in sources.
    from fin_graph_rag.hybrid import hybrid_search
    driver, db = neo4j
    res = hybrid_search(driver, "Compare Apple and Microsoft gross margin", 5, db)
    assert any(s.startswith("Apple ") for s in res["sources"])
    assert any(s.startswith("Microsoft ") for s in res["sources"])
    assert {f["company"] for f in res["graph_facts"] if f["type"] == "metric"} == {"apple", "microsoft"}


# --- pipeline.txt phase 8: golden set ----------------------------------------

def test_golden_questions(neo4j):
    from fin_graph_rag.answer import ask
    driver, db = neo4j
    passed = []
    for g in GOLDEN_QUESTIONS:
        r = ask(g["question"], driver, db)
        ok = (len(r["answer"]) > 50                                                    # not empty
              and len(r["sources"]) >= 1                                                # >= 1 valid citation
              and not re.search(r"i don't know|not in my training", r["answer"], re.I)  # no fallback
              and not r["invalid_citations"]
              and all(e in r["answer"] for e in g["expect"]))                           # verified content
        if g["question"] == "What risks does Apple mention?":
            ok = ok and r["n_graph_facts"] > 0
        passed.append((g["question"], ok))
    # pipeline.txt acceptance: at least 3 of 4 golden questions answered with citations.
    assert sum(ok for _, ok in passed) >= 3, passed


# --- acceptance checks of earlier phases (moved here from the unit files) ----

def test_graph_has_metrics_for_all_companies(neo4j):
    # Phase 3 (pipeline.txt): every company has >= 5 HAS_METRIC relationships.
    # IBM and NVIDIA were added on 2026-10-29.
    from fin_graph_rag.graph_builder import graph_summary
    driver, db = neo4j
    s = graph_summary(driver, db)
    assert set(s["has_metric_per_company"]) == {"Apple Inc.", "Microsoft Corporation",
                                                 "International Business Machines Corporation", "NVIDIA Corporation"}
    assert all(n >= 5 for n in s["has_metric_per_company"].values())


def test_text_chunks_and_vector_index_online(neo4j):
    # Phase 4: >= 500 TextChunk nodes, vector index ONLINE.
    from fin_graph_rag.graph_builder import VECTOR_INDEX
    driver, db = neo4j
    n = driver.execute_query("MATCH (t:TextChunk)-[:SOURCED_FROM]->(:Company) RETURN count(t) AS n",
                             database_=db).records[0]["n"]
    state = driver.execute_query("SHOW VECTOR INDEXES YIELD name, state WHERE name = $n RETURN state",
                                 n=VECTOR_INDEX, database_=db).records[0]["state"]
    assert n >= 500 and state == "ONLINE"


def test_compare_query_returns_both_companies_with_pages(neo4j):
    # Phase 5: the compare query returns facts for both companies; sources always have pages.
    from fin_graph_rag.hybrid import hybrid_search
    driver, db = neo4j
    res = hybrid_search(driver, "Compare Apple and Microsoft revenue", 5, db)
    assert {f["company"] for f in res["graph_facts"]} == {"apple", "microsoft"}
    assert {c["company"] for c in res["text_chunks"]} == {"apple", "microsoft"}
    assert res["sources"] and all(", p." in s for s in res["sources"])


def test_app_end_to_end():
    # Phase 7: the real Streamlit app, headless - start, Q&A with cited answer, graph tab.
    from streamlit.testing.v1 import AppTest
    cwd = os.getcwd()
    os.chdir(ROOT)
    try:
        at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=240)
        at.run()
        assert not at.exception
        assert [t.label for t in at.tabs] == ["Q&A", "Knowledge Graph"]
        assert at.sidebar.multiselect[0].value == ["Apple", "Microsoft"]
        assert any("nodes" in c.value and "relationships" in c.value for c in at.caption)   # graph tab rendered
        at.text_input[0].input("What was Microsoft's net income for fiscal year 2024?")
        at.button[0].click()
        at.run()
        assert not at.exception
        md = [m.value for m in at.markdown]
        assert any("88,136" in m and "[Source:" in m for m in md)
        # "$" must be escaped, or markdown renders "$88,136 ... $72,361" as a LaTeX formula.
        assert any(r"\$88,136" in m for m in md)
        assert any("High confidence" in m for m in md)
        assert any(m.startswith("- **Microsoft** - FY2024 - p.") for m in md)
        assert len(at.session_state["history"]) == 1
    finally:
        os.chdir(cwd)
