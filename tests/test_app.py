"""
COMPONENT : Evaluation / tests - user interface
PURPOSE   : Check the graph view (captions, label escaping, colours, empty
            selections) offline, and - with live services - run the Streamlit
            app headless end to end (sidebar, Q&A with cited answer and badge,
            sources, history, knowledge-graph tab).
INPUTS    : Synthetic graph dicts; app.py via streamlit.testing.v1.AppTest.
OUTPUTS   : pytest results.
DEPENDS ON: pytest, pyvis, streamlit
PANEL NOTE: The end-to-end test needs Neo4j and a cached Gemini answer, so it
            runs only with FGR_RUN_INTEGRATION=1. It asks a question whose
            answer is already in the LLM cache, so it spends no API quota.
"""

import os
from pathlib import Path

import pytest

from fin_graph_rag.graph_view import NODE_COLORS, caption, fetch_subgraph, to_html

ROOT = Path(__file__).resolve().parent.parent


def test_pipeline_colours_for_named_labels():
    assert NODE_COLORS["Company"] == "#1f77b4" and NODE_COLORS["FinancialMetric"] == "#2ca02c"
    assert NODE_COLORS["Risk"] == "#d62728" and NODE_COLORS["BusinessSegment"] == "#ff7f0e"


def test_metric_caption_shows_metric_year_value_and_non_gaap_basis():
    gaap = {"metric": "net_income", "fiscal_year": 2024, "value": 88136.0, "unit": "USD millions", "basis": "gaap"}
    assert caption(["FinancialMetric"], gaap) == "net income\nFY2024: 88,136"
    pf = {**gaap, "value": 88308.0, "basis": "pro_forma"}
    assert caption(["FinancialMetric"], pf).endswith("[pro_forma]")
    eps = {"metric": "diluted_eps", "fiscal_year": 2025, "value": 7.46, "unit": "USD per share", "basis": "gaap",
           "product": None, "segment": None}
    assert "7.46" in caption(["FinancialMetric"], eps)


def test_stored_text_cannot_inject_script_into_graph_page():
    html = to_html({"nodes": {"a": {"label": "Risk", "caption": "<script>alert(1)</script>",
                                    "title": "x</script><img src=x onerror=alert(2)>"}}, "edges": []})
    assert "<script>alert(1)" not in html and "onerror=alert(2)>" not in html


def test_no_relationship_groups_returns_empty_graph_without_querying():
    assert fetch_subgraph(None, "apple", []) == {"nodes": {}, "edges": []}


@pytest.mark.skipif(os.getenv("FGR_RUN_INTEGRATION") != "1", reason="needs Neo4j + cached answer")
def test_integration_app_end_to_end():
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
        # "$" must be escaped, or markdown renders "$88,136 ... $72,361" as a LaTeX formula
        # (seen in the browser before the fix).
        assert any(r"\$88,136" in m for m in md)
        assert any("High confidence" in m for m in md)
        assert any(m.startswith("- **Microsoft** - FY2024 - p.") for m in md)
        assert len(at.session_state["history"]) == 1
    finally:
        os.chdir(cwd)
