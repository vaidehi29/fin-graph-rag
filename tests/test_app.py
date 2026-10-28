"""
COMPONENT : Evaluation / tests - user interface
PURPOSE   : Check the graph view offline: captions, label escaping, colours,
            empty selections.
INPUTS    : Synthetic graph dicts.
OUTPUTS   : pytest results.
DEPENDS ON: pytest, pyvis
PANEL NOTE: The end-to-end run of the real app (Streamlit AppTest against live
            Neo4j) is in tests/test_integration.py (pytest -m integration).
"""

from fin_graph_rag.graph_view import NODE_COLORS, caption, fetch_subgraph, to_html


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
