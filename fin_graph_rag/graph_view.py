"""
COMPONENT : User interface - knowledge graph view
PURPOSE   : Fetch a company's 2-hop neighbourhood from Neo4j and render it as
            an interactive pyvis network (HTML) for the Streamlit app.
INPUTS    : Neo4j driver; company id; relationship groups; optional fiscal year;
            node limit
OUTPUTS   : {"nodes": {...}, "edges": [...]} and an HTML string
DEPENDS ON: neo4j driver, pyvis
PANEL NOTE: Read-only query, parameterised, and capped (default 120 first-hop
            nodes) so the browser stays responsive - a full company
            neighbourhood is ~1,000 edges. Colours follow pipeline.txt:
            Company blue, FinancialMetric green, Risk red, BusinessSegment orange.
"""

from __future__ import annotations

import html

# pipeline.txt colours for the four named labels; the rest are distinct and muted.
NODE_COLORS = {
    "Company": "#1f77b4", "FinancialMetric": "#2ca02c", "Risk": "#d62728", "BusinessSegment": "#ff7f0e",
    "Product": "#ffbb78", "Person": "#9467bd", "Report": "#7f7f7f", "RiskCategory": "#e377c2",
    "Offering": "#17becf",
}
# UI groups -> relationship types leaving the Company node.
REL_GROUPS = {
    "Financial metrics": ["HAS_METRIC"],
    "Segments & products": ["HAS_SEGMENT", "SELLS"],
    "Risks": ["FACES_RISK"],
    "People": ["HAS_EXECUTIVE", "BOARD_MEMBER"],
    "Competitors": ["COMPETES_WITH"],
    "Reports": ["FILED"],
}

# The node budget is split across the selected relationship types (newest
# fiscal years first), so 109 financial metrics cannot crowd out risks,
# people and competitors.
SUBGRAPH_QUERY = """
UNWIND $types AS t
CALL {
  WITH t
  MATCH (c:Company {id: $company})-[r1]->(n)
  WHERE type(r1) = t
    AND ($year IS NULL OR coalesce(r1.fiscal_year, n.fiscal_year, n.year, $year) = $year)
  RETURN c, r1, n ORDER BY coalesce(r1.fiscal_year, n.fiscal_year, n.year, 0) DESC LIMIT $per_type
}
OPTIONAL MATCH (n)-[r2]->(m)
WHERE NOT m:TextChunk AND NOT m:Company AND NOT (type(r2) = 'HAS_METRIC' AND $year IS NOT NULL
      AND m.fiscal_year <> $year)
RETURN c, r1, n, collect(DISTINCT [r2, m])[..$per_node] AS hop2
"""


def caption(labels: list[str], p: dict) -> str:
    label = labels[0] if labels else "?"
    if label == "FinancialMetric":
        scope = p.get("segment") or p.get("product")
        value = p.get("value")
        shown = f"{value:,.2f}" if p.get("unit") == "USD per share" else f"{value:,.0f}"
        return (f"{p.get('metric', '').replace('_', ' ')}{' (' + scope + ')' if scope else ''}\n"
                f"FY{p.get('fiscal_year')}: {shown}{' [' + p['basis'] + ']' if p.get('basis') != 'gaap' else ''}")
    if label == "Risk":
        return (p.get("title") or "")[:40]
    return str(p.get("name") or p.get("key") or label)


def _node(n) -> tuple[str, dict]:
    props = dict(n)
    labels = list(n.labels)
    tip = "\n".join(f"{k}: {v}" for k, v in props.items() if k not in ("embedding", "text") and v not in (None, ""))
    return n.element_id, {"label": labels[0] if labels else "?", "caption": caption(labels, props), "title": tip}


def fetch_subgraph(driver, company: str, groups: list[str], year: int | None = None, limit: int = 120,
                   per_node: int = 6, database=None) -> dict:
    types = [t for g in groups for t in REL_GROUPS[g]]
    nodes: dict[str, dict] = {}
    edges: set[tuple[str, str, str]] = set()
    if not types:
        return {"nodes": nodes, "edges": []}
    per_type = max(5, limit // len(types))
    for rec in driver.execute_query(SUBGRAPH_QUERY, company=company, types=types, year=year, per_type=per_type,
                                    per_node=per_node, database_=database).records:
        for n in (rec["c"], rec["n"]):
            k, v = _node(n)
            nodes[k] = v
        edges.add((rec["c"].element_id, rec["n"].element_id, rec["r1"].type))
        for r2, m in rec["hop2"]:
            if r2 is None or m is None:
                continue
            k, v = _node(m)
            nodes[k] = v
            edges.add((rec["n"].element_id, m.element_id, r2.type))
    return {"nodes": nodes, "edges": sorted(edges)}


def to_html(graph: dict, height: str = "650px") -> str:
    from pyvis.network import Network
    # Library JS is inlined so the graph also renders without CDN access.
    net = Network(height=height, width="100%", directed=True, cdn_resources="in_line", bgcolor="#ffffff")
    # Compact layout that settles before display and is zoomed to fit the frame
    # (the default spread put most of a ~100-node graph outside the view).
    net.set_options("""{
      "physics": {"barnesHut": {"gravitationalConstant": -2500, "springLength": 90, "avoidOverlap": 0.3},
                  "stabilization": {"enabled": true, "iterations": 400, "fit": true}},
      "interaction": {"hover": true, "navigationButtons": true},
      "edges": {"smooth": false}
    }""")
    # Captions and tooltips come from the database (some written by the LLM):
    # escape them so no stored text can inject markup or script into the page.
    for nid, n in graph["nodes"].items():
        net.add_node(nid, label=html.escape(n["caption"]), title=html.escape(n["title"]),
                     color=NODE_COLORS.get(n["label"], "#bbbbbb"),
                     shape="dot", size=22 if n["label"] == "Company" else 12)
    for src, dst, rel in graph["edges"]:
        net.add_edge(src, dst, title=rel, label=rel.lower().replace("_", " "), font={"size": 8}, arrows="to")
    return net.generate_html()
