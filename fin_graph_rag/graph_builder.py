"""
COMPONENT : Knowledge graph
PURPOSE   : Create the Neo4j schema (uniqueness constraints) and write
            validated facts with MERGE, keeping provenance on every metric.
INPUTS    : list[Fact] from fin_graph_rag.entity_extractor; Neo4j connection
            settings NEO4J_URI / NEO4J_USERNAME / NEO4J_PASSWORD / NEO4J_DATABASE
            (environment or .env)
OUTPUTS   : nodes and relationships in Neo4j (idempotent: re-running updates,
            never duplicates)
DEPENDS ON: neo4j Python driver, fin_graph_rag.embedders.load_env

Graph schema (as written by this module):

    (:Company {name, id})-[:FILED]->(:Report {name, company, year, doc_id, source_file})
    (:Company)-[:HAS_METRIC]->(:FinancialMetric {key, metric, fiscal_year, basis,
                                                 unit, value, segment, product, conflict})
    (:Company)-[:HAS_SEGMENT]->(:BusinessSegment {key, company, name})-[:HAS_METRIC]->(:FinancialMetric)
    (:Company)-[:SELLS]->(:Product {key, company, name})-[:HAS_METRIC]->(:FinancialMetric)
    (:FinancialMetric)-[:REPORTED_IN {page, table_id, chunk_id, row_text, value}]->(:Report)
    (:Company)-[:FACES_RISK {fiscal_year}]->(:Risk {key, category, title, description})
    (:Risk)-[:IN_CATEGORY]->(:RiskCategory {name})
    (:Risk)-[:DISCLOSED_IN {page, chunk_id, quote}]->(:Report)
    (:Company)-[:HAS_EXECUTIVE | BOARD_MEMBER {fiscal_year, role, page, chunk_id, quote}]->(:Person {key, name})
    (:Company)-[:COMPETES_WITH {fiscal_year, context, page, chunk_id, quote}]->(:Company)
    (:Company)-[:COMPETES_WITH {...}]->(:Offering {name, kind})   (competing products / open source)

PANEL NOTE: One FinancialMetric node per (company, metric, segment/product,
            fiscal year, basis), with one REPORTED_IN edge per place a report
            states it. So "where does this number come from?" is a one-hop
            query, and a metric whose reports disagree (Microsoft's FY2025
            segment recast) keeps every value on its edges with conflict=true
            instead of silently picking one. GAAP, non-GAAP and pro forma
            figures are separate nodes (basis), never merged.
"""

from __future__ import annotations

import os
from contextlib import contextmanager

from fin_graph_rag.embedders import load_env
import re

from fin_graph_rag.entity_extractor import COMPANY_NAMES, Fact, TextEntity

CONSTRAINTS = [
    "CREATE CONSTRAINT company_name IF NOT EXISTS FOR (c:Company) REQUIRE c.name IS UNIQUE",
    "CREATE CONSTRAINT report_company_year IF NOT EXISTS FOR (r:Report) REQUIRE (r.company, r.year) IS UNIQUE",
    "CREATE CONSTRAINT metric_key IF NOT EXISTS FOR (m:FinancialMetric) REQUIRE m.key IS UNIQUE",
    "CREATE CONSTRAINT segment_key IF NOT EXISTS FOR (s:BusinessSegment) REQUIRE s.key IS UNIQUE",
    "CREATE CONSTRAINT product_key IF NOT EXISTS FOR (p:Product) REQUIRE p.key IS UNIQUE",
    "CREATE CONSTRAINT risk_key IF NOT EXISTS FOR (r:Risk) REQUIRE r.key IS UNIQUE",
    "CREATE CONSTRAINT risk_category IF NOT EXISTS FOR (k:RiskCategory) REQUIRE k.name IS UNIQUE",
    "CREATE CONSTRAINT person_key IF NOT EXISTS FOR (p:Person) REQUIRE p.key IS UNIQUE",
    "CREATE CONSTRAINT offering_name IF NOT EXISTS FOR (o:Offering) REQUIRE o.name IS UNIQUE",
]

# Every text entity is attached to the Report it came from, with page, chunk and
# the verbatim quote that validated it.
WRITE_RISKS = """
UNWIND $rows AS row
MERGE (c:Company {name: row.company_name})
MERGE (r:Report {company: row.company, year: row.fiscal_year})
  SET r.name = row.report_name
MERGE (k:Risk {key: row.key})
  SET k.company = row.company, k.fiscal_year = row.fiscal_year, k.category = row.category,
      k.title = row.title, k.description = row.description
MERGE (c)-[:FACES_RISK {fiscal_year: row.fiscal_year}]->(k)
MERGE (cat:RiskCategory {name: row.category})
MERGE (k)-[:IN_CATEGORY]->(cat)
MERGE (k)-[d:DISCLOSED_IN]->(r)
  SET d.page = row.page, d.chunk_id = row.chunk_id, d.quote = row.quote
"""
WRITE_PEOPLE = """
UNWIND $rows AS row
MERGE (c:Company {name: row.company_name})
MERGE (p:Person {key: row.key})
  SET p.name = row.name
FOREACH (_ IN CASE WHEN row.board_only THEN [1] ELSE [] END |
  MERGE (c)-[h:BOARD_MEMBER {fiscal_year: row.fiscal_year}]->(p)
    SET h.role = row.role, h.page = row.page, h.chunk_id = row.chunk_id, h.quote = row.quote)
FOREACH (_ IN CASE WHEN row.board_only THEN [] ELSE [1] END |
  MERGE (c)-[h:HAS_EXECUTIVE {fiscal_year: row.fiscal_year}]->(p)
    SET h.role = row.role, h.page = row.page, h.chunk_id = row.chunk_id, h.quote = row.quote)
"""
# Only names classified as companies become Company nodes; competing products
# and open-source projects (e.g. Linux, MySQL) are Offering nodes.
WRITE_COMPETITORS = """
UNWIND $rows AS row
MERGE (c:Company {name: row.company_name})
FOREACH (_ IN CASE WHEN row.kind = 'company' THEN [1] ELSE [] END |
  MERGE (o:Company {name: row.name})
  MERGE (c)-[w:COMPETES_WITH {fiscal_year: row.fiscal_year}]->(o)
    SET w.context = row.context, w.page = row.page, w.chunk_id = row.chunk_id, w.quote = row.quote)
FOREACH (_ IN CASE WHEN row.kind = 'company' THEN [] ELSE [1] END |
  MERGE (o:Offering {name: row.name})
    SET o.kind = row.kind
  MERGE (c)-[w:COMPETES_WITH {fiscal_year: row.fiscal_year}]->(o)
    SET w.context = row.context, w.page = row.page, w.chunk_id = row.chunk_id, w.quote = row.quote)
"""
# "Director", "Director and Chair of the Board" -> board; anything naming an
# officer role ("Chief Executive Officer and Director") -> executive.
BOARD_ONLY = re.compile(r"^(independent )?(director|chair)\b", re.I)
OFFICER = re.compile(r"officer|president|controller|treasurer|secretary|accounting", re.I)


def person_key(name: str) -> str:
    return " ".join(name.lower().replace(".", " ").split())


def entity_rows(entities: list[TextEntity]) -> dict[str, list[dict]]:
    rows: dict[str, list[dict]] = {"risk": [], "executive": [], "competitor": []}
    for e in entities:
        base = {"company": e.company, "company_name": COMPANY_NAMES[e.company], "fiscal_year": e.fiscal_year,
                "report_name": report_name(e.company, e.fiscal_year),
                "page": e.page, "chunk_id": e.chunk_id, "quote": e.quote}
        if e.kind == "risk":
            title_key = " ".join(re.sub(r"[^a-z0-9 ]", " ", e.fields["title"].lower()).split())
            rows["risk"].append({**base, **e.fields, "key": f"{e.doc_id}|{e.fields['category']}|{title_key}"})
        elif e.kind == "executive":
            role = e.fields["role"]
            rows["executive"].append({**base, "name": e.fields["name"], "role": role, "key": person_key(e.fields["name"]),
                                      "board_only": bool(BOARD_ONLY.search(role)) and not OFFICER.search(role)})
        else:
            rows["competitor"].append({**base, **e.fields})
    return rows


def write_entities(driver, entities: list[TextEntity], database=None, batch: int = 100) -> dict:
    rows = entity_rows(entities)
    for kind, query in (("risk", WRITE_RISKS), ("executive", WRITE_PEOPLE), ("competitor", WRITE_COMPETITORS)):
        for i in range(0, len(rows[kind]), batch):
            driver.execute_query(query, rows=rows[kind][i:i + batch], database_=database)
    return {k: len(v) for k, v in rows.items()}

WRITE_FACTS = """
UNWIND $rows AS row
MERGE (c:Company {name: row.company_name})
  SET c.id = row.company
MERGE (m:FinancialMetric {key: row.key})
  SET m.company = row.company, m.metric = row.metric, m.fiscal_year = row.fiscal_year,
      m.basis = row.basis, m.unit = row.unit, m.value = row.value, m.segment = row.segment,
      m.product = row.product, m.conflict = row.conflict
MERGE (c)-[:HAS_METRIC]->(m)
FOREACH (_ IN CASE WHEN row.segment IS NULL THEN [] ELSE [1] END |
  MERGE (s:BusinessSegment {key: row.company + '|' + row.segment})
    SET s.company = row.company, s.name = row.segment
  MERGE (c)-[:HAS_SEGMENT]->(s)
  MERGE (s)-[:HAS_METRIC]->(m))
FOREACH (_ IN CASE WHEN row.product IS NULL THEN [] ELSE [1] END |
  MERGE (p:Product {key: row.company + '|' + row.product})
    SET p.company = row.company, p.name = row.product
  MERGE (c)-[:SELLS]->(p)
  MERGE (p)-[:HAS_METRIC]->(m))
WITH c, m, row
UNWIND row.evidence AS ev
MERGE (r:Report {company: row.company, year: ev.report_year})
  SET r.doc_id = ev.doc_id, r.source_file = ev.doc_id + '.pdf', r.form = '10-K', r.name = ev.report_name
MERGE (c)-[:FILED]->(r)
MERGE (m)-[e:REPORTED_IN {page: ev.page, table_id: ev.table_id}]->(r)
  SET e.chunk_id = ev.chunk_id, e.row_text = ev.row_text, e.value = ev.value
"""


SHORT_NAMES = {"apple": "Apple", "microsoft": "Microsoft"}


def report_name(company: str, year: int) -> str:
    """Readable Report caption, e.g. "Apple 10-K FY2024" (Neo4j Browser shows `name`)."""
    return f"{SHORT_NAMES.get(company, company.title())} 10-K FY{year}"


def connection_settings() -> dict:
    load_env()
    missing = [k for k in ("NEO4J_URI", "NEO4J_USERNAME", "NEO4J_PASSWORD") if not os.getenv(k)]
    if missing:
        raise RuntimeError(f"Missing Neo4j settings in .env: {', '.join(missing)}")
    return {"uri": os.environ["NEO4J_URI"], "auth": (os.environ["NEO4J_USERNAME"], os.environ["NEO4J_PASSWORD"]),
            "database": os.getenv("NEO4J_DATABASE") or None}


@contextmanager
def driver_session():
    """Yield (driver, database). Errors name the problem, never the password."""
    from neo4j import GraphDatabase
    s = connection_settings()
    driver = GraphDatabase.driver(s["uri"], auth=s["auth"])
    try:
        driver.verify_connectivity()
    except Exception as e:
        driver.close()
        # FALLBACK: no offline substitute for the graph; fail clearly.
        raise RuntimeError(f"Cannot connect to Neo4j at {s['uri']}: {type(e).__name__}: {e}") from None
    try:
        yield driver, s["database"]
    finally:
        driver.close()


def fact_rows(facts: list[Fact]) -> list[dict]:
    """Facts as plain parameter dicts for the UNWIND query."""
    rows = []
    for f in facts:
        rows.append({
            "key": f.key, "company": f.company, "company_name": COMPANY_NAMES[f.company],
            "metric": f.metric, "fiscal_year": f.fiscal_year, "basis": f.basis, "unit": f.unit,
            "value": f.value, "segment": f.segment, "product": f.product, "conflict": f.conflict,
            "evidence": [{"doc_id": e.doc_id, "report_year": int(e.doc_id.rsplit("_", 1)[1]),
                          "report_name": report_name(f.company, int(e.doc_id.rsplit("_", 1)[1])), "page": e.page,
                          "table_id": e.table_id, "chunk_id": e.chunk_id, "row_text": e.row_text,
                          "value": e.value} for e in f.evidence],
        })
    return rows


def ensure_constraints(driver, database=None) -> None:
    for stmt in CONSTRAINTS:
        driver.execute_query(stmt, database_=database)


def write_facts(driver, facts: list[Fact], database=None, batch: int = 50) -> int:
    rows = fact_rows(facts)
    for i in range(0, len(rows), batch):
        driver.execute_query(WRITE_FACTS, rows=rows[i:i + batch], database_=database)
    return len(rows)


def graph_summary(driver, database=None) -> dict:
    """Counts used to verify an ingestion run (all read-only queries)."""
    q = lambda s, **p: driver.execute_query(s, database_=database, **p).records
    return {
        "nodes_by_label": {r["label"]: r["n"] for r in q(
            "MATCH (n) UNWIND labels(n) AS label RETURN label, count(*) AS n ORDER BY label")},
        "relationships_by_type": {r["t"]: r["n"] for r in q(
            "MATCH ()-[r]->() RETURN type(r) AS t, count(*) AS n ORDER BY t")},
        "company_degree": {r["name"]: r["degree"] for r in q(
            "MATCH (c:Company) WHERE c.id IS NOT NULL RETURN c.name AS name, count{(c)-[]->()} AS degree "
            "ORDER BY name")},   # reporting companies only (competitor nodes have no outgoing edges)
        "has_metric_per_company": {r["name"]: r["n"] for r in q(
            "MATCH (c:Company)-[:HAS_METRIC]->(m) RETURN c.name AS name, count(m) AS n ORDER BY name")},
        "conflicting_metrics": q("MATCH (m:FinancialMetric {conflict: true}) RETURN count(m) AS n")[0]["n"],
    }
