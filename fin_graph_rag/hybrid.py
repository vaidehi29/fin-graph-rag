"""
COMPONENT : Retrieval - hybrid (graph + text)
PURPOSE   : One entry point that returns exact facts from the knowledge graph
            and supporting text chunks for a question, with page-level sources.
INPUTS    : question text; Neo4j (graph facts + chunk vectors); chunks.jsonl
            (keyword index over the same chunk text)
OUTPUTS   : {"graph_facts": [...], "text_chunks": [...], "sources": [...]};
            `evaluate` writes eval/results/<timestamp>_hybrid-graph_n<N>_k<k>.json
DEPENDS ON: fin_graph_rag.cypher_retriever, fin_graph_rag.vector_retriever,
            fin_graph_rag.retrieval (BM25, RRF), fin_graph_rag.graph_builder
PANEL NOTE: Graph facts answer "what is the number" exactly; text chunks give
            context and cover what the graph does not hold (employee counts,
            growth commentary). Text ranking fuses Neo4j vector search with
            BM25 (RRF) because that combination measured best on the eval set.
            Sources are "<Company> Annual Report FY<year>, p.<page>" - always
            with a page - graph facts first.

Usage:
    python -m fin_graph_rag.hybrid "Compare Apple and Microsoft revenue"
    python -m fin_graph_rag.hybrid --evaluate [-k 5]
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from itertools import zip_longest

from fin_graph_rag import embeddings as em
from fin_graph_rag.cypher_retriever import cypher_search, detect_companies, source_label
from fin_graph_rag.retrieval import BM25, bm25_text, rrf, tokenize
from fin_graph_rag.vector_retriever import vector_search

log = logging.getLogger("hybrid")

# DEMO-KNOB: candidates taken from each text ranking before fusion.
CANDIDATES = 100
# DEMO-KNOB: keep only one chunk per (company, year, page). pipeline.txt asks to
# dedupe by page; the company and year are needed because page 32 exists in
# every report. Measured with --evaluate (see README) before choosing the default.
DEDUPE_BY_PAGE = False
# DEMO-KNOB: when the question names a company, drop text chunks from the other.
FILTER_BY_COMPANY = True

_bm25 = None


def _keyword_index():
    global _bm25
    if _bm25 is None:
        chunks = em.load_chunks()
        _bm25 = (BM25([tokenize(bm25_text(c)) for c in chunks]), chunks)
    return _bm25


def text_search(driver, query: str, top_k: int = 5, database=None, embedder=None,
                dedupe: bool | None = None, company_filter: bool | None = None,
                default_companies: list[str] | None = None) -> list[dict]:
    dedupe = DEDUPE_BY_PAGE if dedupe is None else dedupe
    company_filter = FILTER_BY_COMPANY if company_filter is None else company_filter
    vec = vector_search(driver, query, CANDIDATES, database, embedder)
    bm25, chunks = _keyword_index()
    scores = bm25.scores(tokenize(query))
    kw = [chunks[i]["chunk_id"] for i in scores.argsort()[::-1][:CANDIDATES] if scores[i] > 0]
    fused = rrf([[v["chunk_id"] for v in vec], kw])
    by_id = {v["chunk_id"]: v for v in vec}
    chunk_by_id = {c["chunk_id"]: c for c in chunks}
    companies = detect_companies(query) or list(default_companies or [])
    ranked = sorted(fused.items(), key=lambda x: -x[1])
    if len(companies) > 1:
        # Comparison question: alternate between the named companies so the
        # text side covers both, instead of whichever company ranks higher.
        per = {co: [(cid, s) for cid, s in ranked if (by_id.get(cid) or chunk_by_id[cid])["company"] == co]
               for co in companies}
        ranked = [x for group in zip_longest(*per.values()) for x in group if x is not None]
    out, seen_pages = [], set()
    for cid, score in ranked:
        c = by_id.get(cid) or chunk_by_id[cid]
        company, year = c["company"], c.get("year", c.get("fiscal_year"))
        page = c.get("page_num", c.get("page_start"))
        if company_filter and companies and company not in companies:
            continue
        if dedupe:
            if (company, year, page) in seen_pages:
                continue
            seen_pages.add((company, year, page))
        out.append({"chunk_id": cid, "text": c["text"], "company": company, "year": year, "page_num": page,
                    "overlap_chars": c.get("overlap_chars", 0), "qualifiers": c.get("qualifiers", []),
                    "rrf_score": round(score, 6),
                    "source": source_label(f"{company.title()} 10-K FY{year}", page)})
        if len(out) == top_k:
            break
    return out


def hybrid_search(driver, query: str, top_k: int = 5, database=None, embedder=None,
                  default_companies: list[str] | None = None, **text_opts) -> dict:
    facts = cypher_search(driver, query, database, default_companies=default_companies)
    chunks = text_search(driver, query, top_k, database, embedder, default_companies=default_companies,
                         **text_opts)
    sources: list[str] = []
    for s in [s for f in facts for s in f["sources"]] + [c["source"] for c in chunks]:
        if s not in sources:
            sources.append(s)
    return {"graph_facts": facts, "text_chunks": chunks, "sources": sources}


# ---------------------------------------------------------------------------
# Evaluation on eval/cases.jsonl
# ---------------------------------------------------------------------------

def _figure(s: str) -> float | None:
    m = re.fullmatch(r"\$?([\d,]+(?:\.\d+)?)", s.strip())
    return float(m.group(1).replace(",", "")) if m else None


def graph_supports(facts: list[dict], sup: dict) -> bool:
    """A metric fact whose value equals the expected figure (sign ignored: outflows are negative)."""
    want = _figure(sup["figure"])
    return want is not None and any(f["type"] == "metric" and abs(abs(f["value"]) - want) < 0.005 for f in facts)


def evaluate(driver, k: int = 5, database=None, **text_opts) -> dict:
    cases = [json.loads(l) for l in em.CASES_PATH.read_text(encoding="utf-8").splitlines() if l.strip()]
    embedder = None
    rows = []
    for case in cases:
        res = hybrid_search(driver, case["question"], k, database, embedder, **text_opts)
        sups = case.get("answer_support", [])
        g = [graph_supports(res["graph_facts"], s) for s in sups]
        t = [any(em._supports(c, s) for c in res["text_chunks"]) for s in sups]
        rows.append({
            "id": case["id"], "category": case["category"], "question": case["question"],
            "graph_facts": len(res["graph_facts"]), "graph_support": all(g) if sups else None,
            f"text_support@{k}": all(t) if sups else None,
            f"combined_support@{k}": all(a or b for a, b in zip(g, t)) if sups else None,
            "sources_have_pages": all(re.search(r", p\.\d+$", s) for s in res["sources"]),
            "n_sources": len(res["sources"]),
            "facts": [f["text"] for f in res["graph_facts"][:5]],
            "chunks": [c["chunk_id"] for c in res["text_chunks"]],
        })
    scored = [r for r in rows if r["graph_support"] is not None]
    missing = [r for r in rows if r["category"] == "missing_evidence"]
    return {
        "k": k, "options": {"dedupe": text_opts.get("dedupe", DEDUPE_BY_PAGE),
                            "company_filter": text_opts.get("company_filter", FILTER_BY_COMPANY),
                            "candidates": CANDIDATES},
        "summary": {
            "cases_with_support": len(scored),
            "graph_support": sum(r["graph_support"] for r in scored),
            f"text_support@{k}": sum(r[f"text_support@{k}"] for r in scored),
            f"combined_support@{k}": sum(r[f"combined_support@{k}"] for r in scored),
            "missing_evidence_cases_with_graph_facts": sum(r["graph_facts"] > 0 for r in missing),
            "all_sources_have_pages": all(r["sources_have_pages"] for r in rows),
        },
        "cases": rows,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Hybrid graph + text retrieval.")
    ap.add_argument("query", nargs="?")
    ap.add_argument("-k", type=int, default=5)
    ap.add_argument("--evaluate", action="store_true", help="score on eval/cases.jsonl")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    if not args.evaluate and not args.query:
        ap.error("give a query or --evaluate")

    from fin_graph_rag.graph_builder import driver_session
    try:
        with driver_session() as (driver, db):
            if args.evaluate:
                result = evaluate(driver, args.k, db)
                path = em.results_path("hybrid-graph", args.k, len(result["cases"]))
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(result, indent=2), encoding="utf-8")
                for r in result["cases"]:
                    print(f'{r["id"]} {r["category"]:22} graph={r["graph_support"]!s:5} '
                          f'text={r[f"text_support@{args.k}"]!s:5} combined={r[f"combined_support@{args.k}"]!s:5}')
                print(json.dumps(result["summary"], indent=2))
                print(f"Saved {path}")
                return 0
            res = hybrid_search(driver, args.query, args.k, db)
            print(f"Graph facts ({len(res['graph_facts'])}):")
            for f in res["graph_facts"][:15]:
                print(f"  - {f['text']}")
            print("\nTop text chunks:")
            for c in res["text_chunks"][:3]:
                own = em._norm(c["text"][c["overlap_chars"]:])
                print(f"  - [{c['source']}] {own[:220]}")
            print("\nSources:")
            for s in res["sources"]:
                print(f"  - {s}")
    except RuntimeError as e:
        log.error("%s", e)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
