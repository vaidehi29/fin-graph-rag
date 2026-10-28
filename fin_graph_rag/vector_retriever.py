"""
COMPONENT : Retrieval - vector search in Neo4j
PURPOSE   : Embed a question with the same model used for the stored chunk
            vectors and query the Neo4j `chunk_embeddings` index.
INPUTS    : question text; Neo4j driver; the embedding model named on the
            stored TextChunk nodes (embedding_model)
OUTPUTS   : list[dict] chunks with text, company, year, page_num, chunk_id,
            score (cosine) and neo4j_score ((1 + cosine) / 2 as Neo4j reports it)
DEPENDS ON: fin_graph_rag.embedders (query embedding), neo4j driver
PANEL NOTE: The query model must be the model that embedded the chunks; it
            is read from the graph rather than assumed, so switching models
            without re-loading fails loudly instead of returning nonsense.
"""

from __future__ import annotations

import logging

from fin_graph_rag.embedders import get_embedder
from fin_graph_rag.graph_builder import VECTOR_INDEX

_embedders: dict = {}

# TODO: Neo4j 5.27 reports db.index.vector.queryNodes as deprecated in favour of
# the Cypher SEARCH clause. pipeline.txt specifies queryNodes and it still works
# on AuraDB; silence the per-query deprecation notice and migrate later.
logging.getLogger("neo4j.notifications").setLevel(logging.ERROR)

VECTOR_QUERY = f"""
CALL db.index.vector.queryNodes('{VECTOR_INDEX}', $k, $vector) YIELD node, score
MATCH (node)-[:PART_OF]->(r:Report)
RETURN node.chunk_id AS chunk_id, node.text AS text, node.company AS company, node.year AS year,
       node.page_num AS page_num, node.page_end AS page_end, node.section AS section,
       node.overlap_chars AS overlap_chars, node.qualifiers AS qualifiers, node.in_exhibits AS in_exhibits,
       r.name AS report, score AS neo4j_score
ORDER BY score DESC
"""


def stored_model(driver, database=None) -> str:
    models = [r["m"] for r in driver.execute_query(
        "MATCH (t:TextChunk) RETURN DISTINCT t.embedding_model AS m", database_=database).records]
    if len(models) != 1:
        raise RuntimeError(f"Expected one embedding model on TextChunk nodes, found {models}; re-run ingestion")
    return models[0]


def query_embedder(driver, database=None, embedder=None):
    if embedder is not None:
        return embedder
    name = stored_model(driver, database)
    if name not in _embedders:
        _embedders[name] = get_embedder(name)
    return _embedders[name]


def vector_search(driver, query: str, top_k: int = 5, database=None, embedder=None) -> list[dict]:
    vec = query_embedder(driver, database, embedder).embed_query(query)
    rows = driver.execute_query(VECTOR_QUERY, k=top_k, vector=[float(x) for x in vec], database_=database).records
    out = []
    for rank, r in enumerate(rows, start=1):
        d = dict(r)
        d["rank"] = rank
        d["score"] = round(2 * d["neo4j_score"] - 1, 4)     # back to plain cosine similarity
        out.append(d)
    return out
