"""
COMPONENT : Evaluation / tests - retrieval (BM25, RRF, hybrid)
PURPOSE   : Check tokenization, BM25 scoring, Reciprocal Rank Fusion and the
            hybrid Retriever's ranking without loading an embedding model.
INPUTS    : Synthetic chunks; a fake vector index with a fixed ranking.
OUTPUTS   : pytest results.
DEPENDS ON: pytest, numpy
PANEL NOTE: MOCK: FakeVector returns a fixed ranking instead of embedding
            queries; these tests verify retrieval logic, not model quality.
"""

import pytest

from fin_graph_rag.retrieval import BM25, Retriever, rrf, tokenize


def chunk(cid, text, company="microsoft", fy=2024, tables=()):
    return {"chunk_id": cid, "doc_id": f"{company}_{fy}", "company": company, "fiscal_year": fy,
            "section": "Item 8", "in_exhibits": False, "tables": list(tables), "text": text,
            "overlap_chars": 0, "page_start": 1, "page_end": 1, "qualifiers": []}


class FakeVector:
    manifest = {"model": "fake"}

    def __init__(self, chunks, order):
        self.chunks = chunks
        self.order = order          # chunk_ids, best first

    def search(self, query, k=5):
        by_id = {c["chunk_id"]: c for c in self.chunks}
        return [{"rank": r + 1, "score": 1.0, **by_id[cid]} for r, cid in enumerate(self.order[:k])]


def test_tokenize_splits_fy_years_keeps_numbers_drops_stopwords():
    assert tokenize("What was Microsoft's FY2024 net income of $88,136?") == \
        ["microsoft", "s", "fy", "2024", "net", "income", "88,136"]


def test_bm25_prefers_rarer_matching_terms_and_ignores_unknown_terms():
    docs = [tokenize(t) for t in ["net income net income", "income statements net income", "cash flows"]]
    bm = BM25(docs)
    s = bm.scores(tokenize("income statements"))
    assert s[1] > s[0] > 0 and s[2] == 0          # "statements" is rare, so doc 1 wins
    assert (bm.scores(["nonexistent"]) == 0).all()


def test_rrf_rewards_agreement_between_rankings():
    fused = rrf([[0, 1, 2], [2, 1, 0]], k=60)
    assert fused[1] == pytest.approx(2 / 62)
    assert fused[0] == fused[2] == pytest.approx(1 / 61 + 1 / 63)
    assert fused[0] > fused[1]                    # rank 1 in one list beats rank 2 in both


def test_hybrid_lifts_chunk_that_matches_keywords_but_ranks_low_on_vectors():
    chunks = [chunk("prose", "Revenue increased driven by cloud growth."),
              chunk("stmt", "INCOME STATEMENTS Net income $ 88,136 $ 72,361",
                    tables=[{"title": "INCOME STATEMENTS", "column_years": [2024, 2023], "qualifiers": []}]),
              chunk("other", "Risk factors about competition.")]
    vec = FakeVector(chunks, ["prose", "other", "stmt"])          # vectors put the statement last
    hybrid = Retriever("hybrid", vector_index=vec)
    top = hybrid.search("Microsoft income statements net income", k=3)
    # Every chunk's header matches "microsoft"/"statements", so "prose" is bm25 #2:
    # prose = 1/61 + 1/62 edges out stmt = 1/63 + 1/61, but stmt climbs from vector #3 to #2.
    assert [r["chunk_id"] for r in top] == ["prose", "stmt", "other"]
    assert top[1]["bm25_rank"] == 1 and top[1]["vector_rank"] == 3
    # bm25 mode needs no vector index and returns a full ranking.
    only_bm25 = Retriever("bm25", chunks=chunks)
    assert only_bm25.search("income statements", k=3)[0]["chunk_id"] == "stmt"
    assert len(only_bm25.search("income statements", k=10)) == 3


def test_unknown_mode_is_rejected():
    with pytest.raises(ValueError):
        Retriever("magic", chunks=[])
