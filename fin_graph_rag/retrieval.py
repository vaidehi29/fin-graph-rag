"""
COMPONENT : Retrieval
PURPOSE   : Keyword (BM25) search over chunks and hybrid search that fuses
            BM25 with the vector index using Reciprocal Rank Fusion (RRF).
INPUTS    : data/processed/chunks.jsonl, the vector index built by
            fin_graph_rag.embeddings, eval/cases.jsonl (for `evaluate`)
OUTPUTS   : ranked chunks (CLI `search`); eval/results/<date>_<mode>_k<k>.json
DEPENDS ON: numpy, fin_graph_rag.embeddings (vector index, chunk headers)
PANEL NOTE: Embeddings capture meaning but blur exact tokens - a table row
            "Net income $ 88,136" says little to a vector model. BM25 rewards
            exact words ("income statements", "Microsoft", "2024"). RRF
            combines the two rankings without needing their scores to be on
            the same scale. All local and free: no API, no new dependency.

Usage:
    python -m fin_graph_rag.retrieval search "Microsoft net income fiscal 2024" [--mode hybrid|bm25|vector] [-k 5]
    python -m fin_graph_rag.retrieval evaluate [--mode hybrid|bm25|vector] [-k 5]
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from fin_graph_rag import embeddings as em

log = logging.getLogger("retrieval")

# BM25 parameters: the common defaults (Robertson et al.), not tuned here.
BM25_K1 = 1.5
BM25_B = 0.75
# DEMO-KNOB: RRF constant. 60 is the value from the original RRF paper
# (Cormack et al., 2009); larger values flatten the advantage of top ranks.
RRF_K = 60

# Letters and digits are split apart so "FY2024" matches a query's "2024";
# numbers keep their separators so "88,136" stays one token.
TOKEN_RE = re.compile(r"[a-z]+|\d+(?:[.,]\d+)*")
# Question words and function words: they appear everywhere, and keeping
# them only adds noise to short queries.
STOPWORDS = set("""
a an and are as at be by did do does for from had has have how in is it its of on or
that the their this to was were what when where which who why with
""".split())


def tokenize(text: str) -> list[str]:
    return [t for t in TOKEN_RE.findall(text.lower()) if t not in STOPWORDS]


def bm25_text(chunk: dict) -> str:
    """Same context header the vector index embeds, plus the chunk text."""
    return f"{em.chunk_header(chunk)}\n{chunk['text']}"


class BM25:
    """Okapi BM25 over pre-tokenized documents."""

    def __init__(self, docs: list[list[str]], k1: float = BM25_K1, b: float = BM25_B):
        self.n = len(docs)
        self.k1, self.b = k1, b
        self.lengths = np.array([len(d) for d in docs], dtype=np.float32)
        self.avg_len = float(self.lengths.mean()) if self.n else 0.0
        self.postings: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        tmp: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for i, d in enumerate(docs):
            for term, f in Counter(d).items():
                tmp[term].append((i, f))
        for term, plist in tmp.items():
            idx, freq = zip(*plist)
            self.postings[term] = (np.array(idx), np.array(freq, dtype=np.float32))
        # BM25+ style idf: always positive, so very common terms never subtract.
        self.idf = {t: math.log(1 + (self.n - len(p[0]) + 0.5) / (len(p[0]) + 0.5))
                    for t, p in self.postings.items()}

    def scores(self, query_tokens: list[str]) -> np.ndarray:
        s = np.zeros(self.n, dtype=np.float32)
        norm = self.k1 * (1 - self.b + self.b * self.lengths / self.avg_len) if self.n else None
        for term in set(query_tokens):
            if term not in self.postings:
                continue
            idx, f = self.postings[term]
            s[idx] += self.idf[term] * f * (self.k1 + 1) / (f + norm[idx])
        return s


def rrf(rankings: list[list[int]], k: int = RRF_K) -> dict[int, float]:
    """Reciprocal Rank Fusion: score(d) = sum over rankings of 1 / (k + rank)."""
    fused: dict[int, float] = defaultdict(float)
    for ranking in rankings:
        for rank, doc in enumerate(ranking, start=1):
            fused[doc] += 1.0 / (k + rank)
    return fused


class Retriever:
    """Vector, BM25 or hybrid (RRF) search over one set of chunks.

    Exposes .chunks, .manifest and .search(query, k) so it can be scored by
    fin_graph_rag.embeddings.evaluate exactly like the vector index.
    """

    def __init__(self, mode: str = "hybrid", vector_index: em.Index | None = None,
                 chunks: list[dict] | None = None):
        if mode not in ("hybrid", "bm25", "vector"):
            raise ValueError(f"unknown mode {mode!r}")
        self.mode = mode
        self.vector = vector_index if vector_index is not None or mode == "bm25" else em.Index()
        # Use the vector index's chunk order when present so row i means the same chunk in both.
        self.chunks = self.vector.chunks if self.vector is not None else (chunks or em.load_chunks())
        self.bm25 = BM25([tokenize(bm25_text(c)) for c in self.chunks])
        self.manifest = {
            "mode": mode, "rrf_k": RRF_K, "bm25_k1": BM25_K1, "bm25_b": BM25_B,
            "bm25_text": "chunk_header + chunk text",
            "vector_index": self.vector.manifest if self.vector is not None else None,
        }

    def _vector_ranking(self, query: str) -> list[int]:
        pos = {c["chunk_id"]: i for i, c in enumerate(self.chunks)}
        return [pos[r["chunk_id"]] for r in self.vector.search(query, k=len(self.chunks))]

    def _bm25_ranking(self, query: str) -> tuple[list[int], np.ndarray]:
        s = self.bm25.scores(tokenize(query))
        order = np.argsort(-s, kind="stable")
        # Chunks sharing no query term are left out of the keyword ranking.
        return [int(i) for i in order if s[i] > 0], s

    def search(self, query: str, k: int = 5) -> list[dict]:
        if self.mode == "vector":
            return self.vector.search(query, k)
        bm_rank, bm_scores = self._bm25_ranking(query)
        if self.mode == "bm25":
            ranked = [(i, float(bm_scores[i])) for i in bm_rank]
            # Keep a full ranking for evaluation: unmatched chunks follow in index order.
            seen = set(bm_rank)
            ranked += [(i, 0.0) for i in range(len(self.chunks)) if i not in seen]
            vec_pos = {}
        else:
            vec_rank = self._vector_ranking(query)
            fused = rrf([vec_rank, bm_rank])
            ranked = sorted(fused.items(), key=lambda x: -x[1])
            vec_pos = {d: r for r, d in enumerate(vec_rank, start=1)}
        bm_pos = {d: r for r, d in enumerate(bm_rank, start=1)}
        return [{"rank": r + 1, "score": round(s, 6), "vector_rank": vec_pos.get(i),
                 "bm25_rank": bm_pos.get(i), **self.chunks[i]}
                for r, (i, s) in enumerate(ranked[:k])]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Keyword / hybrid retrieval over the chunk index.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("search"); s.add_argument("query")
    e = sub.add_parser("evaluate")
    for p in (s, e):
        p.add_argument("-k", type=int, default=5)
        p.add_argument("--mode", choices=["hybrid", "bm25", "vector"], default="hybrid")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    try:
        r = Retriever(args.mode)
        if args.cmd == "search":
            for x in r.search(args.query, args.k):
                own = x["text"][x.get("overlap_chars", 0):]
                print(f'#{x["rank"]} {x["score"]:.4f} vec#{x.get("vector_rank")} bm25#{x.get("bm25_rank")} '
                      f'{x["chunk_id"]} p{x["page_start"]}-{x["page_end"]} {x["section"]} {x["qualifiers"] or ""}'
                      f'\n    {em._norm(own)[:200]}')
        else:
            result = em.evaluate(r, args.k)
            result["model"] = f"{em.MODEL_NAME} ({args.mode})"
            em.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
            path = em.results_path(f"{em.index_dir(em.INDEX_ROOT, True).name}-{args.mode}", args.k,
                                   len(result["cases"]))
            path.write_text(json.dumps(result, indent=2), encoding="utf-8")
            for row in result["cases"]:
                print(f'{row["id"]} {row["category"]:22} hit@{args.k}={row[f"hit@{args.k}"]!s:5} '
                      f'support@{args.k}={row[f"support@{args.k}"]!s:5} evidence_ranks={row["evidence_ranks"]} '
                      f'support_ranks={row["support_ranks"]} trap_ranks={row["trap_ranks"]}')
            print(json.dumps(result["summary"], indent=2))
            print(f"Saved {path}")
    except (FileNotFoundError, ValueError, RuntimeError) as ex:
        log.error("%s", ex)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
