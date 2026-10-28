"""
COMPONENT : Embeddings and index
PURPOSE   : Embed every chunk with a local sentence-embedding model, persist
            the vectors as a flat index, run cosine-similarity search over it,
            and score retrieval against eval/cases.jsonl so embedding backends
            can be compared on the same questions.
INPUTS    : data/processed/chunks.jsonl (fin_graph_rag.preprocess),
            eval/cases.jsonl (for `evaluate`)
OUTPUTS   : data/index/<model-slug>/embeddings.npy  - float32 [n_chunks, dim],
                                                       L2-normalised
            data/index/<model-slug>/chunk_ids.json   - row -> chunk_id
            data/index/<model-slug>/manifest.json    - model, dims, chunks
                                                       file hash, timings
            eval/results/<date>_<model-slug>.json   - retrieval eval results
DEPENDS ON: sentence-transformers (torch, CPU), numpy
PANEL NOTE: Free and local: the model is downloaded once from Hugging Face
            and then runs offline on CPU - no API key, no per-call cost, no
            report text leaves the machine. bge-small-en-v1.5 uses the same
            bert-base-uncased WordPiece vocabulary and 512-token limit the
            chunker was sized against, which `build` re-verifies so nothing is
            silently truncated.

Usage:
    python -m fin_graph_rag.embeddings build [--no-header]
    python -m fin_graph_rag.embeddings search "Microsoft net income fiscal 2024" [-k 5]
    python -m fin_graph_rag.embeddings evaluate [-k 5]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

log = logging.getLogger("embeddings")

# DEMO-KNOB: embedding model. Must have a 512-token (or larger) input limit
# and a vocabulary compatible with the chunker's token counts.
MODEL_NAME = os.getenv("FGR_EMBED_MODEL", "BAAI/bge-small-en-v1.5")
# BGE v1.5 recommends this instruction on short retrieval queries only;
# passages are embedded without it.
QUERY_PREFIX = os.getenv("FGR_QUERY_PREFIX", "Represent this sentence for searching relevant passages: ")
BATCH_SIZE = 32

CHUNKS_PATH = Path(os.getenv("FGR_PROCESSED_DIR", "data/processed")) / "chunks.jsonl"
INDEX_ROOT = Path(os.getenv("FGR_INDEX_DIR", "data/index"))
CASES_PATH = Path("eval/cases.jsonl")
RESULTS_DIR = Path("eval/results")


def slug(model_name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", model_name.lower()).strip("-")


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_chunks(path: Path = CHUNKS_PATH) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found - run fin_graph_rag.preprocess first")
    with path.open(encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def load_model(name: str = MODEL_NAME):
    from sentence_transformers import SentenceTransformer
    try:
        return SentenceTransformer(name, device="cpu")
    except Exception as e:
        # FALLBACK: no silent substitute model - a different model would make
        # the index incomparable. Fail with a clear message instead.
        raise RuntimeError(f"Could not load embedding model {name!r} (first run needs network "
                           f"access to download it from Hugging Face): {e}") from e


# ---------------------------------------------------------------------------
# Chunk headers (experiment: does embedding document context help?)
# ---------------------------------------------------------------------------
# PANEL: A table chunk such as "Net income $ 88,136 $ 72,361 ..." says nothing
# about which company, report or statement it belongs to - that context lives
# in metadata the embedding never sees. The header puts it into the embedded
# text only; the stored chunk text (and its citations) are unchanged.

HEADER_VERSION = "v1"
COMPANY_NAMES = {"apple": "Apple Inc.", "microsoft": "Microsoft Corporation"}
ITEM_NAMES = {
    "1": "Business", "1A": "Risk Factors", "1B": "Unresolved Staff Comments", "1C": "Cybersecurity",
    "2": "Properties", "3": "Legal Proceedings", "4": "Mine Safety Disclosures",
    "5": "Market for Registrant's Common Equity", "6": "[Reserved]",
    "7": "Management's Discussion and Analysis", "7A": "Market Risk Disclosures",
    "8": "Financial Statements and Supplementary Data", "9": "Changes in and Disagreements with Accountants",
    "9A": "Controls and Procedures", "9B": "Other Information", "9C": "Foreign Jurisdictions",
    "10": "Directors, Executive Officers and Corporate Governance", "11": "Executive Compensation",
    "12": "Security Ownership", "13": "Certain Relationships and Related Transactions",
    "14": "Principal Accountant Fees and Services", "15": "Exhibits and Financial Statement Schedules",
    "16": "Form 10-K Summary",
}


def section_name(c: dict) -> str:
    if c["in_exhibits"]:
        return "Exhibits"
    if c["section"] == "front_matter":
        return "Cover page and contents"
    item = c["section"].removeprefix("Item ")
    return f"{c['section']} {ITEM_NAMES.get(item, '')}".strip()


def table_name(t: dict) -> str:
    """Most title-like part of a table title: an ALL-CAPS heading if present.

    "ITEM 8. ..." / "PART II" lines are section headings (already in the
    header), not table names, so they are skipped.
    """
    parts = [p for p in t["title"].split(" / ") if p and not re.match(r"(ITEM|PART)\b", p)]
    name = next((p for p in parts if p.isupper() and len(p) > 3), parts[0] if parts else "untitled")
    years = ", ".join(f"FY{y}" for y in t["column_years"])
    quals = ", ".join(q.replace("_", " ") for q in t["qualifiers"])
    return name[:100] + (f" ({years})" if years else "") + (f" [{quals}]" if quals else "")


def chunk_header(c: dict, with_section: bool = True, with_tables: bool = True) -> str:
    parts = [COMPANY_NAMES.get(c["company"], c["company"].title()), f"FY{c['fiscal_year']} Form 10-K"]
    if with_section:
        parts.append(section_name(c))
    if with_tables and c["tables"]:
        parts.append("Table: " + "; ".join(table_name(t) for t in c["tables"][:2]))
    return " | ".join(parts)


def embed_inputs(chunks: list[dict], tokenizer, limit: int, header: bool) -> tuple[list[str], dict]:
    """Texts to embed. With `header`, prefix each chunk with chunk_header(),
    dropping the table part, then the section, if needed to fit `limit`."""
    def n_tokens(s):
        return len(tokenizer(s, add_special_tokens=True)["input_ids"])

    inputs, levels = [], Counter()
    for c in chunks:
        text = c["text"]
        if header:
            for level, h in (("full", chunk_header(c)), ("no_table", chunk_header(c, with_tables=False)),
                             ("company_year_only", chunk_header(c, False, False))):
                if n_tokens(f"{h}\n{c['text']}") <= limit:
                    text = f"{h}\n{c['text']}"
                    levels[level] += 1
                    break
            else:
                levels["none_fits"] += 1
        inputs.append(text)
    return inputs, dict(levels)


def index_dir(index_root: Path, header: bool) -> Path:
    return index_root / (slug(MODEL_NAME) + ("-hdr" if header else ""))


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

def build(chunks_path: Path = CHUNKS_PATH, index_root: Path = INDEX_ROOT, header: bool = True) -> dict:
    chunks = load_chunks(chunks_path)
    if not chunks:
        raise ValueError(f"{chunks_path} is empty - nothing to index")
    model = load_model()
    limit = model.max_seq_length
    inputs, header_levels = embed_inputs(chunks, model.tokenizer, limit, header)

    # The model silently truncates long inputs; refuse to build instead.
    lengths = [len(model.tokenizer(t, add_special_tokens=True)["input_ids"]) for t in inputs]
    too_long = [(c["chunk_id"], n) for c, n in zip(chunks, lengths) if n > limit]
    if too_long:
        raise ValueError(f"{len(too_long)} chunks exceed the model's {limit}-token limit "
                         f"(e.g. {too_long[:3]}); re-run preprocess with a matching tokenizer")

    t0 = time.time()
    vecs = model.encode(inputs, batch_size=BATCH_SIZE, normalize_embeddings=True,
                        convert_to_numpy=True, show_progress_bar=True).astype(np.float32)
    seconds = time.time() - t0

    out = index_dir(index_root, header)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "embeddings.npy", vecs)
    (out / "chunk_ids.json").write_text(json.dumps([c["chunk_id"] for c in chunks]), encoding="utf-8")
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": MODEL_NAME,
        "dim": int(vecs.shape[1]),
        "n_chunks": int(vecs.shape[0]),
        "normalized": True,
        "similarity": "cosine (dot product of normalised vectors)",
        "query_prefix": QUERY_PREFIX,
        "chunk_header": HEADER_VERSION if header else None,
        "header_levels": header_levels if header else None,
        "example_input": inputs[min(100, len(inputs) - 1)][:300],
        "max_seq_length": limit,
        "max_input_tokens_seen": max(lengths),
        "chunks_file": str(chunks_path),
        "chunks_sha256": sha256_of(chunks_path),
        "encode_seconds": round(seconds, 1),
        "device": "cpu",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    log.info("Indexed %d chunks (dim %d) in %.1fs -> %s", len(chunks), vecs.shape[1], seconds, out)
    return manifest


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

class Index:
    """Flat in-memory cosine index over one model's embeddings."""

    def __init__(self, index_root: Path = INDEX_ROOT, chunks_path: Path = CHUNKS_PATH, model=None,
                 header: bool = True):
        d = index_dir(index_root, header)
        if not (d / "manifest.json").exists():
            raise FileNotFoundError(f"No index at {d} - run `python -m fin_graph_rag.embeddings build"
                                    f"{'' if header else ' --no-header'}`")
        self.manifest = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
        if self.manifest["model"] != MODEL_NAME:
            raise ValueError(f"Index built with {self.manifest['model']}, queries would use {MODEL_NAME}")
        # A stale index would return chunks that no longer match chunks.jsonl.
        if self.manifest["chunks_sha256"] != sha256_of(chunks_path):
            raise ValueError(f"{chunks_path} changed since the index was built - rebuild the index")
        self.vecs = np.load(d / "embeddings.npy")
        ids = json.loads((d / "chunk_ids.json").read_text(encoding="utf-8"))
        by_id = {c["chunk_id"]: c for c in load_chunks(chunks_path)}
        self.chunks = [by_id[i] for i in ids]
        self.model = model or load_model()

    def search(self, query: str, k: int = 5) -> list[dict]:
        q = self.model.encode([QUERY_PREFIX + query], normalize_embeddings=True, convert_to_numpy=True)[0]
        scores = self.vecs @ q.astype(np.float32)
        top = np.argsort(-scores)[:k]
        return [{"rank": r + 1, "score": round(float(scores[i]), 4), **self.chunks[i]} for r, i in enumerate(top)]


# ---------------------------------------------------------------------------
# Evaluate
# ---------------------------------------------------------------------------

def results_path(config: str, k: int, n_cases: int) -> Path:
    """Unique per run (timestamp + case count) so earlier results are never overwritten."""
    return RESULTS_DIR / f"{datetime.now():%Y-%m-%d_%H%M%S}_{config}_n{n_cases}_k{k}.json"


def _norm(s: str) -> str:
    return " ".join(s.split())


def _matches(chunk: dict, ev: dict) -> bool:
    # Match on the chunk's OWN text (not the overlap copied from its neighbour)
    # so a hit is credited to the chunk that actually holds the evidence.
    own = chunk["text"][chunk.get("overlap_chars", 0):]
    return chunk["doc_id"] == ev["doc_id"] and _norm(ev["contains"]) in _norm(own)


def _supports(chunk: dict, sup: dict) -> bool:
    """A line of the chunk's own text starts with the metric label and holds the figure."""
    own = chunk["text"][chunk.get("overlap_chars", 0):]
    return any(l.strip().startswith(sup["label"]) and sup["figure"] in l for l in own.splitlines())


def evaluate(index: Index, k: int = 5, cases_path: Path = CASES_PATH) -> dict:
    """Retrieval-only scoring over eval/cases.jsonl. Two definitions of a hit:

    hit@k (strict)     - every cited evidence line is in a top-k chunk.
    support@k          - for every answer_support {label, figure}, some top-k
                         chunk has a line starting with that label and holding
                         that figure (e.g. "Net income 88,136" in the MD&A
                         summary table also supports the FY2024 answer, even
                         though the case cites the income statement on p80).
    trap_above_evidence- a trap chunk (pro forma / non-GAAP figure) ranks above
                         the cited evidence; answer generation must then rely on
                         the qualifier tags.
    This scores retrieval only; no answer is generated.
    """
    cases = [json.loads(l) for l in cases_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    rows = []
    for case in cases:
        ranked = index.search(case["question"], k=len(index.chunks))

        def best_rank(ev):
            return next((r["rank"] for r in ranked if _matches(r, ev)), None)

        ev_ranks = [best_rank(ev) for ev in case["evidence"]]
        trap_ranks = [best_rank(tr) for tr in case["traps"]]
        hit = bool(ev_ranks) and all(r is not None and r <= k for r in ev_ranks)
        first_ev = min((r for r in ev_ranks if r), default=None)
        sups = case.get("answer_support", [])
        sup_ranks = [next((r["rank"] for r in ranked if _supports(r, s)), None) for s in sups]
        rows.append({
            "id": case["id"], "category": case["category"], "question": case["question"],
            "evidence_ranks": ev_ranks, "trap_ranks": trap_ranks, "support_ranks": sup_ranks,
            f"hit@{k}": hit if case["evidence"] else None,
            f"support@{k}": all(r is not None and r <= k for r in sup_ranks) if sups else None,
            "trap_above_evidence": any(t and first_ev and t < first_ev for t in trap_ranks),
            "top": [{"rank": r["rank"], "score": r["score"], "chunk_id": r["chunk_id"],
                     "pages": f'{r["page_start"]}-{r["page_end"]}', "qualifiers": r["qualifiers"]}
                    for r in ranked[:k]],
        })
    scored = [r for r in rows if r[f"hit@{k}"] is not None]
    sup_scored = [r for r in rows if r[f"support@{k}"] is not None]
    return {
        "run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": MODEL_NAME, "k": k, "index_manifest": index.manifest,
        "summary": {
            "cases_with_evidence": len(scored),
            f"hit@{k}": sum(r[f"hit@{k}"] for r in scored),
            f"support@{k}": sum(r[f"support@{k}"] for r in sup_scored),
            "trap_above_evidence": sum(r["trap_above_evidence"] for r in rows),
            "note": f"{len(scored)} cases only - indicative, not statistically meaningful",
        },
        "cases": rows,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Build, search and evaluate the local embedding index.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    s = sub.add_parser("search"); s.add_argument("query"); s.add_argument("-k", type=int, default=5)
    e = sub.add_parser("evaluate"); e.add_argument("-k", type=int, default=5)
    for p in (b, s, e):
        # Default: chunks embedded with a context header (measured better, see README).
        p.add_argument("--no-header", dest="header", action="store_false",
                       help="use the plain index (chunks embedded without a context header)")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    try:
        if args.cmd == "build":
            print(json.dumps(build(header=args.header), indent=2))
        elif args.cmd == "search":
            for r in Index(header=args.header).search(args.query, args.k):
                own = r["text"][r["overlap_chars"]:]
                print(f'#{r["rank"]} {r["score"]:.3f} {r["chunk_id"]} p{r["page_start"]}-{r["page_end"]} '
                      f'{r["section"]} {r["qualifiers"] or ""}\n    {_norm(own)[:200]}')
        else:
            result = evaluate(Index(header=args.header), args.k)
            RESULTS_DIR.mkdir(parents=True, exist_ok=True)
            path = results_path(index_dir(INDEX_ROOT, args.header).name, args.k, len(result["cases"]))
            path.write_text(json.dumps(result, indent=2), encoding="utf-8")
            for r in result["cases"]:
                print(f'{r["id"]} {r["category"]:22} hit@{args.k}={r[f"hit@{args.k}"]!s:5} '
                      f'support@{args.k}={r[f"support@{args.k}"]!s:5} evidence_ranks={r["evidence_ranks"]} '
                      f'support_ranks={r["support_ranks"]} trap_ranks={r["trap_ranks"]}')
            print(json.dumps(result["summary"], indent=2))
            print(f"Saved {path}")
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        log.error("%s", e)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
