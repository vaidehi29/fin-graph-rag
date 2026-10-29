"""
COMPONENT : Embeddings and index
PURPOSE   : Embed every chunk with a sentence-embedding model, persist
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
DEPENDS ON: fin_graph_rag.embedders (local sentence-transformers or Gemini
            API), tokenizers, numpy
PANEL NOTE: Free and local: the model is downloaded once from Hugging Face
            and then runs offline on CPU - no API key, no per-call cost, no
            report text leaves the machine. bge-small-en-v1.5 uses the same
            bert-base-uncased WordPiece vocabulary and 512-token limit the
            chunker was sized against, which `build` re-verifies so nothing is
            silently truncated.

Usage:
    python -m fin_graph_rag.embeddings build [--no-header]
    python -m fin_graph_rag.embeddings append     # embed only new chunks (e.g. a new company)
    python -m fin_graph_rag.embeddings search "Microsoft net income fiscal 2024" [-k 5]
    python -m fin_graph_rag.embeddings evaluate [-k 5]
    Add --model gemini:gemini-embedding-001 to any command to use Gemini.
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

from fin_graph_rag.embedders import get_embedder

log = logging.getLogger("embeddings")

# DEMO-KNOB: embedding model - a sentence-transformers name or
# "gemini:<model>". Must accept at least 512 tokens.
MODEL_NAME = os.getenv("FGR_EMBED_MODEL", "BAAI/bge-small-en-v1.5")
# Inputs (chunk + header) are always fitted with this tokenizer and limit, for
# every backend, so all models embed exactly the same text.
INPUT_TOKENIZER = os.getenv("FGR_TOKENIZER", "bert-base-uncased")
INPUT_LIMIT = 512

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


def input_tokenizer():
    """Callable shaped like a HF tokenizer: tok(text, add_special_tokens) -> {"input_ids": [...]}."""
    from tokenizers import Tokenizer
    tok = Tokenizer.from_pretrained(INPUT_TOKENIZER)
    return lambda s, add_special_tokens=True: {"input_ids": tok.encode(s, add_special_tokens=add_special_tokens).ids}


# ---------------------------------------------------------------------------
# Chunk headers (experiment: does embedding document context help?)
# ---------------------------------------------------------------------------
# PANEL: A table chunk such as "Net income $ 88,136 $ 72,361 ..." says nothing
# about which company, report or statement it belongs to - that context lives
# in metadata the embedding never sees. The header puts it into the embedded
# text only; the stored chunk text (and its citations) are unchanged.

HEADER_VERSION = "v1"
COMPANY_NAMES = {"apple": "Apple Inc.", "microsoft": "Microsoft Corporation",
                 "ibm": "International Business Machines Corporation", "nvidia": "NVIDIA Corporation"}
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


def set_model(name: str | None) -> None:
    """Select the embedding model for this process (CLI --model)."""
    global MODEL_NAME
    if name:
        MODEL_NAME = name


def index_dir(index_root: Path, header: bool) -> Path:
    return index_root / (slug(MODEL_NAME) + ("-hdr" if header else ""))


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

def build(chunks_path: Path = CHUNKS_PATH, index_root: Path = INDEX_ROOT, header: bool = True) -> dict:
    chunks = load_chunks(chunks_path)
    if not chunks:
        raise ValueError(f"{chunks_path} is empty - nothing to index")
    embedder = get_embedder(MODEL_NAME)
    tok = input_tokenizer()
    inputs, header_levels = embed_inputs(chunks, tok, INPUT_LIMIT, header)

    # Models silently truncate long inputs; refuse to build instead. Use the
    # model's own tokenizer where we have it (local), else the input tokenizer.
    count = getattr(embedder, "count_tokens", None) or (lambda t: len(tok(t)["input_ids"]))
    lengths = [count(t) for t in inputs]
    too_long = [(c["chunk_id"], n) for c, n in zip(chunks, lengths) if n > embedder.max_tokens]
    if too_long:
        raise ValueError(f"{len(too_long)} chunks exceed the model's {embedder.max_tokens}-token limit "
                         f"(e.g. {too_long[:3]}); re-run preprocess with a matching tokenizer")

    t0 = time.time()
    vecs = embedder.embed_documents(inputs)
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
        **embedder.info,
        "chunk_header": HEADER_VERSION if header else None,
        "header_levels": header_levels if header else None,
        "example_input": inputs[min(100, len(inputs) - 1)][:300],
        "max_seq_length": embedder.max_tokens,
        "max_input_tokens_seen": max(lengths),
        "input_tokenizer": INPUT_TOKENIZER,
        "chunks_file": str(chunks_path),
        "chunks_sha256": sha256_of(chunks_path),
        "encode_seconds": round(seconds, 1),
        "api_calls": getattr(embedder, "api_calls", None),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    log.info("Indexed %d chunks (dim %d) in %.1fs -> %s", len(chunks), vecs.shape[1], seconds, out)
    return manifest


def chunk_line_hashes(chunks_path: Path) -> dict[str, str]:
    """chunk_id -> SHA-256 of its exact JSONL line (bytes, incl. line ending)."""
    out = {}
    for line in chunks_path.read_bytes().splitlines(keepends=True):
        if line.strip():
            out[json.loads(line)["chunk_id"]] = hashlib.sha256(line).hexdigest()
    return out


def append(chunks_path: Path = CHUNKS_PATH, index_root: Path = INDEX_ROOT, header: bool = True) -> dict:
    """Embed only chunks that are not in the index yet and append them.

    PANEL: Adding a company should not re-embed (or risk changing) the others.
    Before reusing any stored vector this proves the indexed chunks are
    byte-identical to what was embedded: on the first append, by recomputing the
    manifest's whole-file SHA-256 over those chunks' lines; afterwards, by a
    per-chunk hash (chunk_hashes.json). Indexed chunks that changed or vanished
    are dropped and re-embedded / removed - never silently kept.
    """
    d = index_dir(index_root, header)
    if not (d / "manifest.json").exists():
        raise FileNotFoundError(f"No index at {d} - run `python -m fin_graph_rag.embeddings build` first")
    manifest = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
    if manifest["model"] != MODEL_NAME:
        raise ValueError(f"Index built with {manifest['model']}, this run would use {MODEL_NAME}")
    if manifest.get("chunk_header") != (HEADER_VERSION if header else None):
        raise ValueError(f"Index header {manifest.get('chunk_header')!r} differs from this code's {HEADER_VERSION!r}")
    ids = json.loads((d / "chunk_ids.json").read_text(encoding="utf-8"))
    vecs = np.load(d / "embeddings.npy")
    lines = {json.loads(l)["chunk_id"]: l for l in chunks_path.read_bytes().splitlines(keepends=True) if l.strip()}
    current = chunk_line_hashes(chunks_path)

    hashes_path = d / "chunk_hashes.json"
    if hashes_path.exists():
        stored = json.loads(hashes_path.read_text(encoding="utf-8"))
        keep = [i for i, cid in enumerate(ids) if current.get(cid) == stored.get(cid)]
    else:
        # First append: the index was built from a file whose SHA-256 is in the manifest.
        if any(cid not in lines for cid in ids):
            raise ValueError("Indexed chunks are missing from chunks.jsonl - rebuild the index")
        old_sha = hashlib.sha256(b"".join(lines[cid] for cid in ids)).hexdigest()
        if old_sha != manifest["chunks_sha256"]:
            raise ValueError("Indexed chunks changed since the index was built - rebuild the index")
        keep = list(range(len(ids)))
    kept_ids = {ids[i] for i in keep}
    chunks = load_chunks(chunks_path)
    new = [c for c in chunks if c["chunk_id"] not in kept_ids]
    if not new and len(keep) == len(ids):
        log.info("Index already up to date (%d chunks)", len(ids))
        return manifest

    embedder = get_embedder(MODEL_NAME)
    tok = input_tokenizer()
    inputs, header_levels = embed_inputs(new, tok, INPUT_LIMIT, header)
    count = getattr(embedder, "count_tokens", None) or (lambda t: len(tok(t)["input_ids"]))
    lengths = [count(t) for t in inputs]
    too_long = [(c["chunk_id"], n) for c, n in zip(new, lengths) if n > embedder.max_tokens]
    if too_long:
        raise ValueError(f"{len(too_long)} chunks exceed the model's {embedder.max_tokens}-token limit "
                         f"(e.g. {too_long[:3]}); re-run preprocess with a matching tokenizer")
    t0 = time.time()
    new_vecs = embedder.embed_documents(inputs) if new else np.zeros((0, vecs.shape[1]), np.float32)
    seconds = time.time() - t0
    if new_vecs.shape[1] != vecs.shape[1]:
        raise ValueError(f"New vectors are {new_vecs.shape[1]}-d, index is {vecs.shape[1]}-d")

    all_ids = [ids[i] for i in keep] + [c["chunk_id"] for c in new]
    all_vecs = np.vstack([vecs[keep], new_vecs]).astype(np.float32)
    np.save(d / "embeddings.npy", all_vecs)
    (d / "chunk_ids.json").write_text(json.dumps(all_ids), encoding="utf-8")
    hashes_path.write_text(json.dumps({cid: current[cid] for cid in all_ids}), encoding="utf-8")
    levels = Counter(manifest.get("header_levels") or {})
    levels.update(header_levels)
    manifest.setdefault("appends", []).append({
        "appended_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "new_chunks": len(new), "dropped_chunks": len(ids) - len(keep),
        "companies": sorted({c["doc_id"].rsplit("_", 1)[0] for c in new}),
        "encode_seconds": round(seconds, 1), "max_input_tokens_seen": max(lengths, default=0),
    })
    manifest.update({"n_chunks": len(all_ids), "header_levels": dict(levels) if header else None,
                     "max_input_tokens_seen": max([manifest["max_input_tokens_seen"], *lengths]),
                     "chunks_sha256": sha256_of(chunks_path)})
    (d / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    log.info("Appended %d chunks (%d dropped) in %.1fs -> %s (%d total)",
             len(new), len(ids) - len(keep), seconds, d, len(all_ids))
    return manifest


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

def load_vectors(index_root: Path = INDEX_ROOT, chunks_path: Path = CHUNKS_PATH,
                 header: bool = True) -> tuple[dict, list[dict], np.ndarray]:
    """(manifest, chunks in row order, vectors) of a built index, after checking
    it matches the current model and chunks.jsonl. No model is loaded."""
    d = index_dir(index_root, header)
    if not (d / "manifest.json").exists():
        raise FileNotFoundError(f"No index at {d} - run `python -m fin_graph_rag.embeddings build"
                                f"{'' if header else ' --no-header'}`")
    manifest = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
    if manifest["model"] != MODEL_NAME:
        raise ValueError(f"Index built with {manifest['model']}, queries would use {MODEL_NAME}")
    # A stale index would return chunks that no longer match chunks.jsonl.
    if manifest["chunks_sha256"] != sha256_of(chunks_path):
        raise ValueError(f"{chunks_path} changed since the index was built - rebuild the index")
    vecs = np.load(d / "embeddings.npy")
    ids = json.loads((d / "chunk_ids.json").read_text(encoding="utf-8"))
    by_id = {c["chunk_id"]: c for c in load_chunks(chunks_path)}
    return manifest, [by_id[i] for i in ids], vecs


class Index:
    """Flat in-memory cosine index over one model's embeddings."""

    def __init__(self, index_root: Path = INDEX_ROOT, chunks_path: Path = CHUNKS_PATH, embedder=None,
                 header: bool = True):
        self.manifest, self.chunks, self.vecs = load_vectors(index_root, chunks_path, header)
        self.embedder = embedder or get_embedder(MODEL_NAME)
        dim = getattr(self.embedder, "dim", self.vecs.shape[1])
        if dim != self.vecs.shape[1]:
            raise ValueError(f"Index has {self.vecs.shape[1]}-d vectors but the query model gives {dim}-d")

    def search(self, query: str, k: int = 5) -> list[dict]:
        q = self.embedder.embed_query(query)
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
    a = sub.add_parser("append", help="embed only chunks not yet in the index (e.g. a newly added company)")
    s = sub.add_parser("search"); s.add_argument("query"); s.add_argument("-k", type=int, default=5)
    e = sub.add_parser("evaluate"); e.add_argument("-k", type=int, default=5)
    for p in (b, a, s, e):
        p.add_argument("--model", help=f"embedding model (default {MODEL_NAME}); "
                                       "e.g. gemini:gemini-embedding-001")
        # Default: chunks embedded with a context header (measured better, see README).
        p.add_argument("--no-header", dest="header", action="store_false",
                       help="use the plain index (chunks embedded without a context header)")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    set_model(args.model)

    try:
        if args.cmd == "build":
            print(json.dumps(build(header=args.header), indent=2))
        elif args.cmd == "append":
            print(json.dumps(append(header=args.header), indent=2))
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
