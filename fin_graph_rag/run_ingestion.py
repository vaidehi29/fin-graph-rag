"""
COMPONENT : Knowledge graph - ingestion pipeline
PURPOSE   : Extract facts from the processed reports and load them into Neo4j.
INPUTS    : data/extracted/pages.jsonl, data/processed/chunks.jsonl, Neo4j
            settings in .env (not needed with --dry-run)
OUTPUTS   : data/processed/facts.jsonl and text_entities.jsonl (always), Neo4j nodes/relationships
            and TextChunk nodes with embeddings + vector index (unless --dry-run /
            --skip-embed), a printed verification summary
DEPENDS ON: fin_graph_rag.entity_extractor, fin_graph_rag.graph_builder,
            fin_graph_rag.llm (Gemini free tier; skipped with --skip-llm)
PANEL NOTE: --dry-run shows exactly what would be written, so the graph's
            contents can be reviewed before any database change. Writes use
            MERGE only, so re-running is safe and never duplicates.

Usage:
    python -m fin_graph_rag.run_ingestion --dry-run [--skip-llm] [--llm-limit N] [--skip-embed]
    python -m fin_graph_rag.run_ingestion
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from fin_graph_rag.embedders import QuotaExhausted
from fin_graph_rag.entity_extractor import extract_table_facts, extract_text_entities, fact_dict

log = logging.getLogger("run_ingestion")
FACTS_PATH = Path("data/processed/facts.jsonl")
ENTITIES_PATH = Path("data/processed/text_entities.jsonl")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Extract facts and load them into Neo4j.")
    ap.add_argument("--dry-run", action="store_true", help="extract and print; do not touch Neo4j")
    ap.add_argument("--skip-llm", action="store_true", help="table facts only; no Gemini calls")
    ap.add_argument("--llm-limit", type=int, help="max Gemini batches per task (for trial runs)")
    ap.add_argument("--skip-embed", action="store_true",
                    help="do not load TextChunk nodes / vectors (graph facts only)")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    try:
        facts, stats = extract_table_facts()
    except FileNotFoundError as e:
        log.error("%s - run pdf_extractor and preprocess first", e)
        return 1
    FACTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with FACTS_PATH.open("w", encoding="utf-8") as f:
        for fact in facts:
            f.write(json.dumps(fact_dict(fact), ensure_ascii=False) + "\n")
    print(json.dumps(stats, indent=2))
    print("facts by company/basis:", dict(Counter(f"{x.company}/{x.basis}" for x in facts)))
    print(f"Saved {FACTS_PATH}")

    entities = []
    if not args.skip_llm:
        from fin_graph_rag.llm import GeminiJSON
        try:
            llm = GeminiJSON()
            entities, estats = extract_text_entities(llm, limit=args.llm_limit)
        except (QuotaExhausted, RuntimeError) as e:
            # FALLBACK: table facts are still loaded; text entities resume next run from the cache.
            log.error("Text extraction stopped: %s", e)
            entities, estats = [], {"error": str(e)}
        else:
            estats["api_calls"] = llm.api_calls
            estats["model"] = llm.model
        with ENTITIES_PATH.open("w", encoding="utf-8") as f:
            for e in entities:
                f.write(json.dumps(asdict(e), ensure_ascii=False) + "\n")
        print(json.dumps(estats, indent=2))
        print(f"Saved {ENTITIES_PATH}")

    chunks, vectors, vmanifest = [], None, None
    if not args.skip_embed:
        # Reuse the locally built, evaluated vectors (free; no re-embedding).
        from fin_graph_rag.embeddings import load_vectors
        try:
            vmanifest, chunks, vectors = load_vectors()
        except (FileNotFoundError, ValueError) as e:
            log.error("%s", e)
            return 1
        print(f"Chunks to load: {len(chunks)} x {vectors.shape[1]}-d vectors from {vmanifest['model']} "
              f"(header {vmanifest.get('chunk_header')})")

    if args.dry_run:
        print("\n--dry-run: nothing written to Neo4j. Sample facts:")
        for fact in facts[:: max(1, len(facts) // 12)]:
            e = fact.evidence[0]
            print(f"  {fact.key} = {fact.value:,} {fact.unit}{' [CONFLICT]' if fact.conflict else ''}"
                  f"  <- {e.doc_id} p{e.page}: {e.row_text[:60]!r}")
        for ent in entities[:: max(1, len(entities) // 10)]:
            print(f"  {ent.kind} {ent.doc_id} p{ent.page}: {ent.fields}")
        return 0

    from fin_graph_rag.graph_builder import (driver_session, ensure_constraints, ensure_vector_index, graph_summary,
                                             write_chunks, write_entities, write_facts)
    try:
        with driver_session() as (driver, db):
            ensure_constraints(driver, db)
            n = write_facts(driver, facts, db)
            log.info("Merged %d facts", n)
            if entities:
                log.info("Merged text entities: %s", write_entities(driver, entities, db))
            if chunks:
                n = write_chunks(driver, chunks, vectors, vmanifest["model"], db)
                log.info("Merged %d text chunks with embeddings", n)
                print("vector index:", json.dumps(ensure_vector_index(driver, int(vectors.shape[1]), db)))
            print(json.dumps(graph_summary(driver, db), indent=2))
    except RuntimeError as e:
        log.error("%s", e)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
