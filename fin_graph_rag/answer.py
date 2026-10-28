"""
COMPONENT : Answer generation
PURPOSE   : Answer a question from retrieved evidence only: build a context
            (graph facts first, then text excerpts, each with its source),
            ask Gemini (free tier) with the pipeline.txt system prompt, then
            check the answer - every citation must be a retrieved source and
            every number is looked up in the context.
INPUTS    : question; Neo4j driver (via fin_graph_rag.hybrid.hybrid_search)
OUTPUTS   : {"answer", "sources", "model", "confidence", "insufficient_evidence",
             "invalid_citations", "unverified_numbers", ...}; `--evaluate`
            writes eval/results/<timestamp>_answers_n<N>.json
DEPENDS ON: fin_graph_rag.hybrid, fin_graph_rag.llm (Gemini free tier)
PANEL NOTE: A fluent answer is not evidence that it is right. The model is
            told to use only the context, and the code then verifies what it
            can: citations must match retrieved sources exactly (a fabricated
            source is flagged, not shown as valid), and numbers that do not
            appear in the context are reported as unverified (a legitimate
            calculation must say "(calculated)"). Report text and questions are
            sent to Google's free tier, which may use them to improve products.

Usage:
    python -m fin_graph_rag.answer "What was Microsoft's net income in fiscal 2024?"
    python -m fin_graph_rag.answer --evaluate
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys

from fin_graph_rag import embeddings as em

log = logging.getLogger("answer")

# pipeline.txt: "Use this system prompt verbatim".
SYSTEM_PROMPT = (
    "You are a financial analyst assistant. Answer only using the provided context.\n"
    "Always cite sources using the format [Source: Company Annual Report YEAR, p.N].\n"
    "If the context doesn't contain enough information, say so explicitly.\n"
    "Never invent numbers or facts."
)
# Extra rules go in the user message so the system prompt stays verbatim.
INSTRUCTIONS = (
    "Answer in 1-5 sentences. State figures exactly as written in the context, with the company, the fiscal "
    "year and the unit (for example: $88,136 million). Fiscal years are not calendar years. If a figure is pro "
    "forma, non-GAAP, or the reports disagree, say so. If you calculate something (a difference, a sum, a "
    "percentage change), show the inputs and write (calculated) after the result. After each claim put a "
    "citation copied exactly from the context, e.g. [Source: Apple Annual Report FY2024, p.32]. If the context "
    "does not answer the question, say that the provided reports do not contain enough information, and give "
    "no figures. Set insufficient_evidence to true in that case."
)
ANSWER_SCHEMA = {"type": "object", "properties": {
    "answer": {"type": "string"}, "insufficient_evidence": {"type": "boolean"}},
    "required": ["answer", "insufficient_evidence"]}
# DEMO-KNOB: answer model (free tier). Same family as extraction by default.
ANSWER_MODEL = os.getenv("FGR_ANSWER_MODEL", os.getenv("FGR_EXTRACT_MODEL", "gemini-3.5-flash-lite"))
MAX_FACTS = 40
CITATION_RE = re.compile(r"\[Source:\s*([^\]]+)\]")
NUMBER_RE = re.compile(r"\d[\d,]*(?:\.\d+)?")


def build_context(result: dict, max_facts: int = MAX_FACTS) -> str:
    """Graph facts first (exact, higher confidence), then text excerpts, each with its source label."""
    parts = []
    facts = result.get("graph_facts", [])[:max_facts]
    if facts:
        parts.append("GRAPH FACTS (exact figures from the reports' tables and verified extractions):")
        for f in facts:
            parts.append(f"- {f['text']} " + " ".join(f"[Source: {s}]" for s in f["sources"][:2]))
    chunks = result.get("text_chunks", [])
    if chunks:
        parts.append("\nTEXT EXCERPTS:")
        for c in chunks:
            own = " ".join(c["text"][c.get("overlap_chars", 0):].split())
            parts.append(f"[Source: {c['source']}]\n{own}\n")
    return "\n".join(parts) if parts else "(no evidence was retrieved)"


def context_sources(context: str) -> set[str]:
    return {s.strip() for s in CITATION_RE.findall(context)}


def cited_sources(answer: str) -> list[str]:
    """Individual source labels cited in the answer.

    Handles "[Source: A, p.1; B, p.2]" and the shorthand "[Source: A, p.15, 17]",
    which is expanded to one label per page so each page is checked on its own.
    """
    out = []
    for group in CITATION_RE.findall(answer):
        for s in re.split(r";\s*(?=[A-Z][a-z]+ Annual Report)", group):
            s = s.strip()
            m = re.fullmatch(r"(.+?, p\.)\s*(\d+(?:\s*,\s*\d+)+)", s)
            labels = [f"{m.group(1)}{n.strip()}" for n in m.group(2).split(",")] if m else [s]
            for label in labels:
                if label and label not in out:
                    out.append(label)
    return out


def _numbers(text: str) -> set[str]:
    return {n.replace(",", "").rstrip(".") for n in NUMBER_RE.findall(text)}


def unverified_numbers(answer: str, context: str) -> list[str]:
    """Numbers in the answer that appear nowhere in the context.

    Citations, fiscal years and numbers inside "(calculated)" sentences are
    ignored: years are labels, and calculations are declared as such.
    """
    body = CITATION_RE.sub(" ", answer)
    sentences = re.split(r"(?<=[.!?])\s+", body)
    body = " ".join(s for s in sentences if "(calculated)" not in s)
    known = _numbers(context)
    out = []
    for n in sorted(_numbers(body)):
        if re.fullmatch(r"(19|20)\d{2}", n) or n in known:
            continue
        out.append(n)
    return out


def generate_answer(question: str, context: str, llm=None) -> dict:
    if llm is None:
        from fin_graph_rag.llm import GeminiJSON
        llm = GeminiJSON(model=ANSWER_MODEL)
    prompt = f"Question: {question}\n\nContext:\n{context}\n\nInstructions: {INSTRUCTIONS}"
    try:
        out = llm.generate(prompt, ANSWER_SCHEMA, SYSTEM_PROMPT)
    except json.JSONDecodeError as e:
        # Reported as a model failure (the UI shows it as such), never as an answer.
        raise RuntimeError(f"The answer model returned malformed output ({e})") from None
    answer = out.get("answer", "").strip()
    allowed = context_sources(context)
    cited = cited_sources(answer)
    return {
        "answer": answer,
        "sources": [s for s in cited if s in allowed],
        "invalid_citations": [s for s in cited if s not in allowed],     # cited but never retrieved
        "unverified_numbers": unverified_numbers(answer, context),
        "insufficient_evidence": bool(out.get("insufficient_evidence")),
        "model": llm.model,
    }


def ask(question: str, driver, database=None, llm=None, top_k: int = 5,
        default_companies: list[str] | None = None) -> dict:
    """Single entry point: hybrid retrieval, then a verified answer."""
    from fin_graph_rag.hybrid import hybrid_search
    retrieval = hybrid_search(driver, question, top_k, database, default_companies=default_companies)
    result = generate_answer(question, build_context(retrieval), llm)
    # pipeline.txt: graph facts present -> high confidence; text only -> medium.
    result["confidence"] = "high" if retrieval["graph_facts"] else "medium"
    result["retrieved_sources"] = retrieval["sources"]
    result["n_graph_facts"] = len(retrieval["graph_facts"])
    return result


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def _has_figure(answer: str, figure: str) -> bool:
    """Expected figure appears in the answer ("88,136" or "88136"; "137.4 billion" as written)."""
    a = answer.replace(",", "")
    return figure.replace(",", "") in a


DECLINE_RE = re.compile(r"not contain|does not (include|provide|mention)|no (information|data)|not enough|"
                        r"insufficient|cannot (be )?(determine|answer)|not available|unable to", re.I)


def evaluate(driver, database=None, llm=None, cases_path=em.CASES_PATH, extra_questions=()) -> dict:
    cases = [json.loads(l) for l in cases_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    if llm is None:
        from fin_graph_rag.llm import GeminiJSON
        llm = GeminiJSON(model=ANSWER_MODEL)
    rows = []
    for case in cases:
        r = ask(case["question"], driver, database, llm)
        sups = case.get("answer_support", [])
        # Trap figure = first large number of the trap line, e.g. "88,308" from "Net income 88,308 71,383".
        trap_figs = [f for f in (re.findall(r"\d[\d,]{3,}", t["contains"])[:1] for t in case.get("traps", [])) for f in f]
        is_missing = case["category"] == "missing_evidence"
        correct = all(_has_figure(r["answer"], s["figure"]) for s in sups) if sups else None
        asks_pro_forma = "would" in case["question"].lower() or "pro forma" in case["question"].lower()
        trap_used = any(_has_figure(r["answer"], f) for f in trap_figs) and not asks_pro_forma
        declined = r["insufficient_evidence"] or bool(DECLINE_RE.search(r["answer"]))
        rows.append({
            "id": case["id"], "category": case["category"], "question": case["question"],
            "answer": r["answer"], "confidence": r["confidence"], "sources": r["sources"],
            "has_citation": bool(r["sources"]), "invalid_citations": r["invalid_citations"],
            "unverified_numbers": r["unverified_numbers"], "expected_figures_present": correct,
            "trap_figure_used": trap_used if trap_figs else None,
            "declined": declined, "correct_decline": (declined and not _numbers(CITATION_RE.sub("", r["answer"]))
                                                      - {"2025", "2026"}) if is_missing else None,
        })
    for q in extra_questions:
        r = ask(q, driver, database, llm)
        rows.append({"id": "extra", "category": "pipeline_sample", "question": q, "answer": r["answer"],
                     "confidence": r["confidence"], "sources": r["sources"], "has_citation": bool(r["sources"]),
                     "invalid_citations": r["invalid_citations"], "unverified_numbers": r["unverified_numbers"]})
    answerable = [r for r in rows if r.get("expected_figures_present") is not None]
    missing = [r for r in rows if r.get("correct_decline") is not None]
    trap_rows = [r for r in rows if r.get("trap_figure_used") is not None]
    not_missing = [r for r in rows if r["category"] != "missing_evidence"]
    return {
        "model": ANSWER_MODEL if llm is None else llm.model,
        "summary": {
            "questions": len(rows),
            "answers_with_valid_citation": f"{sum(r['has_citation'] for r in not_missing)}/{len(not_missing)}",
            "invalid_citations_total": sum(len(r["invalid_citations"]) for r in rows),
            "answers_with_unverified_numbers": sum(bool(r["unverified_numbers"]) for r in rows),
            "expected_figures_present": f"{sum(r['expected_figures_present'] for r in answerable)}/{len(answerable)}",
            "trap_figure_used": f"{sum(r['trap_figure_used'] for r in trap_rows)}/{len(trap_rows)}",
            "missing_evidence_declined_without_figures": f"{sum(r['correct_decline'] for r in missing)}/{len(missing)}",
            "note": "automatic checks; expected figure present != fully correct answer - read the answers",
        },
        "cases": rows,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Answer questions from the reports with citations.")
    ap.add_argument("question", nargs="?")
    ap.add_argument("--evaluate", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    if not args.evaluate and not args.question:
        ap.error("give a question or --evaluate")

    from fin_graph_rag.embedders import QuotaExhausted
    from fin_graph_rag.graph_builder import driver_session
    try:
        with driver_session() as (driver, db):
            if args.evaluate:
                result = evaluate(driver, db, extra_questions=["Compare Apple and Microsoft revenue",
                                                               "What risks does Apple mention?"])
                path = em.results_path("answers", 0, len(result["cases"])).with_name(
                    em.results_path("answers", 0, len(result["cases"])).name.replace("_k0", ""))
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(result, indent=2), encoding="utf-8")
                print(json.dumps(result["summary"], indent=2))
                print(f"Saved {path}")
                return 0
            r = ask(args.question, driver, db)
            print(r["answer"])
            print(f"\nconfidence: {r['confidence']}   model: {r['model']}")
            print("sources:", *r["sources"], sep="\n  - ")
            if r["invalid_citations"]:
                print("WARNING - cited but not retrieved:", r["invalid_citations"])
            if r["unverified_numbers"]:
                print("WARNING - numbers not found in the retrieved context:", r["unverified_numbers"])
    except (RuntimeError, QuotaExhausted) as e:
        log.error("%s", e)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
