"""
COMPONENT : Evaluation / tests - answer generation
PURPOSE   : Check context building (graph facts before text), the verbatim
            system prompt, citation parsing and validation, unverified-number
            detection, and confidence labelling - without calling Gemini.
INPUTS    : Synthetic retrieval results; a fake LLM returning fixed answers.
OUTPUTS   : pytest results.
DEPENDS ON: pytest
PANEL NOTE: MOCK: FakeLLM stands in for Gemini, so these tests check the
            verification logic around the model, not the model's quality.
            Measured answer quality is in eval/results/*_answers_*.
"""

from fin_graph_rag import answer as ans

RETRIEVAL = {
    "graph_facts": [{"text": "Microsoft Corporation: net income, fiscal year 2024: $88,136 million (GAAP)",
                     "sources": ["Microsoft Annual Report FY2024, p.80", "Microsoft Annual Report FY2024, p.61"]}],
    "text_chunks": [{"text": "Net income increased 22% to $88.1 billion.", "overlap_chars": 0,
                     "source": "Microsoft Annual Report FY2024, p.56"}],
    "sources": [],
}


class FakeLLM:
    model = "fake-model"

    def __init__(self, answer, insufficient=False):
        self.out = {"answer": answer, "insufficient_evidence": insufficient}
        self.calls = []

    def generate(self, prompt, schema, system):
        self.calls.append((prompt, system))
        return self.out


def test_context_lists_graph_facts_before_text_with_source_labels():
    ctx = ans.build_context(RETRIEVAL)
    assert ctx.index("GRAPH FACTS") < ctx.index("TEXT EXCERPTS")
    assert "[Source: Microsoft Annual Report FY2024, p.80]" in ctx
    assert ans.build_context({"graph_facts": [], "text_chunks": []}) == "(no evidence was retrieved)"


def test_system_prompt_is_pipeline_verbatim():
    llm = FakeLLM("x [Source: Microsoft Annual Report FY2024, p.80]")
    ans.generate_answer("q", ans.build_context(RETRIEVAL), llm)
    assert llm.calls[0][1] == ("You are a financial analyst assistant. Answer only using the provided context.\n"
                               "Always cite sources using the format [Source: Company Annual Report YEAR, p.N].\n"
                               "If the context doesn't contain enough information, say so explicitly.\n"
                               "Never invent numbers or facts.")


def test_citations_must_match_retrieved_sources():
    llm = FakeLLM("Net income was $88,136 million [Source: Microsoft Annual Report FY2024, p.80]; "
                  "see also [Source: Microsoft Annual Report FY2024, p.999].")
    r = ans.generate_answer("q", ans.build_context(RETRIEVAL), llm)
    assert r["sources"] == ["Microsoft Annual Report FY2024, p.80"]
    assert r["invalid_citations"] == ["Microsoft Annual Report FY2024, p.999"]   # never retrieved: flagged


def test_multi_page_shorthand_is_expanded_and_checked_per_page():
    assert ans.cited_sources("[Source: Apple Annual Report FY2025, p.15, 17]") == [
        "Apple Annual Report FY2025, p.15", "Apple Annual Report FY2025, p.17"]


def test_numbers_not_in_context_are_reported_unless_marked_calculated():
    ctx = ans.build_context(RETRIEVAL)
    assert ans.unverified_numbers("Net income was $88,136 million in fiscal 2024.", ctx) == []
    assert ans.unverified_numbers("Net income was $91,000 million.", ctx) == ["91000"]
    # A declared calculation and fiscal years are not flagged.
    assert ans.unverified_numbers("It rose by $15,775 million from FY2023 (calculated).", ctx) == []


def test_ask_sets_confidence_from_graph_facts(monkeypatch):
    import fin_graph_rag.hybrid as hybrid
    monkeypatch.setattr(hybrid, "hybrid_search", lambda d, q, k, db, **kw: RETRIEVAL)
    r = ans.ask("q", None, llm=FakeLLM("ok [Source: Microsoft Annual Report FY2024, p.56]"))
    assert r["confidence"] == "high"
    monkeypatch.setattr(hybrid, "hybrid_search", lambda d, q, k, db, **kw: {**RETRIEVAL, "graph_facts": []})
    r = ans.ask("q", None, llm=FakeLLM("ok [Source: Microsoft Annual Report FY2024, p.56]"))
    assert r["confidence"] == "medium"


def test_combined_citation_with_all_caps_company_names_is_split():
    from fin_graph_rag.answer import cited_sources
    a = "Revenue rose [Source: IBM Annual Report FY2024, p.167; NVIDIA Annual Report FY2024, p.50]."
    assert cited_sources(a) == ["IBM Annual Report FY2024, p.167", "NVIDIA Annual Report FY2024, p.50"]


def test_case_id_ranges():
    from fin_graph_rag.answer import parse_ids
    assert parse_ids("E40,E96-E98") == {"E40", "E96", "E97", "E98"}
    assert parse_ids("E1-E2") == {"E01", "E02"}


def test_fiscal_year_conventions_added_only_for_multi_company_evidence():
    from fin_graph_rag.answer import build_context, context_sources
    one = {"graph_facts": [{"company": "ibm", "text": "IBM revenue", "sources": ["IBM Annual Report FY2024, p.167"]}],
           "text_chunks": []}
    assert "FISCAL YEARS" not in build_context(one)
    two = {**one, "text_chunks": [{"company": "nvidia", "text": "Revenue $ 60,922", "overlap_chars": 0,
                                   "source": "NVIDIA Annual Report FY2024, p.50"}]}
    ctx = build_context(two)
    assert ctx.startswith("FISCAL YEARS") and "last Sunday in January" in ctx and "December 31" in ctx
    assert "NVIDIA Annual Report FY2025, p.57" in context_sources(ctx)        # citable like any source
