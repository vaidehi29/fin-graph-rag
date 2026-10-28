"""
COMPONENT : Evaluation / tests - pipeline.txt phase 8 unit tests
PURPOSE   : The unit tests pipeline.txt phase 8 lists (PDF parsing shape, LLM
            JSON parsing with extra whitespace, graph facts before text in the
            context) mapped to this project's modules, plus CLAUDE.md
            failure-handling checks (malformed model output, missing key,
            unreachable database, missing index). No Neo4j, no API calls.
INPUTS    : synthetic PDFs / fake HTTP responses / fake LLM
OUTPUTS   : pytest results
DEPENDS ON: pytest, pdfplumber
PANEL NOTE: MOCK: every external service is faked here; live behaviour is
            covered by tests/test_integration.py (pytest -m integration).
            Module names differ from pipeline.txt (pdf_parser -> pdf_extractor,
            generation.answer -> fin_graph_rag.answer) because the project
            keeps its fin_graph_rag/ layout.
"""

import io
import json
import logging
import urllib.error

import pytest

from fin_graph_rag import answer as ans
from fin_graph_rag import entity_extractor as ex
from fin_graph_rag.llm import GeminiJSON
from fin_graph_rag.pdf_extractor import extract_pdf
from test_pdf_extractor import make_pdf


# --- pipeline.txt: pdf_parser.parse_pdf on a mocked 2-page PDF ------------------

def test_parse_two_page_pdf_returns_expected_shape(tmp_path):
    path = tmp_path / "apple_2024.pdf"
    path.write_bytes(make_pdf(["Total net sales 391,035 million dollars fiscal 2024",
                               "Net income 93,736 million dollars fiscal 2024"]))
    pages, report = extract_pdf(path, "apple", 2024)
    assert report.page_count == 2 and report.status == "ok"
    assert [p.page_number for p in pages] == [1, 2]
    for p in pages:
        assert (p.company, p.fiscal_year, p.source_file) == ("apple", 2024, "apple_2024.pdf")
        assert isinstance(p.text, str) and p.text.strip()


# --- pipeline.txt: extract_entities JSON parsing tolerates extra whitespace -----

def _gemini_reply(text):
    return lambda url, body: {"candidates": [{"content": {"parts": [{"text": text}]}}]}


def test_llm_json_with_extra_whitespace_is_parsed(tmp_path):
    llm = GeminiJSON("fake-model", api_key="k", cache_dir=tmp_path, sleep=lambda s: None,
                     post=_gemini_reply('\n\n   {"items": [ ]}  \n\t'))
    assert llm.generate("prompt", {"type": "object"}) == {"items": []}


def test_llm_skips_thinking_parts(tmp_path):
    post = lambda url, body: {"candidates": [{"content": {"parts": [
        {"text": "let me think...", "thought": True}, {"text": '{"answer": "ok", "insufficient_evidence": false}'}]}}]}
    llm = GeminiJSON("fake-model", api_key="k", cache_dir=tmp_path, sleep=lambda s: None, post=post)
    assert llm.generate("p", {})["answer"] == "ok"


def test_malformed_model_json_is_logged_and_extraction_continues(tmp_path, caplog):
    # pipeline.txt: "On parse failure, log the raw response and return empty lists."
    llm = GeminiJSON("fake-model", api_key="k", cache_dir=tmp_path, sleep=lambda s: None,
                     post=_gemini_reply("Sure! Here are the risks: ..."))
    chunks = tmp_path / "chunks.jsonl"
    chunks.write_text(json.dumps({"chunk_id": "a_c1", "doc_id": "apple_2024", "company": "apple", "fiscal_year": 2024,
                                  "section": "Item 1A", "in_exhibits": False, "page_start": 9, "text": "Risk text.",
                                  "overlap_chars": 0, "token_count": 3}) + "\n", encoding="utf-8")
    with caplog.at_level(logging.WARNING):
        entities, stats = ex.extract_text_entities(llm, chunks_path=chunks, tasks=("risks",))
    assert entities == [] and stats["rejected"] == {"risks:malformed_json_batch": 1}
    assert "Sure! Here are the risks" in caplog.text           # raw response logged
    assert llm.cache == {}                                      # not cached: the next run retries


def test_malformed_answer_is_a_model_error_not_an_answer():
    class Bad:
        model = "fake"

        def generate(self, *a):
            raise json.JSONDecodeError("Expecting value", "oops", 0)
    with pytest.raises(RuntimeError, match="malformed output"):
        ans.generate_answer("q", "context", Bad())


# --- pipeline.txt: build_context puts graph facts before text chunks -------------

def test_build_context_graph_facts_before_text():
    ctx = ans.build_context({
        "graph_facts": [{"text": "Apple Inc.: revenue, fiscal year 2025: $416,161 million (GAAP)",
                         "sources": ["Apple Annual Report FY2025, p.32"]}],
        "text_chunks": [{"text": "Total net sales increased 6%.", "overlap_chars": 0,
                         "source": "Apple Annual Report FY2025, p.25"}]})
    assert ctx.index("$416,161 million") < ctx.index("Total net sales increased")


# --- CLAUDE.md 5.1 #10: failure handling ---------------------------------------

def test_missing_gemini_key_gives_setup_message(tmp_path, monkeypatch):
    import fin_graph_rag.llm as llm_mod
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.setattr(llm_mod, "load_env", lambda *a, **k: None)
    with pytest.raises(RuntimeError, match="GEMINI_API_KEY is not set"):
        GeminiJSON("fake-model", cache_dir=tmp_path)


def test_rate_limit_then_success_is_retried(tmp_path):
    calls = []

    def post(url, body):
        calls.append(1)
        if len(calls) == 1:
            raise urllib.error.HTTPError(url, 503, "overloaded", {}, io.BytesIO(b'{"error": "overloaded"}'))
        return {"candidates": [{"content": {"parts": [{"text": '{"items": []}'}]}}]}
    llm = GeminiJSON("fake-model", api_key="k", cache_dir=tmp_path, sleep=lambda s: None, post=post)
    assert llm.generate("p", {}) == {"items": []} and len(calls) == 2


def test_unreachable_neo4j_fails_clearly_without_leaking_password(monkeypatch):
    import fin_graph_rag.graph_builder as gb
    monkeypatch.setattr(gb, "connection_settings", lambda: {
        "uri": "bolt://127.0.0.1:1", "auth": ("neo4j", "s3cret-pw-xyz"), "database": None})
    with pytest.raises(RuntimeError) as exc:
        with gb.driver_session():
            pass
    assert "Cannot connect to Neo4j at bolt://127.0.0.1:1" in str(exc.value)
    assert "s3cret-pw-xyz" not in str(exc.value)


def test_missing_vector_index_tells_how_to_build_it(tmp_path):
    from fin_graph_rag.embeddings import load_vectors
    with pytest.raises(FileNotFoundError, match="embeddings build"):
        load_vectors(index_root=tmp_path / "no-index")
