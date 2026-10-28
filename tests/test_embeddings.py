"""
COMPONENT : Evaluation / tests - embeddings and index
PURPOSE   : Check retrieval scoring (strict evidence hits, answer-support hits,
            trap ranking), chunk context headers and stale-index detection
            without loading a model.
INPUTS    : Synthetic chunks and cases written to a pytest tmp dir.
OUTPUTS   : pytest results.
DEPENDS ON: pytest, numpy
PANEL NOTE: MOCK: FakeIndex returns a fixed ranking instead of embedding
            queries, so these tests verify the scoring logic, not model
            quality. Measured retrieval results live in eval/results/.
"""

import json

import pytest

from fin_graph_rag import embeddings as em


def chunk(cid, doc, text, overlap_chars=0):
    return {"chunk_id": cid, "doc_id": doc, "text": text, "overlap_chars": overlap_chars,
            "page_start": 1, "page_end": 1, "qualifiers": []}


class FakeIndex:
    manifest = {"model": "fake"}

    def __init__(self, ranking):
        self.chunks = ranking

    def search(self, query, k=5):
        return [{"rank": i + 1, "score": 1.0 - i / 100, **c} for i, c in enumerate(self.chunks[:k])]


def write_cases(tmp_path, cases):
    p = tmp_path / "cases.jsonl"
    p.write_text("\n".join(json.dumps(c) for c in cases), encoding="utf-8")
    return p


def test_overlap_text_does_not_count_as_evidence():
    # The figure appears only in the overlap copied from the previous chunk.
    c = chunk("c1", "msft", "Net income $ 88,136\nOther text", overlap_chars=len("Net income $ 88,136\n"))
    ev = {"doc_id": "msft", "contains": "Net income $ 88,136"}
    assert not em._matches(c, ev)
    assert not em._supports(c, {"label": "Net income", "figure": "88,136"})
    assert em._matches(chunk("c2", "msft", "Net income $ 88,136"), ev)


def test_evaluate_strict_support_and_trap(tmp_path):
    trap = chunk("trap", "msft", "Net income 88,308 71,383")
    summary = chunk("mdna", "msft", "Net income 88,136 72,361 22%")
    statement = chunk("stmt", "msft", "Net income $ 88,136 $ 72,361 $ 72,738")
    filler = [chunk(f"f{i}", "msft", "unrelated prose") for i in range(10)]
    ranking = [trap, summary] + filler[:5] + [statement] + filler[5:]
    cases = write_cases(tmp_path, [{
        "id": "E01", "category": "direct_lookup", "question": "q",
        "evidence": [{"doc_id": "msft", "page": 80, "contains": "Net income $ 88,136 $ 72,361 $ 72,738"}],
        "traps": [{"doc_id": "msft", "page": 108, "contains": "Net income 88,308 71,383", "why": "pro forma"}],
        "answer_support": [{"label": "Net income", "figure": "88,136"}],
    }])
    r = em.evaluate(FakeIndex(ranking), k=5, cases_path=cases)
    row = r["cases"][0]
    assert row["evidence_ranks"] == [8] and row["hit@5"] is False      # cited line only at rank 8
    assert row["support_ranks"] == [2] and row["support@5"] is True    # MD&A table supports the answer
    assert row["trap_ranks"] == [1] and row["trap_above_evidence"] is True
    assert r["summary"]["hit@5"] == 0 and r["summary"]["support@5"] == 1


def test_stale_index_is_rejected(tmp_path, monkeypatch):
    chunks_path = tmp_path / "chunks.jsonl"
    chunks_path.write_text(json.dumps(chunk("c1", "d", "text")) + "\n", encoding="utf-8")
    d = em.index_dir(tmp_path / "index", header=True)
    d.mkdir(parents=True)
    (d / "manifest.json").write_text(json.dumps({"model": em.MODEL_NAME, "chunks_sha256": "old"}), encoding="utf-8")
    with pytest.raises(ValueError, match="changed since the index was built"):
        em.Index(index_root=tmp_path / "index", chunks_path=chunks_path, embedder=object())


def test_missing_index_is_reported(tmp_path):
    with pytest.raises(FileNotFoundError, match="No index"):
        em.Index(index_root=tmp_path / "none", embedder=object())


def meta_chunk(**kw):
    c = {"chunk_id": "c1", "doc_id": "microsoft_2024", "company": "microsoft", "fiscal_year": 2024,
         "section": "Item 8", "in_exhibits": False, "text": "Net income $ 88,136", "overlap_chars": 0,
         "tables": [{"title": "ITEM 8. FINANCIAL STATEMENTS / INCOME STATEMENTS / (In millions)",
                     "column_years": [2024, 2023, 2022], "qualifiers": []}]}
    c.update(kw)
    return c


def test_chunk_header_names_company_report_section_and_table():
    assert em.chunk_header(meta_chunk()) == (
        "Microsoft Corporation | FY2024 Form 10-K | Item 8 Financial Statements and Supplementary Data"
        " | Table: INCOME STATEMENTS (FY2024, FY2023, FY2022)")
    pro_forma = meta_chunk(tables=[{"title": "Following are ... unaudited pro forma / (In millions)",
                                    "column_years": [2024, 2023], "qualifiers": ["pro_forma", "unaudited"]}])
    assert em.chunk_header(pro_forma).endswith("(FY2024, FY2023) [pro forma, unaudited]")
    assert "| Exhibits" in em.chunk_header(meta_chunk(in_exhibits=True, tables=[]))


def test_header_is_shortened_to_fit_the_model_limit():
    tok = lambda s, add_special_tokens=True: {"input_ids": s.split() + ["[CLS]", "[SEP]"]}   # MOCK tokenizer
    c = meta_chunk(text=" ".join(["w"] * 20))
    full = len(em.chunk_header(c).split()) + 20 + 2
    no_table = len(em.chunk_header(c, with_tables=False).split()) + 20 + 2
    inputs, levels = em.embed_inputs([c], tok, limit=full, header=True)
    assert levels == {"full": 1} and inputs[0].startswith("Microsoft Corporation | FY2024")
    inputs, levels = em.embed_inputs([c], tok, limit=no_table, header=True)
    assert levels == {"no_table": 1} and "Table:" not in inputs[0]
    inputs, levels = em.embed_inputs([c], tok, limit=22, header=True)    # no room for any header
    assert levels == {"none_fits": 1} and inputs[0] == c["text"]
    # The stored chunk text itself is never modified.
    assert c["text"] == " ".join(["w"] * 20)
