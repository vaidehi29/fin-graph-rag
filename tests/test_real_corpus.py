"""
COMPONENT : Evaluation / tests - real-corpus checks
PURPOSE   : Verify extraction/chunking outputs on the actual Apple and
            Microsoft 10-Ks, and that every eval case's cited evidence
            really appears on the cited page.
INPUTS    : data/extracted/pages.jsonl, data/processed/chunks.jsonl,
            eval/cases.jsonl
OUTPUTS   : pytest results (skipped when the data files are absent, since
            source reports are not committed to Git).
DEPENDS ON: pytest; outputs of pdf_extractor and preprocess
PANEL NOTE: Guards against fabricated citations: an eval case may only cite
            a page if the quoted text is on that page.
"""

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
PAGES = ROOT / "data/extracted/pages.jsonl"
CHUNKS = ROOT / "data/processed/chunks.jsonl"
CASES = ROOT / "eval/cases.jsonl"


def load(path):
    if not path.exists():
        pytest.skip(f"{path.relative_to(ROOT)} not present - run the pipeline first")
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def test_eval_case_evidence_is_on_cited_page():
    pages = {(p["doc_id"], p["page_number"]): p["text"] for p in load(PAGES)}
    for case in load(CASES):
        for ev in case["evidence"] + case["traps"]:
            assert ev["contains"] in pages[(ev["doc_id"], ev["page"])], (case["id"], ev)


def test_pro_forma_tables_are_tagged_with_their_column_years():
    chunks = load(CHUNKS)
    hits = [(c["doc_id"], t) for c in chunks for t in c["tables"]
            if "Net income 88,308" in c["text"] and "pro_forma" in t["qualifiers"]]
    assert {d for d, _ in hits} == {"microsoft_2024", "microsoft_2025"}
    for _, t in hits:
        assert t["qualifiers"] == ["pro_forma", "unaudited"]
        assert t["column_years"] == [2024, 2023]
        assert "unaudited pro forma" in t["title"]


def test_chunking_is_lossless_and_within_size():
    from collections import defaultdict
    from fin_graph_rag.preprocess import DocStats, MAX_TOKENS, clean_doc, count_tokens, to_units
    by_doc = defaultdict(list)
    for p in load(PAGES):
        by_doc[p["doc_id"]].append(p)
    for doc_id, pages in by_doc.items():
        lines = clean_doc(pages, DocStats(doc_id))
        # Dropping generated labels and repeated headers must give back the cleaned text exactly.
        assert [l.text for u in to_units(lines) for l in u.lines[u.repeated:]] == [l.text for l in lines]
    # Re-count with the real tokenizer; +2 for [CLS]/[SEP] must stay within a 512-token model.
    for c in load(CHUNKS):
        assert count_tokens(c["text"]) == c["token_count"] <= MAX_TOKENS
        assert c["token_count"] + 2 <= 512


def test_split_table_parts_are_labelled_with_the_years_they_contain():
    import re
    from fin_graph_rag.preprocess import SUB_YEAR_RE
    parts = [c for c in load(CHUNKS) if any(t["n_parts"] > 1 for t in c["tables"])]
    assert parts
    for c in parts:
        label, *lines = c["text"].splitlines()
        assert label.startswith("[Table part ")
        sub_years = {int(m.group(1)) for l in lines if (m := SUB_YEAR_RE.match(l))}
        assert sub_years <= {int(y) for y in re.findall(r"FY(\d{4})", label)}, c["chunk_id"]


def test_income_statements_are_whole_with_title():
    chunks = load(CHUNKS)
    expected = {"microsoft_2024": "Net income $ 88,136", "apple_2023": "Net income $ 96,995"}
    for doc_id, net_income in expected.items():
        [c] = [c for c in chunks if c["doc_id"] == doc_id and any(
            "INCOME STATEMENTS" in t["title"] or "STATEMENTS OF OPERATIONS" in t["title"] for t in c["tables"])
            and net_income in c["text"]]
        assert "(In millions" in c["text"] and "Diluted" in c["text"]


def test_overlap_is_20_to_30_tokens_copied_from_previous_chunk():
    from fin_graph_rag.preprocess import OVERLAP_MAX_TOKENS, OVERLAP_MIN_TOKENS, count_tokens
    chunks = load(CHUNKS)
    by_id = {c["chunk_id"]: c for c in chunks}
    with_overlap = [c for c in chunks if c["overlap_chars"]]
    assert len(with_overlap) > len(chunks) / 2
    for c in with_overlap:
        overlap = c["text"][:c["overlap_chars"] - 1]
        src = by_id[c["overlap_from"]]
        assert src["text"][src["overlap_chars"]:].endswith(overlap), c["chunk_id"]
        assert OVERLAP_MIN_TOKENS <= count_tokens(overlap) <= OVERLAP_MAX_TOKENS
        assert not c["text"][c["overlap_chars"]:].startswith("[Table part")


def test_ibm_and_nvidia_filing_layout():
    """IBM's Exhibit 13 (incorporated Items 7/8) is report content, not exhibits;
    NVIDIA's "Signatures" page is found; both have titled income statements."""
    import json as _json
    chunks = load(CHUNKS)
    report = _json.loads((ROOT / "data/processed/chunking_report.json").read_text(encoding="utf-8"))
    docs = {d["doc_id"]: d for d in report["documents"]}
    if "ibm_2025" not in docs or "nvidia_2025" not in docs:
        pytest.skip("IBM / NVIDIA not processed")
    for y in (2023, 2024, 2025):
        assert docs[f"ibm_{y}"]["signatures_page"] == 26 and docs[f"ibm_{y}"]["exhibit13_pages"] > 100
    assert [docs[f"nvidia_{y}"]["signatures_page"] for y in (2023, 2024, 2025)] == [91, 84, 86]
    ibm_is = [c for c in chunks if c["doc_id"] == "ibm_2025" and c["section"] == "Exhibit 13 Consolidated Income Statement"]
    assert ibm_is and not any(c["in_exhibits"] for c in ibm_is)
    assert any(t["title"].startswith("Consolidated Income Statement") and t["column_years"] == [2025, 2024, 2023]
               for c in ibm_is for t in c["tables"])
    nv = [c for c in chunks if c["doc_id"] == "nvidia_2025" and "Revenue $ 130,497 $ 60,922 $ 26,974" in c["text"]]
    assert any("Consolidated Statements of Income" in t["title"] and t["column_years"] == [2025, 2024, 2023]
               for c in nv for t in c["tables"])
