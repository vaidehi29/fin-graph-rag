"""
COMPONENT : Evaluation / tests - preprocessing and chunking
PURPOSE   : Check cleaning, table detection, whole-table chunking, qualifier
            tagging and column-year detection in fin_graph_rag.preprocess.
INPUTS    : Synthetic page dicts shaped like pages.jsonl records.
OUTPUTS   : pytest results.
DEPENDS ON: pytest
PANEL NOTE: MOCK: page texts are synthetic, modelled on layouts seen in the
            Apple/Microsoft 10-Ks (footers, PDFCrowd watermark, tables split
            across page breaks, pro forma tables). See test_real_corpus.py
            for checks against the actual reports.
            MOCK: token counts here use whitespace word counts (set by the
            fixture below) so tests run offline and limits are easy to reason
            about; the real-corpus tests use the real tokenizer.
"""

import re

import pytest

from fin_graph_rag import preprocess as pp
from fin_graph_rag.preprocess import is_row, preprocess_doc


@pytest.fixture(autouse=True)
def word_tokens(monkeypatch):
    monkeypatch.setattr(pp, "_counter", lambda s: len(s.split()))

WM = "Explore our developer-friendly HTML to PDF API Printed using PDFCrowd HTML to PDF"
PROSE = ("Revenue increased due to growth in cloud services and higher sales of devices across all of our "
         "geographic markets during the year.")


def page(n, text, doc_id="msft_2024", company="msft", fy=2024):
    return {"doc_id": doc_id, "company": company, "fiscal_year": fy,
            "source_file": f"{doc_id}.pdf", "page_number": n, "text": text}


def test_is_row():
    assert is_row("Net income $ 88,136 $ 72,361 22%")
    assert is_row("Year Ended June 30, 2024 2023 2022")
    assert is_row("Marketing-related $ 11,619 24 years")
    assert is_row("As a percent of revenue 13% 12% 1ppt")
    assert not is_row("September 30, September 24, September 25,")
    assert not is_row("the U.S. dollar accounted for more than the entire year-over-year decrease as of September 30, 2023")


def test_cleaning_removes_watermark_footer_and_running_header():
    chunks, stats = preprocess_doc([
        page(1, f"PART II\nItem 7\n{PROSE}\n56\n{WM}"),
        page(2, f"{WM}"),   # blank HTML-print page
    ])
    text = "\n".join(c.text for c in chunks)
    assert "PDFCrowd" not in text and "PART II" not in text and "\n56" not in text
    assert chunks[0].section == "Item 7"
    assert chunks[0].printed_pages == ["56"]
    assert stats.pages == 2 and stats.pages_after_cleaning == 1
    assert stats.removed["watermark"] == 2


def test_table_split_across_pages_stays_in_one_chunk_with_title():
    chunks, _ = preprocess_doc([
        page(1, f"{PROSE}\nINCOME STATEMENTS\n(In millions, except per share amounts)\n"
                "Year Ended June 30, 2024 2023 2022\nRevenue:\nProduct $ 64,773 $ 64,699 $ 72,732\n"
                f"Service and other 180,349 147,216 125,538\n{WM}"),
        page(2, f"Total revenue 245,122 211,915 198,270\nNet income $ 88,136 $ 72,361 $ 72,738\n"
                f"Refer to accompanying notes.\n57\n{WM}"),
    ])
    [c] = [c for c in chunks if c.has_table]
    assert (c.page_start, c.page_end) == (1, 2)
    for s in ("INCOME STATEMENTS", "(In millions", "Year Ended June 30", "Total revenue 245,122",
              "Net income $ 88,136", "Refer to accompanying notes."):
        assert s in c.text
    t = c.tables[0]
    assert t["title"].startswith("INCOME STATEMENTS") and t["column_years"] == [2024, 2023, 2022]
    assert (t["page_start"], t["page_end"]) == (1, 2)


def test_pro_forma_intro_is_kept_and_tagged_without_absorbing_previous_table():
    text = "\n".join([
        "Following are the details of the purchase price allocated to the intangible assets acquired:",
        "(In millions, except average life) Amount Average Life",
        "Marketing-related $ 11,619 24 years",
        "Technology-based 9,689 4 years",
        "Following are the supplemental consolidated financial results of Microsoft Corporation on an unaudited pro forma",
        "basis, as if the acquisition had been consummated on July 1, 2022:",
        "(In millions, except per share amounts)",
        "Year Ended June 30, 2024 2023",
        "Revenue $ 247,442 $ 219,790",
        "Net income 88,308 71,383",
    ])
    chunks, _ = preprocess_doc([page(1, text)])
    tables = [t for c in chunks for t in c.tables]
    assert len(tables) == 2
    intangibles, pro_forma = tables
    assert intangibles["qualifiers"] == []
    assert pro_forma["title"].startswith("Following are the supplemental")
    assert pro_forma["qualifiers"] == ["pro_forma", "unaudited"]
    assert pro_forma["column_years"] == [2024, 2023]


def test_large_table_is_split_into_labelled_parts_with_repeated_header(monkeypatch):
    monkeypatch.setattr(pp, "MAX_TOKENS", 60)
    body = "\n".join(
        ["Operating activities:"] + [f"Operating item {i} {i},000 {i},500" for i in range(1, 7)]
        + ["Financing activities:"] + [f"Financing item {i} ({i},000) ({i},500)" for i in range(1, 15)])
    chunks, stats = preprocess_doc([page(1, f"{PROSE}\nCASH FLOWS STATEMENTS\n(In millions)\n"
                                            f"Year Ended June 30, 2024 2023\n{body}\n{PROSE}")])
    parts = [c for c in chunks if c.has_table]
    assert len(parts) > 1 and stats.split_tables == 1 and stats.tables == 1
    n = len(parts)
    for i, c in enumerate(parts, start=1):
        t = c.tables[0]
        assert (t["part"], t["n_parts"]) == (i, n)
        assert t["table_id"] == parts[0].tables[0]["table_id"]
        assert t["column_years"] == [2024, 2023]
        assert c.text.startswith(f"[Table part {i}/{n} | FY2024, FY2023 | CASH FLOWS STATEMENTS]")
        assert "(In millions)\nYear Ended June 30, 2024 2023" in c.text
        assert c.token_count <= 60
    # Every source row appears exactly once across the parts; the split prefers "Financing activities:".
    rows = [l for c in parts for l in c.text.splitlines() if " item " in l]
    assert len(rows) == len(set(rows)) == 20
    assert any(c.text.split("2023\n", 1)[1].startswith("Financing activities:") for c in parts[1:])


def test_stacked_year_table_parts_carry_their_own_year(monkeypatch):
    # Layout of Apple's segment table: one block per year under one title.
    # Row values start with their block's year (e.g. 202503) so we can check
    # that every row sits under the right year heading and label.
    monkeypatch.setattr(pp, "MAX_TOKENS", 60)
    blocks = [f"{y}\nAmericas Europe Total\n" + "\n".join(f"Line {i} $ {y}{i:02d} $ {i}" for i in range(1, 9))
              for y in (2025, 2024, 2023)]
    text = "SEGMENT INFORMATION\n(In millions)\n" + "\n".join(blocks)
    chunks, _ = preprocess_doc([page(1, text)])
    assert len(chunks) > 3
    assert chunks[0].tables[0]["column_years"] == [2025, 2024, 2023]
    for c in chunks:
        label_years = {int(y) for y in re.findall(r"FY(\d{4})", c.text.splitlines()[0])}
        year = None
        for l in c.text.splitlines()[1:]:
            if re.fullmatch(r"20\d\d", l):
                year = int(l)
            elif l.startswith("Line "):
                row_year = int(l.split()[3][:4])
                assert row_year == year, (c.chunk_id, l, year)   # right heading above the row
                assert row_year in label_years, (c.chunk_id, l)  # right label
        assert c.token_count <= 60


def test_long_prose_without_sentence_ends_is_split_evenly(monkeypatch):
    monkeypatch.setattr(pp, "TARGET_TOKENS", 40)
    monkeypatch.setattr(pp, "MAX_TOKENS", 60)
    monkeypatch.setattr(pp, "MIN_TOKENS", 20)
    # Exhibit-index style lines: no sentence ends and no trailing figures (so not a table).
    listing = "\n".join(f"Form of Supplemental Indenture between the Company and the Trustee, series {chr(65 + i)}"
                        for i in range(20))
    chunks, stats = preprocess_doc([page(1, listing)])
    assert len(chunks) > 1 and stats.oversized_chunks == 0
    assert all(20 <= c.token_count <= 60 for c in chunks)   # no tiny tail fragment
    assert "\n".join(c.text for c in chunks) == listing


def test_short_heading_moves_to_the_chunk_it_introduces(monkeypatch):
    monkeypatch.setattr(pp, "TARGET_TOKENS", 40)
    monkeypatch.setattr(pp, "MAX_TOKENS", 60)
    para_a = " ".join(["Alpha"] * 29) + "."
    heading = "Future operating results depend upon supply."
    para_b = " ".join(["Beta"] * 29) + "."
    chunks, _ = preprocess_doc([page(1, f"{para_a}\n{heading}\n{para_b}")])
    assert [c.text.split("\n")[0] for c in chunks] == [para_a, heading]
    assert chunks[1].text == f"{heading}\n{para_b}"


def test_tiny_fragment_is_merged_into_previous_chunk_of_same_section(monkeypatch):
    monkeypatch.setattr(pp, "TARGET_TOKENS", 40)
    monkeypatch.setattr(pp, "MIN_TOKENS", 10)
    para = " ".join(["Exhibit"] * 37) + "."
    chunks, _ = preprocess_doc([page(1, f"Item 15. Exhibits\n{para}\n** Furnished, not filed.\n"
                                        "Item 16. Form 10-K Summary\nNone.")])
    assert chunks[0].text.endswith("** Furnished, not filed.")
    assert chunks[0].section == "Item 15" and chunks[1].section == "Item 16"


def test_pages_after_signatures_are_flagged_as_exhibits():
    chunks, stats = preprocess_doc([
        page(1, PROSE),
        page(2, f"SIGNATURES\n{PROSE}"),
        page(3, "Exhibit 10.1\nRESTRICTED STOCK UNIT AWARD AGREEMENT terms and conditions apply here."),
    ])
    assert stats.signatures_page == 2 and stats.exhibit_pages == 1
    assert [c.in_exhibits for c in chunks if c.page_start == 3] == [True]
    assert not any(c.in_exhibits for c in chunks if c.page_end <= 2)
