"""
COMPONENT : Evaluation / tests - PDF extraction
PURPOSE   : Check page counts, metadata, empty-page reporting, bad filenames
            and corrupt-file handling for fin_graph_rag.pdf_extractor.
INPUTS    : Synthetic PDFs written to a pytest tmp dir.
OUTPUTS   : pytest results.
DEPENDS ON: pytest, pdfplumber
PANEL NOTE: MOCK: these PDFs are tiny synthetic fixtures, not the real annual
            reports. They verify extractor behaviour, not real-data counts.
"""

import json

from fin_graph_rag.pdf_extractor import parse_filename, run


def make_pdf(page_texts: list[str]) -> bytes:
    """Build a minimal valid PDF; an empty string yields a blank page."""
    n = len(page_texts)
    font_id = 3 + 2 * n
    objs = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [" + b" ".join(f"{3 + 2 * i} 0 R".encode() for i in range(n))
        + f"] /Count {n} >>".encode(),
    ]
    for i, text in enumerate(page_texts):
        stream = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode() if text else b""
        objs.append(f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
                    f"/Resources << /Font << /F1 {font_id} 0 R >> >> /Contents {4 + 2 * i} 0 R >>".encode())
        objs.append(f"<< /Length {len(stream)} >>\nstream\n".encode() + stream + b"\nendstream")
    objs.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")

    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for num, body in enumerate(objs, start=1):
        offsets.append(len(out))
        out += f"{num} 0 obj\n".encode() + body + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n".encode()
    out += b"".join(f"{o:010d} 00000 n \n".encode() for o in offsets)
    out += f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    return bytes(out)


def read_pages(out_dir):
    return [json.loads(l) for l in (out_dir / "pages.jsonl").read_text(encoding="utf-8").splitlines()]


def test_parse_filename():
    assert parse_filename("apple_2023.pdf") == ("apple", 2023)
    assert parse_filename("Microsoft_2025.PDF") == ("microsoft", 2025)
    assert parse_filename("apple-10k.pdf") is None
    assert parse_filename("apple_23.pdf") is None


def test_extracts_pages_with_metadata_and_flags_empty(tmp_path):
    raw, out = tmp_path / "raw", tmp_path / "out"
    raw.mkdir()
    (raw / "apple_2023.pdf").write_bytes(make_pdf([
        "Total net sales 383,285 million dollars fiscal 2023",
        "",
        "Products and Services segment revenue table",
    ]))

    summary = run(raw, out)
    pages = read_pages(out)

    assert [p["page_number"] for p in pages] == [1, 2, 3]
    assert all(p["company"] == "apple" and p["fiscal_year"] == 2023
               and p["source_file"] == "apple_2023.pdf" for p in pages)
    assert "383,285" in pages[0]["text"]
    assert pages[1]["is_empty"] and not pages[0]["is_empty"]

    doc = summary["documents"][0]
    assert doc["status"] == "ok"
    assert doc["page_count"] == 3 and doc["pages_with_text"] == 2
    assert doc["empty_pages"] == [2]
    assert len(doc["sha256"]) == 64
    assert summary["totals"]["pages"] == 3


def test_bad_filename_is_skipped_and_reported(tmp_path):
    raw, out = tmp_path / "raw", tmp_path / "out"
    raw.mkdir()
    (raw / "random-report.pdf").write_bytes(make_pdf(["hello world text here for test"]))

    summary = run(raw, out)
    assert summary["skipped_files"] == ["random-report.pdf"]
    assert summary["documents"] == []


def test_corrupt_pdf_is_reported_as_failed(tmp_path):
    raw, out = tmp_path / "raw", tmp_path / "out"
    raw.mkdir()
    (raw / "microsoft_2024.pdf").write_bytes(b"not a pdf at all")
    (raw / "apple_2024.pdf").write_bytes(make_pdf(["Apple fiscal 2024 annual report text"]))

    summary = run(raw, out)
    status = {d["doc_id"]: d["status"] for d in summary["documents"]}
    assert status == {"apple_2024": "ok", "microsoft_2024": "failed"}
    assert summary["totals"]["documents_failed"] == 1
    # The good document is still extracted.
    assert {p["doc_id"] for p in read_pages(out)} == {"apple_2024"}


def test_empty_raw_dir(tmp_path):
    summary = run(tmp_path / "missing", tmp_path / "out")
    assert summary["totals"]["pages"] == 0
    assert (tmp_path / "out" / "pages.jsonl").read_text() == ""
