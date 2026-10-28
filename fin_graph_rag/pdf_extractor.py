"""
COMPONENT : PDF extraction
PURPOSE   : Extract page-level text and metadata from annual-report PDFs.
INPUTS    : PDFs in a raw directory named <company>_<fiscal_year>.pdf
            (e.g. apple_2023.pdf, microsoft_2025.pdf).
OUTPUTS   : <out_dir>/pages.jsonl  - one record per page
            <out_dir>/extraction_report.json - per-document counts, empty
            pages, errors, and source file hashes.
DEPENDS ON: pdfplumber
PANEL NOTE: Every downstream answer must trace back to a document and page.
            This step is where that provenance is created, so it records
            company, fiscal year, source file and page number on every page
            and reports (rather than hides) empty pages and failures.

Usage:
    python -m fin_graph_rag.pdf_extractor [--raw-dir DIR] [--out-dir DIR]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import pdfplumber

log = logging.getLogger("pdf_extractor")

# PANEL: Why derive company/year from the filename instead of the PDF text?
# Cover pages vary in layout; a strict naming convention is deterministic and
# auditable. Files that don't match are reported as skipped, not guessed.
# Note: the year is the company's FISCAL year (Apple FY ends late September,
# Microsoft FY ends June 30), not the calendar year.
FILENAME_RE = re.compile(r"^(?P<company>[a-z][a-z0-9-]*)_(?P<fy>(19|20)\d{2})\.pdf$", re.IGNORECASE)

# DEMO-KNOB: pages with fewer non-whitespace characters than this are flagged
# as empty (e.g. blank separators or image-only pages that need OCR).
MIN_PAGE_CHARS = 20


@dataclass
class PageRecord:
    doc_id: str
    company: str
    fiscal_year: int
    source_file: str
    page_number: int  # 1-based, matches PDF viewers
    text: str
    char_count: int
    is_empty: bool


@dataclass
class DocReport:
    doc_id: str
    source_file: str
    sha256: str
    company: str | None = None
    fiscal_year: int | None = None
    page_count: int = 0
    pages_with_text: int = 0
    empty_pages: list[int] = field(default_factory=list)
    page_errors: dict[int, str] = field(default_factory=dict)
    status: str = "pending"  # ok | partial | failed
    error: str | None = None


def parse_filename(name: str) -> tuple[str, int] | None:
    m = FILENAME_RE.match(name)
    if not m:
        return None
    return m.group("company").lower(), int(m.group("fy"))


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def extract_pdf(path: Path, company: str, fiscal_year: int) -> tuple[list[PageRecord], DocReport]:
    doc_id = f"{company}_{fiscal_year}"
    report = DocReport(doc_id=doc_id, source_file=path.name, sha256=sha256_of(path),
                       company=company, fiscal_year=fiscal_year)
    pages: list[PageRecord] = []
    try:
        with pdfplumber.open(path) as pdf:
            report.page_count = len(pdf.pages)
            for i, page in enumerate(pdf.pages, start=1):
                try:
                    text = page.extract_text() or ""
                except Exception as e:  # one bad page must not drop the whole document
                    report.page_errors[i] = f"{type(e).__name__}: {e}"
                    log.warning("%s page %d: extraction error: %s", path.name, i, e)
                    text = ""
                chars = len("".join(text.split()))
                is_empty = chars < MIN_PAGE_CHARS
                if is_empty:
                    report.empty_pages.append(i)
                pages.append(PageRecord(doc_id, company, fiscal_year, path.name, i,
                                        text, chars, is_empty))
    except Exception as e:
        report.status = "failed"
        report.error = f"{type(e).__name__}: {e}"
        log.error("%s: could not open/read PDF: %s", path.name, e)
        return [], report

    report.pages_with_text = report.page_count - len(report.empty_pages)
    # A document with page-level errors is "partial", never silently "ok".
    report.status = "partial" if report.page_errors else "ok"
    return pages, report


def run(raw_dir: Path, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf_paths = sorted(p for p in raw_dir.iterdir() if p.suffix.lower() == ".pdf") if raw_dir.is_dir() else []

    docs: list[DocReport] = []
    skipped: list[str] = []
    pages_path = out_dir / "pages.jsonl"
    with pages_path.open("w", encoding="utf-8") as out:
        for path in pdf_paths:
            parsed = parse_filename(path.name)
            if parsed is None:
                skipped.append(path.name)
                log.warning("Skipping %s: name must match <company>_<fiscal_year>.pdf", path.name)
                continue
            pages, report = extract_pdf(path, *parsed)
            for rec in pages:
                out.write(json.dumps(asdict(rec), ensure_ascii=False) + "\n")
            docs.append(report)
            log.info("%s: %s, %d pages, %d empty, %d errors", report.doc_id, report.status,
                     report.page_count, len(report.empty_pages), len(report.page_errors))

    summary = {
        "run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "raw_dir": str(raw_dir),
        "min_page_chars": MIN_PAGE_CHARS,
        "documents": [asdict(d) for d in docs],
        "skipped_files": skipped,
        "totals": {
            "documents_ok": sum(d.status == "ok" for d in docs),
            "documents_partial": sum(d.status == "partial" for d in docs),
            "documents_failed": sum(d.status == "failed" for d in docs),
            "pages": sum(d.page_count for d in docs),
            "pages_with_text": sum(d.pages_with_text for d in docs),
            "empty_pages": sum(len(d.empty_pages) for d in docs),
        },
    }
    (out_dir / "extraction_report.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if not pdf_paths:
        log.warning("No PDFs found in %s", raw_dir)
    return summary


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("Usage:")[0].strip(),
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw-dir", type=Path, default=Path(os.getenv("FGR_RAW_DIR", "data/raw")))
    ap.add_argument("--out-dir", type=Path, default=Path(os.getenv("FGR_EXTRACTED_DIR", "data/extracted")))
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    summary = run(args.raw_dir, args.out_dir)
    print(json.dumps(summary["totals"], indent=2))
    if summary["skipped_files"]:
        print("Skipped (bad filename):", ", ".join(summary["skipped_files"]))
    t = summary["totals"]
    # Non-zero exit when nothing was processed or anything failed, so scripts notice.
    if t["documents_ok"] + t["documents_partial"] == 0 or t["documents_failed"]:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
