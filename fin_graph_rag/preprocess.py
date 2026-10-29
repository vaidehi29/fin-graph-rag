"""
COMPONENT : Preprocessing and chunking
PURPOSE   : Clean extracted 10-K pages (watermarks, page footers, running
            headers), detect tables, and pack text into retrievable chunks
            in which every table is kept whole together with its title,
            unit line, introductory sentence and footnotes.
INPUTS    : <in_dir>/pages.jsonl from fin_graph_rag.pdf_extractor
OUTPUTS   : <out_dir>/chunks.jsonl         - one record per chunk
            <out_dir>/chunking_report.json - config, per-document counts,
            removed-line counts, qualifier-flagged tables.
DEPENDS ON: tokenizers (HF WordPiece vocabulary, downloaded once and cached)
PANEL NOTE: A number is only useful if it stays attached to its metric label,
            period and unit. Splitting a table across chunks (or across a PDF
            page break) separates "88,136" from "Net income" and "(In
            millions)". Tables are therefore atomic units, and tables whose
            figures are pro forma / unaudited / non-GAAP are tagged so later
            stages can tell them apart from reported GAAP figures.

Usage:
    python -m fin_graph_rag.preprocess [--in-dir DIR] [--out-dir DIR] [--companies ibm nvidia]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

log = logging.getLogger("preprocess")

# --- chunk size (measured in tokens, not characters) -----------------------
# PANEL: Why tokens? Embedding models truncate at a token limit (commonly 512)
# and numbers tokenize poorly ("383,285" -> 4 WordPiece tokens), so a
# character budget silently let table chunks reach 825 tokens.
# DEMO-KNOB: tokenizer used for counting. bert-base-uncased WordPiece is the
# vocabulary of common 512-token embedding models (all-MiniLM, BGE-small/base,
# E5-small/base); set FGR_TOKENIZER to match the embedding model once chosen.
TOKENIZER_NAME = os.getenv("FGR_TOKENIZER", "bert-base-uncased")
# Hard cap on content tokens per chunk: 512 minus [CLS]/[SEP] and headroom
# for an instruction prefix such as E5's "passage: ".
MAX_TOKENS = 500
# DEMO-KNOB: soft size for prose chunks; a chunk is closed once adding the
# next paragraph would exceed it.
TARGET_TOKENS = 350
# Chunks smaller than this are merged into a same-section neighbour when it fits.
MIN_TOKENS = 50
# A trailing prose unit this small (typically a risk-factor heading) is moved
# into the next chunk so it stays with the text it introduces.
HEADING_TOKENS = 60
# DEMO-KNOB: a chunk that starts with prose is prefixed with the last 20-30
# tokens of the previous chunk in the same section (starting at a sentence
# start when possible), so a sentence cut at a chunk boundary is still
# retrievable. Packing reserves OVERLAP_MAX_TOKENS so chunks stay <= MAX_TOKENS.
OVERLAP_MIN_TOKENS = 20
OVERLAP_MAX_TOKENS = 30
# Tables longer than MAX_TOKENS are split at row boundaries into parts. Each
# part repeats the table's title/unit/year-header lines and starts with a
# generated "[Table part i/n | FY... | title]" label, so no part loses context.
# DEMO-KNOB: minimum number of numeric rows for a run of lines to be a table.
MIN_TABLE_ROWS = 2
# Lines up to this length without a sentence ending may be table sub-labels
# ("Net sales:", wrapped row labels) or table titles.
SHORT_LINE = 90
MAX_BRIDGE_LINES = 2     # consecutive label-only lines allowed inside a table
MAX_HEADER_LINES = 10    # lines walked back to find a table's title / intro
MAX_FOOTNOTE_LINES = 8

# --- cleaning patterns (derived from inspecting the six reports) ----------
# FALLBACK: Microsoft FY24/FY25 PDFs were printed from HTML with PDFCrowd,
# which stamps this line on every page.
WATERMARK_RE = re.compile(r"^Explore our developer-friendly HTML to PDF API.*PDFCrowd", re.I)
# Apple footer: "Apple Inc. | 2023 Form 10-K | 21"
FOOTER_RE = re.compile(r"^.{1,40} \| \d{4} Form 10-K \| (?P<page>\d{1,3})$")
PAGE_NUM_RE = re.compile(r"^\d{1,3}$")
# Microsoft running headers at page top: "PART II" / "Item 8"
PART_RE = re.compile(r"^PART (I|II|III|IV)$")
RUNNING_ITEM_RE = re.compile(r"^Item (\d{1,2}[A-C]?)$")
# Body headings: "Item 7. Management's ..." / "ITEM 8. FINANCIAL STATEMENTS ..."
# Lines ending in a digit are table-of-contents entries and are ignored.
ITEM_HEADING_RE = re.compile(r"^item\s+(\d{1,2}[A-C]?)\.\s+\S.*\D$", re.I)
SIGNATURES_RE = re.compile(r"^SIGNATURES?$")
# The signatures page proper says "Pursuant to the requirements of Section 13 or
# 15(d) ...". PANEL: IBM's table of contents has a bare "SIGNATURES" line on
# page 2, and NVIDIA FY2024/FY2025 head the page "Signatures" - a heading alone
# flagged 343 of IBM's 345 pages as exhibits and none of NVIDIA's.
SIGNATURES_ANY_RE = re.compile(r"^signatures?$", re.I)
SIGNED_RE = re.compile(r"pursuant to the requirements of section 13 or 15\s*\(d\)", re.I)
# Filings printed from EDGAR HTML (IBM, NVIDIA) start most pages with this link.
TOC_LINK_RE = re.compile(r"^Table of Contents$")
# Exhibit 13 = the annual report to shareholders. IBM's 10-K incorporates its
# MD&A (Item 7) and financial statements (Item 8) from it by reference, so its
# pages are report content, not exhibits. It ends where the next exhibit starts.
EX13_RE = re.compile(r"\bExhibit 13\b")
EXHIBIT_START_RE = re.compile(r"^EXHIBIT \d", re.I)
# Exhibit 13 running header: "6 Management Discussion" / "Management Discussion 7"
# followed by "International Business Machines Corporation and Subsidiary Companies".
AR_HEADER_RE = re.compile(r"^(?:(?P<pre>\d{1,3}) )?(?P<title>[A-Z][A-Za-z ,&'’-]*[a-z])(?: (?P<post>\d{1,3}))?$")
SUBSIDIARIES_RE = re.compile(r"^.+ and Subsidiar(y Companies|ies)$")

# --- table detection -------------------------------------------------------
# A "value" token: 1,234 | $ | (4) | 22% | 9.68 | — | * . A trailing comma is
# deliberately excluded so dates such as "September 25," are not values.
VALUE_TOKEN_RE = re.compile(r"^(\$|%|—|–|-|\*|\(?\$?-?\d(?:[\d,]*\d)?(\.\d+)?\)?(%|ppts?|pts?|bps)?)$")
# Unit words that may follow a value in a row ("Marketing-related $ 11,619 24 years").
UNIT_WORD_RE = re.compile(r"^(years?|months?|days?)$", re.I)
# Lines carrying figures; the header walk-back must never absorb these.
DATA_RE = re.compile(r"\$\s*\(?\d|\d,\d{3}|\(\d[\d,]{2,}\)")
UNIT_LINE_RE = re.compile(r"^\((in|dollars|amounts|shares)\b", re.I)
FOOTNOTE_RE = re.compile(r"^(\*|\(\d{1,2}\)|\([a-z]\)|Refer to accompanying notes)", re.I)
SENTENCE_END_RE = re.compile(r"[.?!][\"”’)]?$")
CORP_NAME_RE = re.compile(r"\b(Inc|Corp|Corporation|Ltd)\.$")
# IBM balance-sheet rows: a parenthetical "(net of allowances of $99 in 2025 ...)",
# optionally a Notes-column letter, then two or more figures. Matched on the whole
# line: "... in 2023) 6,804 7,214" - the year before ")" also parses as a value.
# The parenthetical must contain words, so a negative figure "(1,670)" never counts
# (two Microsoft rows matched that way - measured).
LONG_ROW_RE = re.compile(r"\([^()]*[a-z][^()]*\)(?: [A-Z](?:&[A-Z])*)?(?: (?:\$ )?\(?\d[\d,]*\)?){2,}$")

# Qualifiers marking figures that are NOT the reported GAAP result.
# PANEL: Why tag these? Microsoft FY24 p108 shows "Net income 88,308" - an
# unaudited pro forma figure as if Activision closed on July 1, 2022 - while
# reported net income is 88,136. Without the tag both look equally valid.
QUALIFIERS = {
    "pro_forma": re.compile(r"\bpro\s+forma\b", re.I),
    "unaudited": re.compile(r"\bunaudited\b", re.I),
    "non_gaap": re.compile(r"\bnon-GAAP\b", re.I),
}


@dataclass
class Line:
    text: str
    page: int
    printed_page: str | None
    section: str
    in_exhibits: bool


@dataclass
class Unit:
    kind: str            # "prose" | "table"
    lines: list[Line]
    title: str = ""      # tables only: header lines joined
    n_header: int = 0    # tables only: number of leading title/intro lines
    table_no: int = 0    # tables only: index of the table within the document
    years: list[int] = field(default_factory=list)  # tables only: column years
    part: int = 1        # split tables: 1-based part number
    n_parts: int = 1
    repeated: int = 0    # leading lines copied from part 1 (not new source text)
    context: str = ""    # generated label for split parts; not report text

    @property
    def text(self) -> str:
        body = "\n".join(l.text for l in self.lines)
        return f"{self.context}\n{body}" if self.context else body

    def tokens(self) -> int:
        return count_tokens(self.text)


_counter = None
tokenizer_used = ""


def count_tokens(text: str) -> int:
    """Content tokens (no [CLS]/[SEP]) under TOKENIZER_NAME.

    WordPiece pre-tokenizes on whitespace and punctuation, so the count of
    lines joined by newlines equals the sum of the per-line counts.
    """
    global _counter, tokenizer_used
    if _counter is None:
        try:
            from tokenizers import Tokenizer
            tok = Tokenizer.from_pretrained(TOKENIZER_NAME)
            _counter = lambda s: len(tok.encode(s, add_special_tokens=False).ids)
            tokenizer_used = TOKENIZER_NAME
        except Exception as e:
            # FALLBACK: offline or package missing. 1 token per 2 chars
            # over-counts WordPiece on these reports (worst measured chunk:
            # ~2.9 chars/token), so limits stay safe; chunks get smaller.
            log.warning("Tokenizer %r unavailable (%s); using conservative chars/2 estimate",
                        TOKENIZER_NAME, e)
            _counter = lambda s: -(-len(s) // 2)
            tokenizer_used = "fallback:chars/2"
    return _counter(text)


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    company: str
    fiscal_year: int
    source_file: str
    page_start: int
    page_end: int
    printed_pages: list[str]
    section: str
    in_exhibits: bool
    has_table: bool
    tables: list[dict]
    qualifiers: list[str]
    oversized: bool
    char_count: int
    token_count: int     # content tokens under TOKENIZER_NAME, excl. [CLS]/[SEP]
    # text[:overlap_chars] is repeated from chunk `overlap_from` (incl. the
    # separating newline); text[overlap_chars:] is this chunk's own text.
    overlap_chars: int
    overlap_from: str | None
    text: str


@dataclass
class DocStats:
    doc_id: str
    pages: int = 0
    pages_after_cleaning: int = 0
    exhibit_pages: int = 0
    exhibit13_pages: int = 0       # after the signatures page but incorporated by reference (not exhibits)
    signatures_page: int | None = None
    removed: Counter = field(default_factory=Counter)
    tables: int = 0
    split_tables: int = 0
    chunks: int = 0
    chunks_with_tables: int = 0
    oversized_chunks: int = 0
    max_chunk_tokens: int = 0


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------

def signatures_page(pages: list[dict]) -> int | None:
    """Page number of the signatures page (pages sorted by number).

    The page with a "Signatures" heading AND the "Pursuant to the requirements
    of Section 13 or 15(d)" sentence; failing that, the first page with an
    upper-case "SIGNATURES" heading line (the original rule).
    """
    def has_heading(p, rx):
        return any(rx.match(l.strip()) for l in p["text"].splitlines())
    signed = next((p["page_number"] for p in pages
                   if has_heading(p, SIGNATURES_ANY_RE) and SIGNED_RE.search(p["text"])), None)
    if signed is not None:
        return signed
    return next((p["page_number"] for p in pages if has_heading(p, SIGNATURES_RE)), None)


def exhibit13_pages(pages: list[dict], sig_page: int | None) -> set[int]:
    """Pages of an Exhibit 13 (annual report) filed after the signatures page:
    from the page whose top names "Exhibit 13" up to the next "EXHIBIT <n>" page."""
    out: set[int] = set()
    if sig_page is None:
        return out
    inside = False
    for p in pages:
        if p["page_number"] <= sig_page:
            continue
        top = [s for l in p["text"].splitlines() if (s := l.strip()) and not TOC_LINK_RE.match(s)][:2]
        if inside and top and EXHIBIT_START_RE.match(top[0]):
            break
        if not inside and any(EX13_RE.search(l) for l in top):
            inside = True
        if inside:
            out.add(p["page_number"])
    return out


def split_long_line(text: str) -> list[str]:
    """Cut a line too long for one chunk into pieces of <= TARGET_TOKENS.

    PANEL: Packing only cuts between lines, so a single line over the limit
    became an oversized chunk. Some IBM exhibit PDFs extract a whole page as
    one line (up to ~3,000 tokens). Cuts fall after a sentence end where
    possible, else between words; joining the pieces with spaces gives back
    the line (whitespace normalised).
    """
    if count_tokens(text) <= MAX_TOKENS - OVERLAP_MAX_TOKENS:
        return [text]
    pieces, cur, size, last_sentence = [], [], 0, 0
    for w in text.split():
        t = count_tokens(w)
        if cur and size + t > TARGET_TOKENS:
            cut = last_sentence or len(cur)
            pieces.append(" ".join(cur[:cut]))
            cur = cur[cut:]
            size = sum(count_tokens(x) for x in cur)
            last_sentence = 0
        cur.append(w)
        size += t
        if SENTENCE_END_RE.search(w):
            last_sentence = len(cur)
    if cur:
        pieces.append(" ".join(cur))
    return pieces


def clean_doc(pages: list[dict], stats: DocStats) -> list[Line]:
    """Turn one document's pages into a single stream of cleaned lines."""
    pages = sorted(pages, key=lambda p: p["page_number"])
    stats.pages = len(pages)
    sig_page = signatures_page(pages)
    stats.signatures_page = sig_page
    ex13 = exhibit13_pages(pages, sig_page)
    stats.exhibit13_pages = len(ex13)

    section = before_ex13 = "front_matter"
    out: list[Line] = []
    for p in pages:
        lines = [l.strip() for l in p["text"].splitlines()]
        lines = [l for l in lines if l]
        kept = []
        for l in lines:
            if WATERMARK_RE.match(l):
                stats.removed["watermark"] += 1
            else:
                kept.append(l)
        lines = kept
        if lines and TOC_LINK_RE.match(lines[0]):
            lines.pop(0); stats.removed["toc_link"] += 1

        printed = None
        if lines and (m := FOOTER_RE.match(lines[-1])):
            printed = m.group("page"); lines.pop(); stats.removed["footer"] += 1
        elif lines and PAGE_NUM_RE.match(lines[-1]):
            printed = lines.pop(); stats.removed["page_number"] += 1
        # HTML-printed PDFs sometimes push the previous page's number to the top.
        if lines and PAGE_NUM_RE.match(lines[0]):
            lines.pop(0); stats.removed["page_number"] += 1

        # Running headers ("PART II" / "Item 8") give the section directly.
        while lines and (PART_RE.match(lines[0]) or RUNNING_ITEM_RE.match(lines[0])):
            if m := RUNNING_ITEM_RE.match(lines[0]):
                section = f"Item {m.group(1).upper()}"
            lines.pop(0); stats.removed["running_header"] += 1

        in_ex13 = p["page_number"] in ex13
        if not in_ex13 and section.startswith("Exhibit 13"):
            section = before_ex13          # later exhibits are not part of the annual report
        if in_ex13:
            # The running header names the part of the annual report ("Consolidated
            # Balance Sheet", "Management Discussion"); it becomes the section.
            if not section.startswith("Exhibit 13"):
                before_ex13, section = section, "Exhibit 13"
            if len(lines) > 1 and SUBSIDIARIES_RE.match(lines[1]) and (m := AR_HEADER_RE.match(lines[0])):
                section = f"Exhibit 13 {m.group('title')}"
                printed = printed or m.group("pre") or m.group("post")
                if m.group("title").startswith("Consolidated"):
                    # On statement pages the header is the statement's only title
                    # ("Consolidated Income Statement"): keep it, minus the page
                    # number, so the table is titled. Drop the company line.
                    lines[0] = m.group("title")
                    del lines[1]; stats.removed["annual_report_header"] += 1
                else:
                    del lines[:2]; stats.removed["annual_report_header"] += 2

        in_exhibits = sig_page is not None and p["page_number"] > sig_page and not in_ex13
        if in_exhibits:
            stats.exhibit_pages += 1
        if lines:
            stats.pages_after_cleaning += 1
        for l in lines:
            if not in_ex13 and (m := ITEM_HEADING_RE.match(l)):
                section = f"Item {m.group(1).upper()}"
            pieces = split_long_line(l)
            if len(pieces) > 1:
                stats.removed["long_line_split"] += 1
            for piece in pieces:
                out.append(Line(piece, p["page_number"], printed, section, in_exhibits))
    return out


# ---------------------------------------------------------------------------
# Table detection
# ---------------------------------------------------------------------------

def numeric_tail(text: str) -> tuple[str, int]:
    """Split a line into (label, number of trailing numeric values)."""
    toks = text.split()
    i, n = len(toks), 0
    while i > 0:
        t = toks[i - 1]
        if UNIT_WORD_RE.match(t) and i > 1 and toks[i - 2][-1:].isdigit():
            i -= 1
            continue
        if not VALUE_TOKEN_RE.match(t):
            break
        if any(c.isdigit() for c in t):
            n += 1
        i -= 1
    return " ".join(toks[:i]), n


def is_row(text: str) -> bool:
    label, n = numeric_tail(text)
    # Long labels are wrapped prose lines that happen to end in a number
    # (e.g. "... as of September 30, 2023").
    return n >= 1 and len(label) <= 80 and not SENTENCE_END_RE.search(label)


def ends_sentence(text: str) -> bool:
    return bool(SENTENCE_END_RE.search(text)) and not CORP_NAME_RE.search(text)


def is_bridge(text: str) -> bool:
    """Label-only line - or a row whose label is too long for is_row - that can
    sit between table rows.

    PANEL: IBM's balance sheet has rows such as "Notes and accounts receivable-
    trade (net of allowances of $99 in 2025 and $114 in 2024) C 8,112 6,804";
    without this the table broke in two and its second half (total assets,
    liabilities, equity) lost its title and year columns.
    """
    if ends_sentence(text) or FOOTNOTE_RE.match(text):
        return False
    if len(text) <= SHORT_LINE:
        return True
    # Only long labels ending in a parenthetical (plus optional Notes letter) before
    # their figures: a looser rule merged 74 Microsoft chunks' tables (measured).
    return bool(LONG_ROW_RE.search(text))


def find_tables(lines: list[Line]) -> list[tuple[int, int, int]]:
    """Return (header_start, row_start, end_exclusive) for each table."""
    texts = [l.text for l in lines]
    rows = [is_row(t) for t in texts]
    tables = []
    i, prev_end = 0, 0
    n = len(texts)
    while i < n:
        if not rows[i]:
            i += 1
            continue
        start, last, count, k = i, i, 1, i + 1
        while k < n:
            if lines[k].in_exhibits != lines[start].in_exhibits:
                break
            if rows[k]:
                last, count, k = k, count + 1, k + 1
                continue
            b = k
            while b < n and b - k < MAX_BRIDGE_LINES and is_bridge(texts[b]) and not rows[b]:
                b += 1
            if b > k and b < n and rows[b]:
                k = b
                continue
            break
        if count < MIN_TABLE_ROWS:
            i += 1
            continue

        end = last + 1
        # Footnotes directly under the table ("* Not meaningful.", "(1) ...").
        f = 0
        while end < n and f < MAX_FOOTNOTE_LINES and FOOTNOTE_RE.match(texts[end]):
            end += 1; f += 1
            while end < n and f < MAX_FOOTNOTE_LINES and not ends_sentence(texts[end - 1]) \
                    and not rows[end] and not FOOTNOTE_RE.match(texts[end]):
                end += 1; f += 1

        # The title search never crosses into a previous section or table.
        lo = prev_end
        for j in range(start - 1, prev_end - 1, -1):
            if lines[j].section != lines[start].section or lines[j].in_exhibits != lines[start].in_exhibits:
                lo = j + 1
                break
        tables.append((walk_header(texts, rows, start, lo), start, end))
        prev_end = i = end
    return tables


def walk_header(texts: list[str], rows: list[bool], start: int, lo: int) -> int:
    """Walk back from the first table row to include title, units and intro."""
    h, budget = start, MAX_HEADER_LINES
    while h > lo and budget:
        t = texts[h - 1]
        if rows[h - 1] or DATA_RE.search(t):
            break
        if t.endswith(":"):
            # An intro sentence ("The following table ... (dollars in millions):"
            # or "... on an unaudited pro forma basis ...:"). Pull in the whole
            # paragraph - it often carries the qualifier that explains the figures -
            # then stop: anything above it belongs to the preceding content.
            h -= 1; budget -= 1
            while h > lo and budget and not ends_sentence(texts[h - 1]) and not rows[h - 1] \
                    and not DATA_RE.search(texts[h - 1]):
                h -= 1; budget -= 1
            break
        if UNIT_LINE_RE.match(t) or (len(t) <= SHORT_LINE and not ends_sentence(t)):
            h -= 1; budget -= 1
            continue
        break
    return h


# ---------------------------------------------------------------------------
# Units and chunk packing
# ---------------------------------------------------------------------------

def to_units(lines: list[Line]) -> list[Unit]:
    units: list[Unit] = []
    pos = 0

    def emit(buf: list[Line]) -> None:
        # Text with no sentence-ending lines (exhibit index lists, legal
        # clauses) or a very long paragraph is cut at line boundaries into
        # pieces of roughly equal size, so no tiny tail fragment is left over.
        sizes = [count_tokens(l.text) for l in buf]
        total = sum(sizes)
        if total <= TARGET_TOKENS:
            units.append(Unit("prose", buf))
            return
        goal = total / -(-total // TARGET_TOKENS)
        part: list[Line] = []
        size = 0
        for l, t in zip(buf, sizes):
            if part and size + t > goal:
                units.append(Unit("prose", part)); part = []; size = 0
            part.append(l)
            size += t
        if part:
            units.append(Unit("prose", part))

    def add_prose(a: int, b: int) -> None:
        buf: list[Line] = []
        for l in lines[a:b]:
            if buf and (l.section != buf[-1].section or l.in_exhibits != buf[-1].in_exhibits):
                emit(buf); buf = []
            buf.append(l)
            if ends_sentence(l.text):    # paragraph / sentence boundary
                emit(buf); buf = []
        if buf:
            emit(buf)

    for n, (h, s, e) in enumerate(find_tables(lines)):
        add_prose(pos, h)
        title = " / ".join(l.text for l in lines[h:s])
        table = Unit("table", lines[h:e], title=title, n_header=s - h, table_no=n)
        table.years = column_years(table)
        units.extend(split_table(table))
        pos = e
    add_prose(pos, len(lines))
    return units


def qualifiers_of(text: str) -> list[str]:
    return sorted(k for k, rx in QUALIFIERS.items() if rx.search(text))


YEAR_RE = re.compile(r"^(19|20)\d{2}$")
# "June 30," / "September 28," prefixes in "Year Ended June 30, 2024 2023";
# IBM writes "For the year ended December 31: Notes 2025 2024 2023".
MONTH_DAY_RE = re.compile(r"\b[A-Z][a-z]+ \d{1,2}[,:]")


def year_header(u: Unit) -> tuple[int, list[int]] | None:
    """(line index, years) of the table's year-header row, if found."""
    # Year headers appear as "2024 2023 2022", "Year Ended June 30, 2024 2023"
    # or "(In millions, ...) 2024 2023 Change"; they may sit in the title lines.
    for i, l in enumerate(u.lines[:u.n_header + 4]):
        if "$" in l.text:
            continue
        nums = [t for t in MONTH_DAY_RE.sub("", l.text).split() if any(c.isdigit() for c in t)]
        if nums and all(YEAR_RE.match(t) for t in nums):
            return i, [int(t) for t in nums]
    return None


def column_years(u: Unit) -> list[int]:
    """Fiscal years labelling the table's columns, from its year-header row.

    PANEL: a chunk's fiscal_year is the REPORT's year, but tables also carry
    prior-year columns - Microsoft's FY2025 report repeats the FY2024/FY2023
    pro forma table. Answer generation must use the column year, not the
    report year.
    """
    subs = sub_tables(u)
    if subs:
        return [y for _, y in subs]
    found = year_header(u)
    return found[1] if found else []


# A line that is only a year (or "June 30, 2024") heading a sub-table.
SUB_YEAR_RE = re.compile(r"^(?:[A-Z][a-z]+ \d{1,2}, )?((?:19|20)\d{2})$")


def sub_tables(u: Unit) -> list[tuple[int, int]]:
    """(line index, year) of each sub-table heading in a stacked table.

    Some tables stack one block per year ("2025" ... "2024" ... "2023") under
    a single title. Only tables with two or more such headings count.
    """
    subs = [(i, int(m.group(1))) for i, l in enumerate(u.lines[u.n_header:], start=u.n_header)
            if (m := SUB_YEAR_RE.match(l.text))]
    return subs if len(subs) > 1 else []


def split_table(u: Unit) -> list[Unit]:
    """Split a table longer than MAX_TOKENS into self-describing parts.

    Every part repeats the table's header (title, intro, unit line and the
    column-header lines up to the year row) and gets a generated label such
    as "[Table part 2/3 | FY2024, FY2023, FY2022 | CONSOLIDATED STATEMENTS OF
    CASH FLOWS]". Splits happen only after a complete row, preferring
    sub-section labels such as "Investing activities:".

    Stacked tables (one block per year) are split between blocks where
    possible; a part that starts inside a block repeats that block's own year
    heading and column headers, and is labelled with the years it contains.
    PANEL: repeating the first block's year instead would mislabel FY2023
    segment figures as FY2025 - the exact error this tagging exists to stop.
    """
    if u.tokens() <= MAX_TOKENS:
        return [u]
    subs = sub_tables(u)
    if subs:
        k = u.n_header
    else:
        found = year_header(u)
        k = max(u.n_header, found[0] + 1) if found else u.n_header
    header, body = u.lines[:k], u.lines[k:]
    sub_at = {i - k: y for i, y in subs}           # body index -> sub-table year
    # Each sub-table's heading block: the year line plus column headers up to the first figures.
    blocks: dict[int, list[Line]] = {}
    for j in sub_at:
        end = j + 1
        while end < len(body) and not DATA_RE.search(body[end].text) and end not in sub_at:
            end += 1
        blocks[j] = body[j:end]
    block_reserve = max((sum(count_tokens(l.text) for l in b) for b in blocks.values()), default=0)

    all_years = ", ".join(f"FY{y}" for y in u.years) or "years not detected"
    name = next((l.text for l in header if l.text.isupper()), u.title)[:120]
    label_tokens = count_tokens(f"[Table part 99/99 | {all_years} | {name}]")
    budget = MAX_TOKENS - label_tokens - block_reserve - sum(count_tokens(l.text) for l in header)
    if budget < 150:
        # A very long intro paragraph: repeat only the lines nearest the rows.
        header = header[-3:]
        budget = MAX_TOKENS - label_tokens - block_reserve - sum(count_tokens(l.text) for l in header)

    sizes = [count_tokens(l.text) for l in body]
    groups: list[list[int]] = []                   # body indices per part
    cur: list[int] = []
    last_break: int | None = None                  # position in cur where a break is allowed
    for i, l in enumerate(body):
        starts_sub = i in sub_at
        allowed = bool(cur) and (starts_sub or is_row(body[i - 1].text)) and not FOOTNOTE_RE.match(l.text)
        size = sum(sizes[j] for j in cur)
        if allowed:
            prefer = (starts_sub or l.text.endswith(":")) and size > budget / 2
            if prefer or size + sizes[i] > budget:
                groups.append(cur); cur = []; last_break = None
            else:
                last_break = len(cur)
        elif cur and size + sizes[i] > budget:
            # Cannot break here (e.g. inside a wrapped row label): go back to the last allowed break.
            cut = last_break or len(cur)
            groups.append(cur[:cut]); cur = cur[cut:]; last_break = None
        cur.append(i)
    if cur:
        groups.append(cur)
    if len(groups) == 1:
        return [u]

    parts = []
    for n, g in enumerate(groups, start=1):
        lead: list[Line] = []
        part_years = [sub_at[j] for j in g if j in sub_at]
        if sub_at and g[0] not in sub_at:
            active = max((j for j in sub_at if j < g[0]), default=None)
            if active is not None:
                lead = blocks[active]
                part_years.insert(0, sub_at[active])
        years = ", ".join(f"FY{y}" for y in part_years) if sub_at else all_years
        parts.append(Unit(
            "table", (header if n == 1 else header + lead) + [body[j] for j in g],
            title=u.title, n_header=u.n_header, table_no=u.table_no, years=u.years,
            part=n, n_parts=len(groups), repeated=0 if n == 1 else len(header) + len(lead),
            context=f"[Table part {n}/{len(groups)} | {years} | {name}]",
        ))
    return parts


def same_scope(a: Unit, b: Unit) -> bool:
    return a.lines[-1].section == b.lines[0].section and a.lines[-1].in_exhibits == b.lines[0].in_exhibits


def group_units(units: list[Unit]) -> list[list[Unit]]:
    """Pack units into chunk-sized groups, leaving room for overlap.

    Groups are kept <= MAX_TOKENS - OVERLAP_MAX_TOKENS so that prefixing the
    overlap never pushes a chunk past MAX_TOKENS. Split-table parts (sized
    against MAX_TOKENS) get no overlap, so they are not affected.
    """
    cap = MAX_TOKENS - OVERLAP_MAX_TOKENS
    groups: list[list[Unit]] = []
    cur: list[Unit] = []

    def size(us: list[Unit]) -> int:
        return sum(u.tokens() for u in us)

    def close() -> None:
        if cur:
            groups.append(cur.copy())
            cur.clear()

    for u in units:
        if cur and not same_scope(cur[-1], u):
            close()
        # Each part of a split table is its own chunk.
        if u.n_parts > 1:
            close()
            groups.append([u])
            continue
        limit = cap if u.kind == "table" or any(c.kind == "table" for c in cur) else TARGET_TOKENS
        if cur and size(cur) + u.tokens() > limit:
            carry = []
            if cur[-1].kind == "prose" and cur[-1].tokens() <= HEADING_TOKENS \
                    and cur[-1].tokens() + u.tokens() <= cap:
                carry = [cur.pop()]
            close()
            cur.extend(carry)
        cur.append(u)
    close()

    # Merge leftover fragments (sentence tails, "** Furnished, not filed.")
    # into a same-section neighbour, preferring the preceding chunk.
    i = 0
    while i < len(groups):
        g = groups[i]
        if size(g) < MIN_TOKENS and g[0].n_parts == 1:
            prev = groups[i - 1] if i > 0 else None
            nxt = groups[i + 1] if i + 1 < len(groups) else None
            if prev and prev[0].n_parts == 1 and same_scope(prev[-1], g[0]) \
                    and size(prev) + size(g) <= cap:
                prev.extend(g); del groups[i]
                continue
            if nxt and nxt[0].n_parts == 1 and same_scope(g[-1], nxt[0]) \
                    and size(g) + size(nxt) <= cap:
                nxt[0:0] = g; del groups[i]
                continue
        i += 1
    return groups


def tail_overlap(text: str) -> str:
    """Last OVERLAP_MIN..MAX tokens of `text`, cut at a word boundary.

    Prefers the longest tail that starts at a sentence start; otherwise the
    longest tail within OVERLAP_MAX_TOKENS. Returns "" if the text is too short.
    """
    words = list(re.finditer(r"\S+", text))
    best, best_sentence, total = None, None, 0
    for i in range(len(words) - 1, -1, -1):
        total += count_tokens(words[i].group())
        if total > OVERLAP_MAX_TOKENS:
            break
        if total >= OVERLAP_MIN_TOKENS:
            best = words[i].start()
            if i == 0 or SENTENCE_END_RE.search(words[i - 1].group()):
                best_sentence = words[i].start()
    start = best_sentence if best_sentence is not None else best
    return "" if start is None else text[start:]


def pack_chunks(units: list[Unit], doc: dict, stats: DocStats) -> list[Chunk]:
    chunks: list[Chunk] = []
    prev: list[Unit] | None = None

    for cur in group_units(units):
        # Page range from source lines only, not from headers repeated into a part.
        ls = [l for u in cur for l in u.lines[u.repeated:]]
        body = "\n".join(u.text for u in cur)
        # Overlap only between prose in the same section: a table chunk already
        # starts with its own title, and table rows make a poor lead-in.
        overlap = ""
        if prev and cur[0].kind == "prose" and prev[-1].kind == "prose" and same_scope(prev[-1], cur[0]):
            overlap = tail_overlap("\n".join(u.text for u in prev))
        text = f"{overlap}\n{body}" if overlap else body
        if overlap and count_tokens(text) > MAX_TOKENS:     # safety net; packing reserves room
            overlap, text = "", body
        prev = cur
        tables = [{
            "table_id": f"{doc['doc_id']}_t{u.table_no:03d}",
            "part": u.part,
            "n_parts": u.n_parts,
            "title": u.title,
            "page_start": u.lines[u.repeated].page,
            "page_end": u.lines[-1].page,
            "n_lines": len(u.lines),
            "column_years": u.years,
            "qualifiers": qualifiers_of(u.text),
        } for u in cur if u.kind == "table"]
        token_count = count_tokens(text)
        if token_count > MAX_TOKENS:
            # Not expected (only a single line longer than MAX_TOKENS could do
            # this); surfaced rather than silently truncated later.
            log.error("%s chunk %d has %d tokens > %d", doc["doc_id"], len(chunks), token_count, MAX_TOKENS)
        chunks.append(Chunk(
            chunk_id=f"{doc['doc_id']}_c{len(chunks):04d}",
            doc_id=doc["doc_id"], company=doc["company"], fiscal_year=doc["fiscal_year"],
            source_file=doc["source_file"],
            page_start=ls[0].page, page_end=ls[-1].page,
            printed_pages=sorted({l.printed_page for l in ls if l.printed_page}, key=int),
            section=ls[0].section, in_exhibits=ls[0].in_exhibits,
            has_table=bool(tables), tables=tables, qualifiers=qualifiers_of(text),
            oversized=token_count > MAX_TOKENS, char_count=len(text),
            token_count=token_count,
            overlap_chars=len(overlap) + 1 if overlap else 0,
            overlap_from=chunks[-1].chunk_id if overlap else None,
            text=text,
        ))

    stats.max_chunk_tokens = max((c.token_count for c in chunks), default=0)
    stats.tables = sum(u.kind == "table" and u.part == 1 for u in units)
    stats.split_tables = sum(u.kind == "table" and u.part == 1 and u.n_parts > 1 for u in units)
    stats.chunks = len(chunks)
    stats.chunks_with_tables = sum(c.has_table for c in chunks)
    stats.oversized_chunks = sum(c.oversized for c in chunks)
    return chunks


def preprocess_doc(pages: list[dict]) -> tuple[list[Chunk], DocStats]:
    doc = {k: pages[0][k] for k in ("doc_id", "company", "fiscal_year", "source_file")}
    stats = DocStats(doc_id=doc["doc_id"])
    lines = clean_doc(pages, stats)
    return pack_chunks(to_units(lines), doc, stats), stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def run(in_dir: Path, out_dir: Path, companies: set[str] | None = None) -> dict:
    pages_path = in_dir / "pages.jsonl"
    if not pages_path.exists():
        raise FileNotFoundError(f"{pages_path} not found - run fin_graph_rag.pdf_extractor first")
    by_doc: dict[str, list[dict]] = defaultdict(list)
    with pages_path.open(encoding="utf-8") as f:
        for line in f:
            p = json.loads(line)
            by_doc[p["doc_id"]].append(p)

    out_dir.mkdir(parents=True, exist_ok=True)
    chunks_path, report_path = out_dir / "chunks.jsonl", out_dir / "chunking_report.json"
    docs, flagged, keep = [], [], []
    if companies:
        # Merge mode: other companies' chunks are copied byte for byte (same
        # order), so the embeddings already built for them stay valid.
        from fin_graph_rag.pdf_extractor import kept_lines
        keep = kept_lines(chunks_path, companies)
        if report_path.exists():
            old = json.loads(report_path.read_text(encoding="utf-8"))
            docs = [d for d in old["documents"] if d["doc_id"].rsplit("_", 1)[0] not in companies]
            flagged = [t for t in old["qualifier_flagged_tables"]
                       if t["chunk_id"].rsplit("_", 2)[0] not in companies]
        by_doc = {d: p for d, p in by_doc.items() if p[0]["company"] in companies}
    tmp_path = chunks_path.with_suffix(".jsonl.tmp")
    with tmp_path.open("w", encoding="utf-8") as out:
        out.writelines(keep)
        for doc_id in sorted(by_doc):
            chunks, stats = preprocess_doc(by_doc[doc_id])
            for c in chunks:
                out.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")
                for t in c.tables:
                    if t["qualifiers"]:
                        flagged.append({"chunk_id": c.chunk_id, "page_start": t["page_start"],
                                        "qualifiers": t["qualifiers"],
                                        "column_years": t["column_years"], "title": t["title"][:200]})
            d = asdict(stats)
            d["removed"] = dict(stats.removed)
            docs.append(d)
            log.info("%s: %d pages -> %d chunks (%d with tables, %d oversized, max %d tokens), %d tables",
                     doc_id, stats.pages, stats.chunks, stats.chunks_with_tables,
                     stats.oversized_chunks, stats.max_chunk_tokens, stats.tables)
    tmp_path.replace(chunks_path)

    report = {
        "run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config": {"tokenizer": tokenizer_used, "MAX_TOKENS": MAX_TOKENS, "TARGET_TOKENS": TARGET_TOKENS,
                   "MIN_TOKENS": MIN_TOKENS, "HEADING_TOKENS": HEADING_TOKENS,
                   "OVERLAP_MIN_TOKENS": OVERLAP_MIN_TOKENS, "OVERLAP_MAX_TOKENS": OVERLAP_MAX_TOKENS,
                   "MIN_TABLE_ROWS": MIN_TABLE_ROWS, "SHORT_LINE": SHORT_LINE,
                   "MAX_BRIDGE_LINES": MAX_BRIDGE_LINES, "MAX_HEADER_LINES": MAX_HEADER_LINES},
        "updated_companies": sorted(companies) if companies else None,
        "documents": docs,
        "qualifier_flagged_tables": flagged,
        "totals": {k: sum(d.get(k, 0) for d in docs) for k in
                   ("pages", "pages_after_cleaning", "exhibit_pages", "exhibit13_pages", "tables", "split_tables",
                    "chunks", "chunks_with_tables", "oversized_chunks")},
        "max_chunk_tokens": max((d["max_chunk_tokens"] for d in docs), default=0),
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Clean extracted pages and build table-preserving chunks.")
    ap.add_argument("--in-dir", type=Path, default=Path(os.getenv("FGR_EXTRACTED_DIR", "data/extracted")))
    ap.add_argument("--out-dir", type=Path, default=Path(os.getenv("FGR_PROCESSED_DIR", "data/processed")))
    ap.add_argument("--companies", nargs="+", metavar="COMPANY",
                    help="only (re-)chunk these companies (e.g. ibm nvidia); other companies' chunks are kept")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    try:
        report = run(args.in_dir, args.out_dir, {c.lower() for c in args.companies} if args.companies else None)
    except FileNotFoundError as e:
        log.error("%s", e)
        return 1
    print(json.dumps(report["totals"], indent=2))
    print(f"Tokenizer: {report['config']['tokenizer']}; max chunk tokens: {report['max_chunk_tokens']} (limit {MAX_TOKENS})")
    print(f"Qualifier-flagged tables: {len(report['qualifier_flagged_tables'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
