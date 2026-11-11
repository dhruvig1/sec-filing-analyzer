# src/text_utils.py
"""Turn raw EDGAR full-submission.txt files into clean text and find the Item 1A / Item 7 sections.

A full-submission.txt bundles every document in the filing (the 10-K itself as
HTML/inline XBRL, exhibits, XBRL schemas, uuencoded images). Feeding that raw
markup to the embedder or the LLM wastes most of the context on tags, so we keep
the main 10-K document (plus any EX-13 annual report, see below) and strip the HTML.

Filers lay sections out in several ways, so extraction tries two strategies:

1. "Item" headings: "Item 1A. Risk Factors" ... "Item 1B. Unresolved Staff Comments".
2. Plain title headings on their own line ("Risk Factors", "Management's Discussion
   and Analysis ..."). Used when strategy 1 finds nothing or only a short pointer
   such as "Information in response to this Item 1A can be found in ...". This is
   how large banks (C, MS, JPM), integrated-oil filers (XOM, CVX) and MCD lay out
   their 10-Ks.
3. Follow a table of contents to the first listed sub-heading when the section
   body never repeats its own title (CVX).
"""
import html
import re

_MAIN_DOC = re.compile(r"<DOCUMENT>\s*<TYPE>10-K\b.*?</DOCUMENT>", re.S | re.I)
# Some filers (WFC, PGR) put MD&A and risk factors in the annual report exhibit and
# only reference it from the 10-K, so EX-13 has to be kept too.
_EX13_DOC = re.compile(r"<DOCUMENT>\s*<TYPE>EX-13\b.*?</DOCUMENT>", re.S | re.I)
_DROP_BLOCKS = re.compile(r"<(script|style|ix:header|head)\b[^>]*>.*?</\1\s*>", re.S | re.I)
_BLOCK_BREAKS = re.compile(r"<br\s*/?>|</(p|div|tr|li|h[1-6]|table)\s*>", re.I)
# opening block tags and table cells become a space so neighbouring words don't merge
_SOFT_BREAKS = re.compile(r"<(p|div|tr|li|h[1-6]|table)\b[^>]*>|</?(td|th)\b[^>]*>", re.I)
_TAGS = re.compile(r"<[^>]+>")
# Lines that are only a "Table of Contents" back-link or a page number (repeated on every page).
_PAGE_NOISE = re.compile(r"^(?:(?:financial )?table of contents|page \d{1,3}|\d{1,3})$\n?", re.I | re.M)

_SEP = r"\s*[.:\-–—|]?\s*"   # "Item 1A." / "ITEM 7 |" / "Item 1A —"

# ── strategy 1: Item headings ──────────────────────────────────
_ITEM_1A = re.compile(rf"item\s*1a{_SEP}risk\s+factors", re.I)
_SECTION_END = re.compile(
    rf"item\s*1b{_SEP}unresolved|item\s*1c{_SEP}cybersecurity|item\s*2{_SEP}properties", re.I)
_ITEM_7 = re.compile(rf"item\s*7{_SEP}management.{{0,2}}s\s+discussion", re.I)
_ITEM_7_END = re.compile(rf"item\s*7a{_SEP}quantitative|item\s*8{_SEP}financial", re.I)

# ── strategy 2: plain title headings (whole line) ──────────────
_LEAD = r"^[^\w\n]*(?:item\s*\d{1,2}[a-c]?\W*)?"   # optional bullet / "Item 7." before the title
_TAIL = r"[^\w\n]*$"
_RISK_TITLE = re.compile(_LEAD + r"risk\s+factors" + _TAIL, re.I | re.M)
_RISK_TITLE_END = re.compile(_LEAD + r"(?:unresolved\s+staff\s+comments|cybersecurity|properties"
                             r"|legal\s+proceedings|sustainability|human\s+capital(?:\s+resources)?"
                             r"|information\s+about\s+our\s+executive\s+officers|executive\s+officers"
                             r"|management.{0,2}s\s+discussion\s+and\s+analysis.*"
                             r"|quantitative\s+and\s+qualitative\s+disclosures.*|controls\s+and\s+procedures"
                             r"|consolidated\s+financial\s+statements|financial\s+statements\s+and\s+supplementary\s+data"
                             r"|report\s+of\s+independent\s+registered\s+public\s+accounting\s+firm)" + _TAIL,
                             re.I | re.M)
_MDNA_TITLE = re.compile(_LEAD + r"(?:management.{0,2}s\s+discussion\s+and\s+analysis"
                         r"(?:\s+of\s+financial\s+condition\s+and\s+results\s+of\s+operations)?"
                         r"|financial\s+review)" + _TAIL,   # WFC's annual report calls its MD&A "Financial Review"
                         re.I | re.M)
_MDNA_TITLE_END = re.compile(_LEAD + r"(?:quantitative\s+and\s+qualitative\s+disclosures?\s+about\s+market\s+risk"
                             r"|financial\s+statements\s+and\s+supplementary\s+data|consolidated\s+financial\s+statements"
                             r"|report\s+of\s+independent\s+registered\s+public\s+accounting\s+firm"
                             r"|management.{0,2}s\s+report\s+on\s+internal\s+control.*|controls\s+and\s+procedures)" + _TAIL,
                             re.I | re.M)


def _html_to_text(doc: str) -> str:
    doc = _DROP_BLOCKS.sub(" ", doc)
    doc = _BLOCK_BREAKS.sub("\n", doc)
    doc = _SOFT_BREAKS.sub(" ", doc)
    # Inline tags (span, font, b, a, ix:*) are removed without adding a space:
    # small-caps headings are coded as R<span>ISK</span> F<span>ACTORS</span>,
    # and replacing the tags with spaces turned them into "R ISK F ACTORS".
    doc = _TAGS.sub("", doc)
    doc = html.unescape(doc).replace("\xa0", " ")
    doc = re.sub(r"[ \t\r\f\v]+", " ", doc)
    doc = re.sub(r" *\n *", "\n", doc)
    doc = _PAGE_NOISE.sub("", doc)  # "Table of Contents" links and bare page numbers
    doc = re.sub(r"\n{3,}", "\n\n", doc)
    return doc.strip()


def clean_filing_text(raw: str) -> str:
    match = _MAIN_DOC.search(raw)
    docs = [match.group(0) if match else raw] + _EX13_DOC.findall(raw)
    return "\n\n".join(_html_to_text(d) for d in docs)


def _is_heading(text: str, pos: int) -> bool:
    """True if the match starts its line, allowing only punctuation or 'PART II' before it.

    Real headings sit on their own line after HTML cleaning; cross-references sit
    mid-sentence ("see Part I, Item 1A. Risk Factors") or in bullets that start
    with words ("– this Part II, Item 7. Management's ...").
    """
    prefix = text[text.rfind("\n", 0, pos) + 1: pos]
    return re.fullmatch(r"[^\w\n]*(?:part\s+[ivx]+[^\w\n]*)?", prefix, re.I) is not None


def _next_heading(end_re, text, pos):
    for m in end_re.finditer(text, pos):
        if _is_heading(text, m.start()):
            return m
    return None


def _looks_like_toc(text, pos, window=1500):
    """True if the text after pos reads like a table of contents (most lines end in a page number)."""
    lines = [l for l in text[pos: pos + window].split("\n")[1:] if l.strip()]
    return len(lines) >= 8 and sum(bool(re.search(r"\d\s*$", l)) for l in lines) / len(lines) >= 0.4


def _best_span(text, starts, end_re, min_len, allow_open_end=False):
    """Pair each start with the next end heading; the earliest start per end wins
    (so a running page header can't cut the section short), then the longest span.

    allow_open_end: if no start has an end heading after it, use the earliest start
    and run to the end of the text. Only safe for whole-line title headings (never
    cross-references); needed when a section is the last thing in a document, e.g.
    PGR's annual report puts the financial statements before the MD&A.
    """
    start_for_end, open_starts = {}, []
    for m in starts:
        end_match = _next_heading(end_re, text, m.end())
        if end_match:  # a mention with no following heading is a cross-reference
            start_for_end.setdefault(end_match.start(), m.start())
        else:
            open_starts.append(m.start())
    spans = [(s, e) for e, s in start_for_end.items() if e - s >= min_len]
    if spans:
        return max(spans, key=lambda se: se[1] - se[0])
    if allow_open_end and open_starts and len(text) - open_starts[0] >= min_len:
        return open_starts[0], len(text)
    return None


def _follow_toc(text, title_matches, end_re, min_len):
    """Strategy 3: the title only appears in a table of contents that lists its sub-sections.

    CVX's "Financial Table of Contents" has "Management's Discussion and Analysis ..."
    followed by its sub-headings, and the MD&A body starts directly at the first
    sub-heading without repeating the title. So: take the first listed sub-heading,
    find it as a heading line after the TOC, and start the section there.
    """
    for m in title_matches:
        toc_end = _next_heading(end_re, text, m.end())
        if not toc_end:
            continue
        lines = [l.strip() for l in text[m.end(): toc_end.start()].split("\n") if l.strip()]
        if len(lines) < 3 or any(len(l) > 80 for l in lines):  # not a list of sub-headings
            continue
        first = re.sub(r"[\s.]*\d{1,3}$", "", lines[0])       # drop a trailing page number
        if len(first) < 4:
            continue
        body = re.compile(r"^[^\w\n]*" + re.escape(first) + r"[^\w\n]*$", re.I | re.M).search(text, toc_end.end())
        if body:
            stop = _next_heading(end_re, text, body.end())
            stop = stop.start() if stop else len(text)
            if stop - body.start() >= min_len:
                return body.start(), stop
    return None


def extract_section(text, start_re, end_re, max_chars=6000, min_len=1500,
                    title_re=None, title_end_re=None):
    """Return the section text, or None.

    Strategy 1 uses Item headings: headings first, then any mention, so the table of
    contents, cross-references and running headers lose to the real section. Spans
    shorter than min_len are treated as TOC entries or "see the annual report"
    pointers. Strategy 2 (plain title lines) runs only when strategy 1 finds nothing.
    """
    matches = list(start_re.finditer(text))
    span = (_best_span(text, [m for m in matches if _is_heading(text, m.start())], end_re, min_len)
            or _best_span(text, matches, end_re, min_len))
    if span is None and title_re is not None:
        all_titles = list(title_re.finditer(text))
        titles = [m for m in all_titles if not _looks_like_toc(text, m.start())]
        span = (_best_span(text, titles, title_end_re, min_len, allow_open_end=True)
                or _follow_toc(text, all_titles, title_end_re, min_len))  # TOC entries are the point here
    if span is None:
        return None
    start, end = span
    return text[start: start + min(end - start, max_chars)]


def extract_risk_section(text: str, max_chars: int = 6000):
    """Item 1A Risk Factors, or None if it can't be located."""
    return extract_section(text, _ITEM_1A, _SECTION_END, max_chars,
                           title_re=_RISK_TITLE, title_end_re=_RISK_TITLE_END)


def extract_mdna_section(text: str, max_chars: int = 15000):
    """Item 7 Management's Discussion and Analysis, or None."""
    return extract_section(text, _ITEM_7, _ITEM_7_END, max_chars,
                           title_re=_MDNA_TITLE, title_end_re=_MDNA_TITLE_END)


# ── debugging: python src/text_utils.py CVX 2025 mdna ──────────
def explain(text, kind="risk"):
    """Print every candidate heading and what it pairs with, to see why a section was or wasn't found."""
    start_re, end_re, title_re, title_end_re = (
        (_ITEM_1A, _SECTION_END, _RISK_TITLE, _RISK_TITLE_END) if kind == "risk"
        else (_ITEM_7, _ITEM_7_END, _MDNA_TITLE, _MDNA_TITLE_END))
    for label, s_re, e_re in (("item", start_re, end_re), ("title", title_re, title_end_re)):
        print(f"-- {label} candidates")
        for m in s_re.finditer(text):
            e = _next_heading(e_re, text, m.end())
            span = (e.start() - m.start()) if e else None
            print(f"  @{m.start():>8} heading={_is_heading(text, m.start())!s:5} toc={_looks_like_toc(text, m.start())!s:5}"
                  f" span={span} | {text[m.start(): m.start() + 80]!r}")
    found = (extract_risk_section if kind == "risk" else extract_mdna_section)(text, max_chars=200)
    print(f"-- result: {found!r}")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, __import__("os").path.dirname(__file__))
    from config import FILINGS_DIR, parse_filing_path

    ticker, year, kind = sys.argv[1].upper(), int(sys.argv[2]), (sys.argv[3] if len(sys.argv) > 3 else "risk")
    for path in (FILINGS_DIR / ticker / "10-K").glob("*/full-submission.txt"):
        if parse_filing_path(path)[2] == year:
            explain(clean_filing_text(open(path, errors="ignore").read()), kind)
