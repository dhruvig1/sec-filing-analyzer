# src/text_utils.py
"""Turn raw EDGAR full-submission.txt files into clean text and find the Item 1A section.

A full-submission.txt bundles every document in the filing (the 10-K itself as
HTML/inline XBRL, exhibits, XBRL schemas, uuencoded images). Feeding that raw
markup to the embedder or the LLM wastes most of the context on tags, so we keep
only the main 10-K document and strip the HTML first.
"""
import html
import re

_MAIN_DOC = re.compile(r"<DOCUMENT>\s*<TYPE>10-K\b.*?</DOCUMENT>", re.S | re.I)
_DROP_BLOCKS = re.compile(r"<(script|style|ix:header|head)\b[^>]*>.*?</\1\s*>", re.S | re.I)
_BLOCK_BREAKS = re.compile(r"<br\s*/?>|</(p|div|tr|li|h[1-6]|table)\s*>", re.I)
_TAGS = re.compile(r"<[^>]+>")

_ITEM_1A = re.compile(r"item\s*1a\s*[.:\-–—]?\s*risk\s+factors", re.I)
_SECTION_END = re.compile(
    r"item\s*1b\s*[.:\-–—]?\s*unresolved|item\s*1c\s*[.:\-–—]?\s*cybersecurity"
    r"|item\s*2\s*[.:\-–—]?\s*properties",
    re.I,
)


def clean_filing_text(raw: str) -> str:
    match = _MAIN_DOC.search(raw)
    doc = match.group(0) if match else raw
    doc = _DROP_BLOCKS.sub(" ", doc)
    doc = _BLOCK_BREAKS.sub("\n", doc)
    doc = _TAGS.sub(" ", doc)
    doc = html.unescape(doc).replace("\xa0", " ")
    doc = re.sub(r"[ \t\r\f\v]+", " ", doc)
    doc = re.sub(r"\n\s*\n+", "\n\n", doc)
    return doc.strip()


_ITEM_7 = re.compile(r"item\s*7\s*[.:\-–—]?\s*management.{0,2}s\s+discussion", re.I)
_ITEM_7_END = re.compile(r"item\s*7a\s*[.:\-–—]?\s*quantitative|item\s*8\s*[.:\-–—]?\s*financial", re.I)


def extract_section(text: str, start_re, end_re, max_chars: int = 6000, min_len: int = 500):
    """Return the longest span that starts at start_re and runs to the next end_re.

    A heading appears several times (table of contents, cross-references, the
    real section). The real section is the one with the most text before the
    next item heading, so we pick that instead of the first or last hit.
    """
    best_start, best_len = None, 0
    for m in start_re.finditer(text):
        end_match = end_re.search(text, m.end())
        end = end_match.start() if end_match else len(text)
        if end - m.start() > best_len:
            best_start, best_len = m.start(), end - m.start()
    if best_start is not None and best_len >= min_len:
        return text[best_start: best_start + min(best_len, max_chars)]
    return None


def extract_risk_section(text: str, max_chars: int = 6000):
    """Item 1A Risk Factors, or None if it can't be located."""
    return extract_section(text, _ITEM_1A, _SECTION_END, max_chars)


def extract_mdna_section(text: str, max_chars: int = 15000):
    """Item 7 Management's Discussion and Analysis, or None."""
    return extract_section(text, _ITEM_7, _ITEM_7_END, max_chars)
