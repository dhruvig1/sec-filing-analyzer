from config import TICKERS, parse_filing_path, year_from_accession
from text_utils import clean_filing_text, extract_risk_section

RISK_BODY = "Our business depends on a single supplier for advanced chips. " * 30


def make_submission():
    toc = "<p>Item 1A. Risk Factors ........ 12</p><p>Item 1B. Unresolved Staff Comments ... 30</p>"
    body = (f"<div>Item&#160;1A.&nbsp;Risk Factors</div><p>{RISK_BODY}</p>"
            "<div>Item 1B. Unresolved Staff Comments</div><p>None.</p>"
            "<p>As discussed in Item 1A. Risk Factors, see above.</p>")
    return ("<SEC-DOCUMENT><DOCUMENT>\n<TYPE>10-K\n<TEXT><html><head><style>.x{}</style></head><body>"
            + toc + body + "</body></html></TEXT></DOCUMENT>"
            "<DOCUMENT>\n<TYPE>GRAPHIC\n<TEXT>begin 644 junk.jpg M@#$%</TEXT></DOCUMENT></SEC-DOCUMENT>")


def test_clean_keeps_only_main_document_and_strips_html():
    text = clean_filing_text(make_submission())
    assert "<" not in text and "junk.jpg" not in text and ".x{}" not in text
    assert "single supplier" in text


def test_risk_section_skips_toc_and_cross_reference():
    section = extract_risk_section(clean_filing_text(make_submission()))
    assert section is not None and "single supplier" in section
    assert "Unresolved" not in section


def test_parse_filing_path_independent_of_depth():
    for root in ["/home/me/sec-analyzer/data", "/opt/airflow/data", "data"]:
        p = f"{root}/sec-edgar-filings/JPM/10-K/0000019617-25-000270/full-submission.txt"
        assert parse_filing_path(p) == ("JPM", "0000019617-25-000270", 2025)
    assert parse_filing_path("data/random.txt") == (None, None, None)
    assert year_from_accession("nonsense") is None


def test_universe_is_40_unique_companies():
    assert len(TICKERS) == 40 == len(set(TICKERS))
