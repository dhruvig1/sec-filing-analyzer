# tests/test_filing_layouts.py
"""One synthetic filing per 10-K layout seen in the real corpus (see the comment on each test)."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from text_utils import clean_filing_text, extract_mdna_section, extract_risk_section  # noqa: E402

RISK = "Our business could be adversely affected by supply chain concentration. " * 60
MDNA = "Net revenue increased 8% driven by higher net interest income. " * 60


def doc(body, doc_type="10-K"):
    return f"<DOCUMENT>\n<TYPE>{doc_type}\n<TEXT><html><body>{body}</body></html></TEXT>\n</DOCUMENT>\n"


def risk(raw):
    return extract_risk_section(clean_filing_text(raw), max_chars=100_000)


def mdna(raw):
    return extract_mdna_section(clean_filing_text(raw), max_chars=100_000)


def test_small_caps_headings_msft_slb():
    raw = doc(f"""<p>Item 1A. Risk Factors</p><p>Item 1B. Unresolved Staff Comments</p>
    <p>ITEM 1A. R<span style="font-size:8pt">ISK</span> F<span style="font-size:8pt">ACTORS</span></p>
    <p>{RISK}</p>
    <p>ITEM 1B. U<span>NRESOLVED</span> S<span>TAFF</span> C<span>OMMENTS</span></p><p>None.</p>""")
    got = risk(raw)
    assert got and got.startswith("ITEM 1A. RISK FACTORS") and "Unresolved" not in got.title()


def test_item_1a_points_to_ex13_annual_report_wfc_pgr():
    main = doc("""<p>ITEM 1A. RISK FACTORS</p><p>Information in response to this Item 1A can be found
    in the "Risk Factors" section of the 2025 Annual Report to Shareholders.</p>
    <p>ITEM 1B. UNRESOLVED STAFF COMMENTS</p><p>None.</p>""")
    ex13 = doc(f"""<p>Risk Factors</p><p>{RISK}</p><p>Controls and Procedures</p><p>Evaluated.</p>""", "EX-13")
    got = risk(main + ex13)
    assert got and got.startswith("Risk Factors\nOur business") and "Evaluated" not in got


def test_plain_titles_no_item_headings_ms_c_jpm():
    raw = doc(f"""<p>Business</p><p>1</p><p>Risk Factors</p><p>1A 13</p><p>Cybersecurity</p><p>1C 30</p>
    <p>Business</p><p>We are a bank. See "Risk Factors" below.</p>
    <p>Risk Factors</p><p>{RISK}</p>
    <p>Cybersecurity</p><p>The CISO oversees the program.</p>
    <p>Management&#8217;s Discussion and Analysis of Financial Condition and Results of Operations</p>
    <p>{MDNA}</p><p>Management&#8217;s discussion and analysis</p><p>{MDNA}</p>
    <p>Quantitative and Qualitative Disclosures about Market Risk</p><p>VaR.</p>""")
    r = risk(raw)
    assert r.startswith("Risk Factors\nOur business") and "CISO" not in r
    m = mdna(raw)
    assert m.startswith("Management’s Discussion and Analysis of Financial Condition")
    assert len(m) > 2 * len(MDNA) - 100 and "VaR" not in m  # running header didn't truncate it


def test_item_7_is_a_pointer_real_mdna_later_xom_cvx():
    raw = doc(f"""<p>ITEM 7. MANAGEMENT'S DISCUSSION AND ANALYSIS OF FINANCIAL CONDITION AND RESULTS OF OPERATIONS</p>
    <p>Reference is made to the section entitled "Management's Discussion and Analysis" in the Financial Section.</p>
    <p>ITEM 7A. QUANTITATIVE AND QUALITATIVE DISCLOSURES ABOUT MARKET RISK</p><p>See the Financial Section.</p>
    <p>ITEM 8. FINANCIAL STATEMENTS AND SUPPLEMENTARY DATA</p><p>See the Financial Section.</p>
    <p>FINANCIAL SECTION</p>
    <p>MANAGEMENT'S DISCUSSION AND ANALYSIS OF FINANCIAL CONDITION AND RESULTS OF OPERATIONS</p><p>{MDNA}</p>
    <p>REPORT OF INDEPENDENT REGISTERED PUBLIC ACCOUNTING FIRM</p><p>Opinion.</p>""")
    got = mdna(raw)
    assert got.startswith("MANAGEMENT'S DISCUSSION AND ANALYSIS OF FINANCIAL CONDITION AND RESULTS OF OPERATIONS\nNet revenue")
    assert "Opinion" not in got


def test_crossref_to_item_8_inside_mdna_does_not_end_it_pfe():
    raw = doc(f"""<p>ITEM 7. MANAGEMENT&#8217;S DISCUSSION AND ANALYSIS OF FINANCIAL CONDITION AND RESULTS OF OPERATIONS</p>
    <p>This MD&amp;A should be read with the statements included in Item 8. Financial Statements and Supplementary Data.</p>
    <p>{MDNA}</p>
    <p>ITEM 7A. QUANTITATIVE AND QUALITATIVE DISCLOSURES ABOUT MARKET RISK</p><p>VaR.</p>""")
    got = mdna(raw)
    assert got and MDNA.strip()[:40] in got and "VaR" not in got


def test_bullet_crossref_and_pipe_running_headers_aig():
    raw = doc(f"""<p>ITEM 7</p><p>Management&#8217;s Discussion and Analysis 50</p><p>ITEM 7A</p><p>Quantitative 120</p>
    <p>Item 1. Business</p><p>Forward-looking statements are discussed in:</p>
    <p>&#8211; this Part II, Item 7. Management's Discussion and Analysis of Financial Condition.</p>
    <p>ITEM 7 | Management&#8217;s Discussion and Analysis of Financial Condition and Results of Operations</p>
    <p>{MDNA}</p>
    <p>ITEM 7 | Management&#8217;s Discussion and Analysis of Financial Condition and Results of Operations</p>
    <p>{MDNA}</p>
    <p>ITEM 7A | Quantitative and Qualitative Disclosures About Market Risk</p><p>VaR.</p>""")
    got = mdna(raw)
    assert got.startswith("ITEM 7 | Management")
    assert "Forward-looking" not in got and len(got) > 2 * len(MDNA) - 100 and "VaR" not in got


def test_mdna_called_financial_review_with_toc_wfc():
    main = doc("""<p>ITEM 7. MANAGEMENT&#8217;S DISCUSSION AND ANALYSIS OF FINANCIAL CONDITION AND RESULTS OF OPERATIONS</p>
    <p>Information in response to this item can be found in the 2025 Annual Report under "Financial Review."</p>
    <p>ITEM 7A. QUANTITATIVE AND QUALITATIVE DISCLOSURES ABOUT MARKET RISK</p><p>See above.</p>""")
    toc = "".join(f"<p>Section {i} {i * 3}</p>" for i in range(1, 12))
    ex13 = doc(f"""<p>Contents</p><p>Financial Review</p>{toc}
    <p>Letter to shareholders: we had a great year.</p>
    <p>Financial Review</p><p>{MDNA}</p><p>Financial Review</p><p>{MDNA}</p>
    <p>Report of Independent Registered Public Accounting Firm</p><p>Opinion.</p>""", "EX-13")
    got = mdna(main + ex13)
    assert got.startswith("Financial Review\nNet revenue")
    assert "Letter to shareholders" not in got and "Opinion" not in got and len(got) > 2 * len(MDNA) - 100


def test_mdna_after_financial_statements_pgr():
    main = doc("""<p>ITEM 7. MANAGEMENT&#8217;S DISCUSSION AND ANALYSIS OF FINANCIAL CONDITION AND RESULTS OF OPERATIONS</p>
    <p>Incorporated by reference from the Annual Report.</p>
    <p>ITEM 8. FINANCIAL STATEMENTS AND SUPPLEMENTARY DATA</p><p>Incorporated by reference.</p>
    <p>Our cybersecurity program is described in Management's Discussion and Analysis of Financial Condition.</p>""")
    ex13 = doc(f"""<p>Report of Independent Registered Public Accounting Firm</p><p>Opinion.</p>
    <p>Consolidated Financial Statements</p><p>Balance sheet.</p>
    <p>Management&#8217;s Discussion and Analysis of Financial Condition and Results of Operations</p>
    <p>{MDNA}</p>""", "EX-13")
    got = mdna(main + ex13)
    assert got.startswith("Management’s Discussion and Analysis of Financial Condition") and "Balance sheet" not in got


def test_mdna_body_starts_at_first_toc_subheading_cvx():
    raw = doc(f"""<p>7.</p><p>Management&#8217;s Discussion and Analysis of Financial Condition and Results of Operations</p>
    <p>7A.</p><p>Quantitative and Qualitative Disclosures About Market Risk</p>
    <p>Item 7. Management&#8217;s Discussion and Analysis of Financial Condition and Results of Operations</p>
    <p>The index to Management's Discussion and Analysis is presented on page 34.</p>
    <p>Item 7A. Quantitative and Qualitative Disclosures About Market Risk</p><p>See page 50.</p>
    <p>Financial Table of Contents</p>
    <p>Management&#8217;s Discussion and Analysis of<br>Financial Condition and Results of Operations</p>
    <p>Key Financial Results 34</p><p>Earnings by Major Operating Area 35</p><p>Liquidity and Capital Resources 40</p>
    <p>Consolidated Financial Statements</p><p>Reports of Management 52</p>
    <p>Key Financial Results</p><p>{MDNA}</p><p>Financial Table of Contents</p><p>{MDNA}</p>
    <p>Reports of Management</p><p>Management is responsible.</p>
    <p>Report of Independent Registered Public Accounting Firm</p><p>Opinion.</p>""")
    got = mdna(raw)
    assert got and got.startswith("Key Financial Results\nNet revenue")
    assert "Opinion" not in got and len(got) > 2 * len(MDNA) - 100
