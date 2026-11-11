# tests/test_sections_and_signals.py
"""Offline tests: synthetic 10-Ks, deterministic embeddings, mocked LLM. Run: pytest -q"""
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import rag                      # noqa: E402
from llm import LLMResult       # noqa: E402
from text_utils import clean_filing_text, extract_mdna_section, extract_risk_section  # noqa: E402

RISK = "Our business could be adversely affected by supply chain concentration. " * 80
MDNA = "The provision for credit losses increased $1.9 billion to $4.4 billion. " * 80


def filing(extra_before_1a="", running_header=False):
    header = "<p>Item 1A. Risk Factors (continued)</p>" if running_header else ""
    return f"""<DOCUMENT>
<TYPE>10-K
<TEXT><html><body>
<p>TABLE OF CONTENTS</p>
<p>Item 1A. Risk Factors 12</p><p>Item 1B. Unresolved Staff Comments 30</p>
<p>Item 7. Management&#8217;s Discussion and Analysis 40</p><p>Item 7A. Quantitative 60</p>
<p>Item 1. Business</p><p>We sell devices. {extra_before_1a}</p>
<p>Item 1A. Risk Factors</p><p>{RISK}</p><p>Table of Contents</p><p>17</p>{header}<p>{RISK}</p>
<p>Item 1B. Unresolved Staff Comments</p><p>None.</p>
<p>Item 1C. Cybersecurity</p><p>The CISO oversees our information security program.</p>
<p>Item 2. Properties</p><p>Offices.</p>
<p>Item 7. Management&#8217;s Discussion and Analysis</p>
<p>{MDNA} For a discussion of risks, see Part I, Item 1A. Risk Factors of this report. {MDNA}</p>
<p>Item 7A. Quantitative and Qualitative Disclosures About Market Risk</p><p>Rates.</p>
<p>Item 8. Financial Statements</p><p>Numbers.</p>
</body></html></TEXT>
</DOCUMENT>
<DOCUMENT>
<TYPE>EX-21
<TEXT>Exhibit text</TEXT>
</DOCUMENT>"""


def test_risk_section_is_real_item_1a_not_toc_or_mdna_crossref():
    text = clean_filing_text(filing())
    risk = extract_risk_section(text, max_chars=100_000)
    assert risk.startswith("Item 1A. Risk Factors\nOur business")
    assert "provision for credit losses" not in risk  # the MD&A cross-ref used to win
    assert "Unresolved" not in risk and "CISO" not in risk
    assert "Table of Contents" not in risk and "Exhibit text" not in text


def test_crossref_in_item_1_does_not_start_the_section():
    text = clean_filing_text(filing(extra_before_1a="See Item 1A. Risk Factors for more."))
    assert extract_risk_section(text, max_chars=100_000).startswith("Item 1A. Risk Factors\nOur business")


def test_running_page_header_does_not_truncate_section():
    text = clean_filing_text(filing(running_header=True))
    risk = extract_risk_section(text, max_chars=100_000)
    assert risk.startswith("Item 1A. Risk Factors\nOur business")
    assert len(risk) > 2 * len(RISK)  # both halves kept


def test_mdna_section_bounds():
    mdna = extract_mdna_section(clean_filing_text(filing()), max_chars=100_000)
    assert mdna.startswith("Item 7. Management")
    assert "Our business could be adversely" not in mdna and "Rates." not in mdna


# ── signal validation + retry ──────────────────────────────────
VALID = {"ticker": "AAPL", "risk_level": "high", "sentiment": " Bearish",
         "key_risks": ["a", "b", "c"], "rationale": "Tariffs rose 10%.", "red_flags": []}


def fake_complete(replies, calls):
    it = iter(replies)
    def _complete(messages, backend=None, model=None, **kwargs):
        calls.append(list(messages))
        return LLMResult(text=next(it), backend="groq", model="m", latency_s=1.0,
                         input_tokens=100, output_tokens=50)
    return _complete


@pytest.fixture
def ctx(monkeypatch):
    monkeypatch.setattr(rag, "build_context", lambda *a, **k: ("[excerpt] text", [{"type": "x"}]))


def test_valid_first_try_and_casing_normalised(ctx, monkeypatch):
    calls = []
    monkeypatch.setattr(rag, "complete", fake_complete([json.dumps(VALID)], calls))
    sig = rag.generate_signal("AAPL", year=2025)
    assert len(calls) == 1 and "error" not in sig
    assert sig["risk_level"] == "HIGH" and sig["sentiment"] == "BEARISH"
    assert sig["_meta"]["attempts"] == 1


def test_retry_fixes_invalid_output_and_sums_cost(ctx, monkeypatch):
    calls = []
    bad = json.dumps({**VALID, "risk_level": "SEVERE", "key_risks": ["a"]})
    monkeypatch.setattr(rag, "complete", fake_complete([bad, json.dumps(VALID)], calls))
    sig = rag.generate_signal("AAPL", year=2025)
    assert "error" not in sig and sig["_meta"]["attempts"] == 2
    assert "risk_level='SEVERE'" in calls[1][-1]["content"]  # errors fed back to the model
    assert sig["_meta"]["input_tokens"] == 200 and sig["_meta"]["latency_s"] == 2.0


def test_still_invalid_after_retry_returns_error(ctx, monkeypatch):
    monkeypatch.setattr(rag, "complete", fake_complete(["nope", json.dumps({**VALID, "sentiment": "UP"})], []))
    sig = rag.generate_signal("AAPL", year=2025)
    assert "schema validation" in sig["error"]


def test_run_llm_defaults_to_single_attempt_for_backend_comparison(monkeypatch):
    calls = []
    monkeypatch.setattr(rag, "complete", fake_complete(["not json"], calls))
    sig, result = rag.run_llm("AAPL", 2025, "q", "ctx")
    assert len(calls) == 1 and sig["error"] == "Could not parse JSON"


# ── filtered retrieval on a real FAISS index ───────────────────
def test_filtered_retrieval_exact_ticker_and_year(tmp_path, monkeypatch):
    import ingest
    from langchain_community.embeddings import DeterministicFakeEmbedding

    monkeypatch.setattr(ingest, "INDEX_DIR", tmp_path / "faiss_index")
    docs = [{"text": f"{t} {y} risk text. " * 300, "source": t, "section": "RISK FACTORS",
             "file": "f", "year": y} for t in ["C", "COP", "MS", "MSFT"] for y in [2024, 2025]]
    vs = ingest.build_vector_store(docs, embeddings=DeterministicFakeEmbedding(size=32))
    got = rag.retrieve_context(vs, "c", "risk", year=2025, k=5)
    assert got and all(d.metadata["source"] == "C" and d.metadata["year"] == 2025 for d in got)
    assert got[0].metadata["chunk_id"].startswith("C-2025-RISK-")
