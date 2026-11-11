import pytest
from langchain_community.embeddings import FakeEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from evaluate import ndcg_at_k, ranked_unique_tickers, recall_at_k, reciprocal_rank, score_backtest
from rag import retrieve_context


@pytest.fixture(scope="module")
def store():
    docs = []
    for t in ["C", "COP", "CVX", "MS", "MSFT"]:
        for y in [2024, 2025, 2026]:
            for i in range(4):
                docs.append(Document(page_content=f"{t} {y} credit risk text {i}",
                                     metadata={"source": t, "year": y, "section": "RISK FACTORS"}))
    return FAISS.from_documents(docs, FakeEmbeddings(size=32))


def test_exact_ticker_match(store):
    for ticker in ["C", "MS"]:
        docs = retrieve_context(store, ticker, "credit risk", k=10)
        assert docs and {d.metadata["source"] for d in docs} == {ticker}


def test_strict_year_filter_no_lookahead(store):
    docs = retrieve_context(store, "COP", "credit risk", year=2024, k=10)
    assert docs and {d.metadata["year"] for d in docs} == {2024}
    assert retrieve_context(store, "COP", "credit risk", year=2019, k=10) == []  # no fallback to other years


class D:
    def __init__(self, t):
        self.metadata = {"source": t}


def test_duplicate_chunks_do_not_inflate_metrics():
    ranked = ranked_unique_tickers([D("AAPL")] * 5)  # five chunks, one company
    assert ranked == ["AAPL"]
    assert recall_at_k(ranked, ["AAPL", "MSFT"]) == 0.5        # old code: 1.0
    assert ndcg_at_k(ranked, ["AAPL", "MSFT"]) <= 1.0          # old code: ~1.81


def test_metric_values():
    ranked = ["XOM", "JPM", "BAC", "GS", "WFC"]
    rel = ["JPM", "BAC", "WFC"]
    assert recall_at_k(ranked, rel, k=5) == 1.0
    assert reciprocal_rank(ranked, rel) == 0.5
    assert ndcg_at_k(ranked, rel, k=5) == pytest.approx(0.7123, abs=1e-3)  # (1/log2(3)+1/log2(4)+1/log2(6)) / (1+1/log2(3)+1/log2(4))
    assert ndcg_at_k(rel + ["XOM"], rel) == pytest.approx(1.0)


def test_backtest_reports_baseline():
    rows = [{"predicted_negative": False, "actual_negative": False}] * 8 + \
           [{"predicted_negative": False, "actual_negative": True}] * 2
    s = score_backtest(rows)
    assert s["accuracy"] == 80.0 and s["majority_baseline"] == 80.0  # model == baseline here
