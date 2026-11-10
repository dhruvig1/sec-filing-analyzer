# src/evaluate.py
"""Evaluation: retrieval quality, signal consistency, and a return-direction backtest.

    python src/evaluate.py retrieval
    python src/evaluate.py backtest   [--backend ollama] [--context-mode section]
    python src/evaluate.py consistency
    python src/evaluate.py all

Retrieval metrics are computed per COMPANY, not per chunk. The index holds many
chunks per company, so the top-5 chunks often come from one company. The old
code counted each chunk as a separate hit, so one relevant company repeated five
times scored as perfect recall, and NDCG could exceed 1.0. Here retrieved
chunks are collapsed to unique tickers in rank order before scoring.
"""
import argparse
import json
import math
import os
import sys
from collections import Counter
from datetime import datetime, timedelta

sys.path.append(os.path.dirname(__file__))

# ── RETRIEVAL EVALUATION: Recall@K, MRR, NDCG@K ─────────────────
# Ground truth: companies whose filings should be retrieved for each query.
GROUND_TRUTH = {
    "cybersecurity threats information systems":           ["AAPL", "MSFT"],
    "research and development expenses investment":        ["AAPL", "MSFT", "GOOGL"],
    "credit risk loan losses provisions":                  ["JPM", "BAC", "WFC"],
    "market risk trading revenue volatility":              ["GS", "MS", "JPM"],
    "regulatory compliance capital requirements banking":  ["JPM", "BAC", "WFC", "GS"],
    "oil gas production reserves energy":                  ["XOM", "CVX", "COP"],
    "drug approval clinical trials FDA":                   ["PFE", "ABBV", "JNJ"],
    "supply chain disruption manufacturing":               ["AAPL", "NVDA", "AMZN"],
    "interest rate risk financial instruments":            ["JPM", "BAC", "MS"],
    "insurance underwriting premium risk":                 ["PGR", "TRV", "AIG"],
}


def ranked_unique_tickers(docs):
    """Collapse retrieved chunks to companies, keeping first-seen rank order."""
    seen, ranked = set(), []
    for d in docs:
        t = d.metadata.get("source")
        if t not in seen:
            seen.add(t)
            ranked.append(t)
    return ranked


def recall_at_k(ranked, relevant, k=5):
    return len(set(ranked[:k]) & set(relevant)) / len(relevant)


def reciprocal_rank(ranked, relevant, k=10):
    for i, t in enumerate(ranked[:k]):
        if t in relevant:
            return 1.0 / (i + 1)
    return 0.0


def ndcg_at_k(ranked, relevant, k=5):
    dcg = sum(1.0 / math.log2(i + 2) for i, t in enumerate(ranked[:k]) if t in relevant)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg / idcg if idcg > 0 else 0.0


def run_retrieval_evaluation(vs=None, k=5, chunk_pool=100):
    """Retrieve a pool of chunks, collapse to companies, then score the top-k companies."""
    if vs is None:
        from rag import load_vectorstore
        vs = load_vectorstore()

    recalls, mrrs, ndcgs = [], [], []
    print(f"\n{'=' * 70}\n{'Query':<45} {'R@5':>6} {'MRR':>6} {'NDCG@5':>8}\n{'=' * 70}")
    for query, relevant in GROUND_TRUTH.items():
        ranked = ranked_unique_tickers(vs.similarity_search(query, k=chunk_pool))
        r, mrr, ndcg = recall_at_k(ranked, relevant, k), reciprocal_rank(ranked, relevant), ndcg_at_k(ranked, relevant, k)
        recalls.append(r); mrrs.append(mrr); ndcgs.append(ndcg)
        print(f"{query[:44]:<45} {r:>6.2f} {mrr:>6.2f} {ndcg:>8.3f}")

    metrics = {
        "recall_at_k": round(sum(recalls) / len(recalls), 3),
        "mrr": round(sum(mrrs) / len(mrrs), 3),
        "ndcg_at_k": round(sum(ndcgs) / len(ndcgs), 3),
        "k": k,
    }
    print(f"{'=' * 70}\n{'AVERAGE':<45} {metrics['recall_at_k']:>6.2f} {metrics['mrr']:>6.2f} {metrics['ndcg_at_k']:>8.3f}")
    return metrics


# ── CONSISTENCY SCORE ───────────────────────────────────────────
def consistency_score(ticker, n=3, **signal_kwargs):
    from rag import generate_signal

    print(f"Running consistency check for {ticker} ({n} runs)...")
    signals = [generate_signal(ticker, **signal_kwargs) for _ in range(n)]
    risk_levels = [s.get("risk_level") for s in signals]
    sentiments = [s.get("sentiment") for s in signals]
    return {
        "ticker": ticker,
        "risk_consistency": round(max(Counter(risk_levels).values()) / n, 2),
        "sentiment_consistency": round(max(Counter(sentiments).values()) / n, 2),
        "all_risk_levels": risk_levels,
        "all_sentiments": sentiments,
    }


# ── BACKTEST SIGNAL VS STOCK RETURNS ────────────────────────────
# Approximate 10-K filing dates, keyed by the filing year from the accession number.
FILING_DATES = {
    "AAPL": {2023: "2023-11-03", 2024: "2024-11-01", 2025: "2025-11-07"},
    "JPM":  {2024: "2024-02-16", 2025: "2025-02-18", 2026: "2026-02-17"},
    "GS":   {2024: "2024-01-26", 2025: "2025-01-27", 2026: "2026-01-28"},
    "BAC":  {2024: "2024-02-16", 2025: "2025-02-18", 2026: "2026-02-18"},
    "MSFT": {2023: "2023-07-27", 2024: "2024-07-30", 2025: "2025-07-29"},
    "NVDA": {2023: "2023-08-25", 2024: "2024-08-28", 2025: "2025-08-27"},
    "XOM":  {2024: "2024-02-23", 2025: "2025-02-21", 2026: "2026-02-20"},
}


def get_stock_return(ticker, filing_date_str, days=90):
    """Percent return over the `days` calendar days after filing, or None if the window isn't complete."""
    import yfinance as yf

    start = datetime.strptime(filing_date_str, "%Y-%m-%d")
    end = start + timedelta(days=days)
    if end > datetime.today():
        return None  # window not finished yet; a partial window isn't a 90-day return
    hist = yf.Ticker(ticker).history(start=start, end=end)
    if len(hist) < 5:
        return None
    ret = (hist["Close"].iloc[-1] - hist["Close"].iloc[0]) / hist["Close"].iloc[0]
    return round(float(ret) * 100, 2)


def score_backtest(rows):
    """Accuracy of 'HIGH risk or BEARISH => negative return', plus a majority-class baseline.

    The baseline matters: if most 90-day returns were positive, always predicting
    "not negative" already scores well, and the model has to beat that.
    """
    total = len(rows)
    if total == 0:
        return {"n": 0}
    correct = sum(r["predicted_negative"] == r["actual_negative"] for r in rows)
    n_neg = sum(r["actual_negative"] for r in rows)
    return {
        "n": total,
        "accuracy": round(correct / total * 100, 1),
        "majority_baseline": round(max(n_neg, total - n_neg) / total * 100, 1),
        "tp": sum(r["predicted_negative"] and r["actual_negative"] for r in rows),
        "fp": sum(r["predicted_negative"] and not r["actual_negative"] for r in rows),
        "tn": sum(not r["predicted_negative"] and not r["actual_negative"] for r in rows),
        "fn": sum(not r["predicted_negative"] and r["actual_negative"] for r in rows),
    }


def backtest_signals(tickers=None, days=90, **signal_kwargs):
    from rag import generate_signal

    rows = []
    for ticker in tickers or FILING_DATES:
        for year, filing_date in FILING_DATES.get(ticker, {}).items():
            ret = get_stock_return(ticker, filing_date, days=days)
            if ret is None:
                print(f"{ticker} {year}: return window incomplete or no price data, skipping")
                continue
            signal = generate_signal(ticker, year=year, **signal_kwargs)
            risk, sent = signal.get("risk_level", "MEDIUM"), signal.get("sentiment", "NEUTRAL")
            row = {
                "ticker": ticker, "year": year, "filing_date": filing_date,
                "risk_level": risk, "sentiment": sent, "return_pct": ret,
                "predicted_negative": risk == "HIGH" or sent == "BEARISH",
                "actual_negative": ret < 0,
            }
            rows.append(row)
            print(f"{ticker} {year}: {risk}/{sent} | {days}d return {ret}% | "
                  f"correct={row['predicted_negative'] == row['actual_negative']}")

    summary = score_backtest(rows)
    if summary["n"]:
        print(f"\nBacktest accuracy: {summary['accuracy']}% on {summary['n']} filings "
              f"(majority-class baseline {summary['majority_baseline']}%)")
    return rows, summary


# ── CLI ─────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("task", choices=["retrieval", "backtest", "consistency", "all"])
    p.add_argument("--backend", choices=["groq", "ollama"], default=None)
    p.add_argument("--model", default=None)
    p.add_argument("--context-mode", choices=["hybrid", "retrieval", "section"], default=None)
    args = p.parse_args()
    kw = {"backend": args.backend, "model": args.model, "context_mode": args.context_mode}

    if args.task in ("retrieval", "all"):
        print(json.dumps(run_retrieval_evaluation(k=5), indent=2))
    if args.task in ("consistency", "all"):
        for t in ["AAPL", "JPM", "GS", "BAC", "MSFT"]:
            print(json.dumps(consistency_score(t, n=3, **kw), indent=2))
    if args.task in ("backtest", "all"):
        _, summary = backtest_signals(**kw)
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
