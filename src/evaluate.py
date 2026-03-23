# src/evaluate.py
import sys, os
sys.path.append(os.path.dirname(__file__))
from rag import generate_signal
from collections import Counter
import json

#CONSISTENCY SCORE 
def consistency_score(ticker, n=3):
    print(f"Running consistency check for {ticker} ({n} runs)...")
    signals     = [generate_signal(ticker) for _ in range(n)]
    risk_levels = [s.get("risk_level") for s in signals]
    sentiments  = [s.get("sentiment")  for s in signals]

    risk_score = max(Counter(risk_levels).values()) / n
    sent_score = max(Counter(sentiments).values()) / n

    return {
        "ticker":                ticker,
        "risk_consistency":      round(risk_score, 2),
        "sentiment_consistency": round(sent_score, 2),
        "most_common_risk":      Counter(risk_levels).most_common(1)[0][0],
        "most_common_sentiment": Counter(sentiments).most_common(1)[0][0],
        "all_risk_levels":       risk_levels,
        "all_sentiments":        sentiments
    }

# RUN EVALUATION 
if __name__ == "__main__":
    tickers = ["AAPL", "JPM", "GS", "BAC", "MSFT"]
    results = []

    for ticker in tickers:
        result = consistency_score(ticker, n=3)
        results.append(result)
        print(json.dumps(result, indent=2))

    # summary
    avg_risk_consistency = sum(r["risk_consistency"] for r in results) / len(results)
    avg_sent_consistency = sum(r["sentiment_consistency"] for r in results) / len(results)
    print(f"\n📊 Average Risk Consistency:      {round(avg_risk_consistency*100)}%")
    print(f"📊 Average Sentiment Consistency: {round(avg_sent_consistency*100)}%")


# BACKTEST SIGNAL VS STOCK RETURNS
import yfinance as yf
from datetime import datetime, timedelta

# approximate 10-K filing dates (fiscal year end + ~60 days for filing)
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
    """Get stock return for N days after filing date."""
    start = datetime.strptime(filing_date_str, "%Y-%m-%d")
    end   = start + timedelta(days=days)
    
    # cap end date at today
    if end > datetime.today():
        end = datetime.today()
    
    stock = yf.Ticker(ticker)
    hist  = stock.history(start=start, end=end)
    
    if len(hist) < 5:
        return None
    
    return_pct = (hist["Close"].iloc[-1] - hist["Close"].iloc[0]) / hist["Close"].iloc[0]
    return round(float(return_pct) * 100, 2)

def backtest_signals(tickers=None, days=90):
    if tickers is None:
        tickers = list(FILING_DATES.keys())
    
    results  = []
    correct  = 0
    total    = 0

    for ticker in tickers:
        if ticker not in FILING_DATES:
            print(f"⚠️ No filing dates for {ticker}, skipping")
            continue

        for year, filing_date in FILING_DATES[ticker].items():
            print(f"\n📊 {ticker} {year} (filed {filing_date})")

            # generate signal for that year
            signal     = generate_signal(ticker, year=year)
            risk_level = signal.get("risk_level", "MEDIUM")
            sentiment  = signal.get("sentiment",  "NEUTRAL")

            # get actual stock return
            ret = get_stock_return(ticker, filing_date, days=days)
            if ret is None:
                print(f"   ⚠️ Not enough price data, skipping")
                continue

            # evaluate: HIGH risk or BEARISH should predict negative returns
            predicted_negative = risk_level == "HIGH" or sentiment == "BEARISH"
            actual_negative    = ret < 0
            is_correct         = predicted_negative == actual_negative

            if is_correct:
                correct += 1
            total += 1

            result = {
                "ticker":             ticker,
                "year":               year,
                "filing_date":        filing_date,
                "risk_level":         risk_level,
                "sentiment":          sentiment,
                "actual_return_90d":  f"{ret}%",
                "predicted_negative": predicted_negative,
                "actual_negative":    actual_negative,
                "correct":            is_correct
            }
            results.append(result)
            print(f"   Risk: {risk_level} | Sentiment: {sentiment} | 90d Return: {ret}% | Correct: {is_correct}")

    accuracy = round(correct / total * 100, 1) if total > 0 else 0
    print(f"\n{'='*50}")
    print(f"✅ Backtest Accuracy: {correct}/{total} = {accuracy}%")
    print(f"📅 Holding period: {days} days post-filing")
    return results, accuracy

# RUN BACKTEST
if __name__ == "__main__":
    print("Running consistency scores...")
    for ticker in ["AAPL", "JPM", "GS"]:
        result = consistency_score(ticker, n=3)
        print(f"{ticker}: Risk {result['risk_consistency']*100}% | Sentiment {result['sentiment_consistency']*100}%")

    print("\n\nRunning backtest...")
    results, accuracy = backtest_signals()
    print(f"\nFinal Backtest Accuracy: {accuracy}%")


# ── RETRIEVAL EVALUATION: Recall@K, MRR, NDCG@K ─────────────────
import math

# ground truth — manually defined relevant ticker for each query
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

def recall_at_k(vs, query, relevant_tickers, k=5):
    results  = vs.similarity_search(query, k=k)
    retrieved = [r.metadata.get("source") for r in results]
    hits     = [t for t in retrieved if t in relevant_tickers]
    return min(len(hits), len(relevant_tickers)) / len(relevant_tickers)

def reciprocal_rank(vs, query, relevant_tickers, k=10):
    results  = vs.similarity_search(query, k=k)
    retrieved = [r.metadata.get("source") for r in results]
    for i, ticker in enumerate(retrieved):
        if ticker in relevant_tickers:
            return 1.0 / (i + 1)
    return 0.0

def ndcg_at_k(vs, query, relevant_tickers, k=5):
    results   = vs.similarity_search(query, k=k)
    retrieved = [r.metadata.get("source") for r in results]
    dcg  = sum(
        1.0 / math.log2(i + 2)
        for i, t in enumerate(retrieved)
        if t in relevant_tickers
    )
    idcg = sum(
        1.0 / math.log2(i + 2)
        for i in range(min(len(relevant_tickers), k))
    )
    return round(dcg / idcg, 3) if idcg > 0 else 0.0

def run_retrieval_evaluation(k=5):
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS

    INDEX_DIR  = os.path.expanduser("~/sec-analyzer/embeddings/faiss_index")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vs         = FAISS.load_local(INDEX_DIR, embeddings,
                                  allow_dangerous_deserialization=True)

    recalls, mrrs, ndcgs = [], [], []

    print(f"\n{'='*70}")
    print(f"{'Query':<45} {'R@5':>6} {'MRR':>6} {'NDCG@5':>8}")
    print(f"{'='*70}")

    for query, relevant in GROUND_TRUTH.items():
        r   = recall_at_k(vs, query, relevant, k=k)
        mrr = reciprocal_rank(vs, query, relevant, k=10)
        ndcg = ndcg_at_k(vs, query, relevant, k=k)

        recalls.append(r)
        mrrs.append(mrr)
        ndcgs.append(ndcg)

        print(f"{query[:44]:<45} {r:>6.2f} {mrr:>6.2f} {ndcg:>8.3f}")

    print(f"{'='*70}")
    print(f"{'AVERAGE':<45} {sum(recalls)/len(recalls):>6.2f} "
          f"{sum(mrrs)/len(mrrs):>6.2f} "
          f"{sum(ndcgs)/len(ndcgs):>8.3f}")

    return {
        "mean_recall_at_k":  round(sum(recalls)/len(recalls), 3),
        "mrr":               round(sum(mrrs)/len(mrrs), 3),
        "mean_ndcg_at_k":    round(sum(ndcgs)/len(ndcgs), 3),
        "k":                 k
    }

if __name__ == "__main__":
    print("Running retrieval evaluation...")
    metrics = run_retrieval_evaluation(k=5)
    print(f"\n📊 Final Metrics:")
    print(f"   Recall@5:  {metrics['mean_recall_at_k']}")
    print(f"   MRR:       {metrics['mrr']}")
    print(f"   NDCG@5:    {metrics['mean_ndcg_at_k']}")