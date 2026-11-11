# src/trend.py
from config import FILINGS_DIR, parse_filing_path
from rag import DEFAULT_QUERY, generate_signal


# GET ALL FILINGS FOR A TICKER WITH YEARS
def get_filings_by_year(ticker):
    filings = []
    for filepath in (FILINGS_DIR / ticker / "10-K").glob("*/full-submission.txt"):
        _, accession, year = parse_filing_path(filepath)
        if year:
            filings.append({"year": year, "folder": accession, "filepath": str(filepath)})
    return sorted(filings, key=lambda x: x["year"])


# GENERATE SIGNAL PER YEAR
def generate_trend(ticker, query=DEFAULT_QUERY, **signal_kwargs):
    trend = []
    for filing in get_filings_by_year(ticker):
        print(f"  Analyzing {ticker} - {filing['year']}...")
        signal = generate_signal(ticker, query, year=filing["year"], **signal_kwargs)
        signal["year"] = filing["year"]
        trend.append(signal)
    return trend


# MAP SIGNALS TO NUMERIC FOR CHARTING
def sentiment_to_score(sentiment):
    return {"BULLISH": 1, "NEUTRAL": 0, "BEARISH": -1}.get(sentiment, 0)


def risk_to_score(risk):
    return {"LOW": 1, "MEDIUM": 2, "HIGH": 3}.get(risk, 0)


if __name__ == "__main__":
    for ticker in ["JPM", "GS"]:
        print(f"\n{'=' * 50} {ticker}")
        for t in generate_trend(ticker):
            print(f"  {t['year']} → Sentiment: {t.get('sentiment')} | Risk: {t.get('risk_level')}")
