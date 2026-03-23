# src/trend.py
import os
import json
import glob
from rag import generate_signal

DATA_DIR = "data/sec-edgar-filings"

#EXTRACT YEAR FROM ACCESSION NUMBER
def extract_year(accession_folder):
    # format: 0000019617-25-000270 → 2025
    parts = accession_folder.split("-")
    if len(parts) >= 2:
        year_short = parts[1]
        year = int("20" + year_short)
        return year
    return None

# GET ALL FILINGS FOR A TICKER WITH YEARS 
def get_filings_by_year(ticker):
    ticker_dir = os.path.join(DATA_DIR, ticker, "10-K")
    if not os.path.exists(ticker_dir):
        return []

    filings = []
    for folder in os.listdir(ticker_dir):
        year = extract_year(folder)
        filepath = os.path.join(ticker_dir, folder, "full-submission.txt")
        if year and os.path.exists(filepath):
            filings.append({"year": year, "folder": folder, "filepath": filepath})

    return sorted(filings, key=lambda x: x["year"])

# GENERATE SIGNAL PER YEAR 
def generate_trend(ticker, query="What are the major risk factors and investment outlook?"):
    filings = get_filings_by_year(ticker)
    if not filings:
        return []

    trend = []
    for filing in filings:
        print(f"  Analyzing {ticker} - {filing['year']}...")
        signal = generate_signal(ticker, query, year=filing["year"])  # ← pass year
        signal["year"] = filing["year"]
        trend.append(signal)

    return trend

# MAP SIGNALS TO NUMERIC FOR CHARTING 
def sentiment_to_score(sentiment):
    return {"BULLISH": 1, "NEUTRAL": 0, "BEARISH": -1}.get(sentiment, 0)

def risk_to_score(risk):
    return {"LOW": 1, "MEDIUM": 2, "HIGH": 3}.get(risk, 0)

# TEST 
if __name__ == "__main__":
    for ticker in ["JPM", "GS"]:
        print(f"\n{'='*50} {ticker}")
        trend = generate_trend(ticker)
        for t in trend:
            print(f"  {t['year']} → Sentiment: {t.get('sentiment')} | Risk: {t.get('risk_level')}")