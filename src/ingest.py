# src/ingest.py
"""Download 10-K filings from SEC EDGAR, extract Item 1A / Item 7, embed locally, and build FAISS."""
import os

from dotenv import load_dotenv

from config import (DATA_DIR, EMBEDDING_MODEL, FILINGS_DIR, FILINGS_PER_TICKER,
                    INDEX_DIR, TICKERS, parse_filing_path)
from text_utils import clean_filing_text, extract_mdna_section, extract_risk_section

load_dotenv()

SEC_USER_AGENT_NAME = os.getenv("SEC_USER_AGENT_NAME", "Dhruvi Gandhi")
SEC_USER_AGENT_EMAIL = os.getenv("SEC_USER_AGENT_EMAIL", "dhruvigandhii05@gmail.com")


# STEP 1: DOWNLOAD FILINGS FROM SEC EDGAR
def download_filings(tickers=TICKERS):
    from sec_edgar_downloader import Downloader

    dl = Downloader(SEC_USER_AGENT_NAME, SEC_USER_AGENT_EMAIL, str(DATA_DIR))
    for ticker in tickers:
        print(f"Downloading 10-K for {ticker}...")
        dl.get("10-K", ticker, limit=FILINGS_PER_TICKER)
    print("All filings downloaded.")


# STEP 2: EXTRACT SECTIONS
def load_and_chunk_filings():
    docs, n_filings = [], 0
    for filepath in sorted(FILINGS_DIR.glob("*/10-K/*/full-submission.txt")):
        ticker, _, year = parse_filing_path(filepath)
        if ticker is None or year is None:
            continue
        n_filings += 1
        with open(filepath, "r", errors="ignore") as f:
            text = clean_filing_text(f.read())
        sections = {
            "RISK FACTORS": extract_risk_section(text, max_chars=20000),
            "MANAGEMENT'S DISCUSSION": extract_mdna_section(text, max_chars=15000),
        }
        for name, body in sections.items():
            if body:
                docs.append({"text": body, "source": ticker, "section": name,
                             "file": str(filepath), "year": year})
    print(f"Found {len(docs)} sections across {n_filings} filings.")
    return docs


# STEP 3: CHUNK + EMBED INTO FAISS
def build_vector_store(docs):
    from langchain_community.vectorstores import FAISS
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = splitter.create_documents(
        [d["text"] for d in docs],
        metadatas=[{k: d[k] for k in ("source", "section", "file", "year")} for d in docs],
    )
    print(f"Split into {len(splits)} chunks.")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectorstore = FAISS.from_documents(splits, embeddings)
    INDEX_DIR.parent.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(INDEX_DIR))
    print(f"Vector store saved to {INDEX_DIR}")
    return vectorstore


if __name__ == "__main__":
    download_filings()
    docs = load_and_chunk_filings()
    if docs:
        build_vector_store(docs)
    else:
        print(f"No sections found — check {FILINGS_DIR}.")
