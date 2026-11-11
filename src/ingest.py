# src/ingest.py
"""Download 10-K filings from SEC EDGAR, extract Item 1A / Item 7, embed locally, and build FAISS."""
import os

from dotenv import load_dotenv

from config import (DATA_DIR, EMBEDDING_MODEL, FILINGS_DIR, FILINGS_PER_TICKER,
                    INDEX_DIR, TICKERS, parse_filing_path)
from text_utils import clean_filing_text, extract_mdna_section, extract_risk_section

load_dotenv()

# Real Item 1A sections are often 40-80k characters; the old 20k cap kept only the start.
RISK_MAX_CHARS = 60_000
MDNA_MAX_CHARS = 40_000

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
    docs, n_filings, missing = [], 0, []
    for filepath in sorted(FILINGS_DIR.glob("*/10-K/*/full-submission.txt")):
        ticker, _, year = parse_filing_path(filepath)
        if ticker is None or year is None:
            continue
        n_filings += 1
        with open(filepath, "r", errors="ignore") as f:
            text = clean_filing_text(f.read())
        sections = {
            "RISK FACTORS": extract_risk_section(text, max_chars=RISK_MAX_CHARS),
            "MANAGEMENT'S DISCUSSION": extract_mdna_section(text, max_chars=MDNA_MAX_CHARS),
        }
        for name, body in sections.items():
            if body:
                docs.append({"text": body, "source": ticker, "section": name,
                             "file": str(filepath), "year": year})
            else:
                missing.append(f"{ticker} {year} {name}")
    print(f"Found {len(docs)} sections across {n_filings} filings.")
    if missing:
        print(f"No section found for {len(missing)}: {', '.join(missing)}")
    return docs


# STEP 3: CHUNK + EMBED INTO FAISS
def split_docs(docs):
    """Chunk sections and give each chunk a stable id. Shared by build_vector_store and remote_embed.py."""
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = splitter.create_documents(
        [d["text"] for d in docs],
        metadatas=[{k: d[k] for k in ("source", "section", "file", "year")} for d in docs],
    )
    # stable chunk ids, e.g. AAPL-2025-RISK-007 (used to cite sources)
    counters = {}
    for doc in splits:
        md = doc.metadata
        key = (md["source"], md["year"], md["section"].split()[0])
        counters[key] = counters.get(key, 0) + 1
        md["chunk_id"] = f"{key[0]}-{key[1]}-{key[2]}-{counters[key]:03d}"
    print(f"Split into {len(splits)} chunks.")
    return splits


def build_vector_store(docs, embeddings=None):
    from langchain_community.vectorstores import FAISS

    splits = split_docs(docs)
    if embeddings is None:
        from embeddings import get_embeddings
        embeddings = get_embeddings(EMBEDDING_MODEL)
    print(f"Embedding with {EMBEDDING_MODEL} -> {INDEX_DIR}")
    vectorstore = FAISS.from_documents(splits, embeddings)
    INDEX_DIR.parent.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(INDEX_DIR))
    print(f"Vector store saved to {INDEX_DIR}")
    return vectorstore


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download 10-Ks and build the FAISS index "
                                                 "(for the model in EMBEDDING_MODEL).")
    parser.add_argument("--skip-download", action="store_true", help="re-index existing filings only")
    if not parser.parse_args().skip_download:
        download_filings()
    docs = load_and_chunk_filings()
    if docs:
        build_vector_store(docs)
    else:
        print(f"No sections found — check {FILINGS_DIR}.")
