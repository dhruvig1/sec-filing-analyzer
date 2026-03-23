# src/ingest.py
from sec_edgar_downloader import Downloader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
import os, glob

load_dotenv()

# CONFIG
#TICKERS   = ["AAPL", "JPM", "BAC", "GS", "BLK"]
TICKERS = [
    "JPM", "BAC", "GS", "MS", "WFC", "C", "BLK", "AXP",
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA",
    "PGR", "TRV", "MET", "AIG", "PRU", "ALL",
    "JNJ", "UNH", "PFE", "ABBV", "CVS",
    "XOM", "CVX", "COP", "SLB",
    "WMT", "HD", "NKE", "MCD", "SBUX"
]
DATA_DIR  = os.path.join(os.path.expanduser("~/sec-analyzer"), "data")
INDEX_DIR = os.path.join(os.path.expanduser("~/sec-analyzer"), "embeddings", "faiss_index")

# STEP 2: DOWNLOAD FILINGS FROM SEC EDGAR
def download_filings():
    dl = Downloader("Dhruvi Gandhi", "dhruvigandhii05@gmail.com", DATA_DIR)
    for ticker in TICKERS:
        print(f"Downloading 10-K for {ticker}...")
        dl.get("10-K", ticker, limit=3)
    print("All filings downloaded.")

# STEP 3: CHUNK + EMBED INTO FAISS
def load_and_chunk_filings():
    docs = []
    for filepath in glob.glob(f"{DATA_DIR}/**/*.txt", recursive=True):
        with open(filepath, "r", errors="ignore") as f:
            text = f.read()
        for section in ["RISK FACTORS", "MANAGEMENT'S DISCUSSION"]:
            idx = text.upper().find(section)
            if idx != -1:
                chunk = text[idx:idx+15000]
                parts = filepath.split(os.sep)
                ticker = parts[6] if len(parts) > 6 else "UNKNOWN"
                
                # extract year from accession folder name e.g. 0000019617-25-000270 → 2025
                accession = parts[8] if len(parts) > 8 else ""
                year_digits = accession.split("-")[1] if "-" in accession else "00"
                year = int("20" + year_digits)
                
                docs.append({
                    "text":    chunk,
                    "source":  ticker,
                    "section": section,
                    "file":    filepath,
                    "year":    year
                })
                
    print(f"Found {len(docs)} sections across all filings.")
    return docs

def build_vector_store(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = splitter.create_documents(
        [d["text"] for d in docs],
        metadatas=[{
            "source":  d["source"],
            "section": d["section"],
            "file":    d["file"],
            "year":    d["year"]
        } for d in docs]
    )
    print(f"Split into {len(splits)} chunks.")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(splits, embeddings)
    os.makedirs("embeddings", exist_ok=True)
    vectorstore.save_local(INDEX_DIR)
    print(f"Vector store saved to {INDEX_DIR}")
    return vectorstore

# RUN
if __name__ == "__main__":
    download_filings()
    docs = load_and_chunk_filings()
    if docs:
        build_vector_store(docs)
    else:
        print("No sections found — check your data/ folder.")