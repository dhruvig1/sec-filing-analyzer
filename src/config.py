# src/config.py
"""Shared configuration: paths, company universe, and filing-path parsing.

All paths resolve relative to the repo root (the parent of src/), so the same
code works when run locally, inside the Streamlit Docker image (/app), and
inside Airflow (/opt/airflow). Override with SEC_ANALYZER_ROOT if needed.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()  # so EMBEDDING_MODEL / RETRIEVER in .env apply before the settings below are read

PROJECT_ROOT = Path(os.getenv("SEC_ANALYZER_ROOT", Path(__file__).resolve().parent.parent))
DATA_DIR = PROJECT_ROOT / "data"                       # sec-edgar-downloader root
FILINGS_DIR = DATA_DIR / "sec-edgar-filings"           # <TICKER>/10-K/<accession>/full-submission.txt
RESULTS_DIR = PROJECT_ROOT / "results"

# ── Retrieval ───────────────────────────────────────────────────
# Embedding model for the vector index. Each model gets its own index folder, so
# you can build a Qwen index next to the MiniLM one and compare them:
#     EMBEDDING_MODEL=Qwen/Qwen3-Embedding-0.6B python src/ingest.py --skip-download
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)

# Qwen3-Embedding expects a one-line task instruction on queries (not on documents).
QUERY_INSTRUCTIONS = {
    "Qwen/Qwen3-Embedding-0.6B": "Given an analyst's question about a company's SEC 10-K filing, "
                                 "retrieve passages from the filing that answer the question",
}


def index_dir_for(model: str) -> Path:
    """embeddings/faiss_index for the default model (unchanged), embeddings/faiss_index__<model> otherwise."""
    if model == DEFAULT_EMBEDDING_MODEL:
        return PROJECT_ROOT / "embeddings" / "faiss_index"
    slug = model.split("/")[-1].lower().replace(".", "_")
    return PROJECT_ROOT / "embeddings" / f"faiss_index__{slug}"


INDEX_DIR = index_dir_for(EMBEDDING_MODEL)

# dense | bm25 | hybrid (dense + BM25 fused with reciprocal rank fusion)
RETRIEVER = os.getenv("RETRIEVER", "hybrid")

# compare_backends.py scores grounding with a fixed small model, so grounding numbers
# don't change when you switch the retrieval embedding model.
GROUNDING_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
FILINGS_PER_TICKER = 3

SECTORS = {
    "Finance & Banking": ["JPM", "BAC", "GS", "MS", "WFC", "C", "BLK", "AXP"],
    "Tech":              ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA"],
    "Insurance":         ["PGR", "TRV", "MET", "AIG", "PRU", "ALL"],
    "Healthcare":        ["JNJ", "UNH", "PFE", "ABBV", "CVS", "LLY"],
    "Energy":            ["XOM", "CVX", "COP", "SLB", "EOG", "PSX"],
    "Consumer":          ["WMT", "HD", "NKE", "MCD", "SBUX", "PG", "COST"],
}
TICKERS = [t for tickers in SECTORS.values() for t in tickers]   # 40 companies


def year_from_accession(accession: str):
    """0000019617-25-000270 -> 2025 (the YY segment is the year the filing was submitted)."""
    parts = accession.split("-")
    if len(parts) >= 2 and parts[1].isdigit():
        return 2000 + int(parts[1])
    return None


def parse_filing_path(filepath):
    """Return (ticker, accession, filing_year) for .../<TICKER>/10-K/<accession>/<file>.

    Parses from the end of the path so it does not depend on where the data
    folder lives (the old code used fixed indices like parts[6], which broke
    whenever the repo was cloned to a different depth or run in Docker).
    """
    parts = Path(filepath).parts
    if len(parts) < 4 or parts[-3] != "10-K":
        return None, None, None
    ticker, accession = parts[-4], parts[-2]
    return ticker, accession, year_from_accession(accession)
