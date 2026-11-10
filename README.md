# SEC Filing Analyzer 📊

An LLM-powered RAG pipeline that extracts point-in-time risk signals from SEC 10-K filings, with a swappable inference layer that runs either **hosted Llama-3.3-70B (Groq)** or a **local quantized model (Ollama)**, plus retrieval and backtest evaluation.

---

## Overview

The pipeline ingests 10-K filings from SEC EDGAR for 40 companies across 6 sectors (3 most recent filings each), extracts the Risk Factors (Item 1A) and MD&A (Item 7) sections, embeds them locally with `all-MiniLM-L6-v2`, and indexes them in FAISS. For a question about a company and filing year, it retrieves matching chunks **from that company's filing for that year only**, combines them with the Item 1A excerpt, and asks an LLM for a structured JSON signal (risk level, sentiment, key risks, red flags). A Streamlit dashboard serves the signals.

---

## Features

- **RAG pipeline**: the user's question drives FAISS retrieval, with exact ticker + filing-year metadata filters
- **Point-in-time retrieval**: no fallback to other years, so a signal never sees a later filing (no look-ahead bias)
- **Swappable inference**: `LLM_BACKEND=groq|ollama`, with JSON-mode output on both and token/latency/cost tracking
- **Local vs. hosted benchmark**: `compare_backends.py` scores schema validity, grounding, verbatim red flags, agreement, latency and cost
- **Evaluation**: company-level Recall@5 / MRR / NDCG@5, consistency, and a 90-day return backtest against a majority-class baseline
- **YoY trends and portfolio heatmap**: 3-tab Streamlit UI
- **Orchestration**: Airflow DAG for weekly ingestion, embedding, signals, and S3 sync; fully Dockerized

---

## Tech Stack

| Layer | Tools |
|---|---|
| LLM (hosted) | Llama-3.3-70B via Groq API |
| LLM (local) | Ollama: Llama-3.1-8B, Qwen2.5-7B, or any pulled model (4-bit quantized) |
| Embeddings | `all-MiniLM-L6-v2` (sentence-transformers, runs locally) |
| Vector store | FAISS |
| Chunking & retrieval | LangChain |
| Storage | AWS S3 |
| Orchestration | Apache Airflow |
| Frontend | Streamlit |
| Containerization | Docker |
| Testing | pytest |

---

## Project Structure

```
sec-filing-analyzer/
├── src/
│   ├── app.py               # Streamlit dashboard (3 tabs, backend picker in sidebar)
│   ├── config.py            # paths, 40-company universe, filing-path parsing
│   ├── text_utils.py        # EDGAR HTML cleanup, Item 1A / Item 7 extraction
│   ├── ingest.py            # EDGAR download → section extraction → FAISS index
│   ├── llm.py               # Groq / Ollama inference layer
│   ├── rag.py               # context building, prompting, JSON parsing/validation
│   ├── trend.py             # year-over-year signals
│   ├── evaluate.py          # retrieval metrics, consistency, backtest
│   └── compare_backends.py  # local vs. hosted model benchmark
├── dags/
│   └── sec_pipeline_dag.py  # weekly Airflow pipeline
├── tests/                   # offline unit tests (no API keys or data needed)
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── requirements-dev.txt
```

---

## Evaluation

```bash
python src/evaluate.py retrieval     # Recall@5, MRR, NDCG@5 over 10 labeled queries
python src/evaluate.py backtest      # signal vs. 90-day post-filing return direction
python src/evaluate.py consistency   # agreement across repeated runs
```

Retrieval metrics are scored per company: retrieved chunks are collapsed to unique tickers in rank order before scoring, so several chunks from the same company count once. The backtest skips filings whose 90-day window hasn't finished yet and reports accuracy next to a majority-class baseline.

| Metric | Score |
|---|---|
| Recall@5 | _re-run pending_ |
| MRR | _re-run pending_ |
| NDCG@5 | _re-run pending_ |
| Backtest accuracy (vs. baseline) | _re-run pending_ |

## Local vs. Hosted Models

```bash
ollama pull llama3.1:8b && ollama serve
python src/compare_backends.py --backends groq ollama:llama3.1:8b
python src/compare_backends.py --backends groq ollama:llama3.1:8b ollama:qwen2.5:7b --judge
```

Each model gets the identical prompt and context per filing. Results are written to `results/` (per-filing CSV, summary JSON, markdown table):

| Metric | What it measures |
|---|---|
| `schema_valid_pct` | output has the right keys, enum values, and exactly 3 risks |
| `grounding` / `grounded_frac` | similarity of each extracted risk to the closest sentence in the context (local embeddings) |
| `redflag_verbatim` | share of "red flag" phrases that actually appear in the filing text |
| `judge_faithful` | optional 1–5 LLM-as-judge faithfulness score (`--judge`) |
| `risk/sentiment_agreement_vs_ref` | agreement with the first model listed |
| `median_latency_s`, `total_cost_usd` | speed and API cost (local = $0 per token) |

---

## Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/dhruvig1/sec-filing-analyzer.git
cd sec-filing-analyzer
```

### 2. Set up environment variables
Create a `.env` file:
```
LLM_BACKEND=groq                 # or ollama
GROQ_API_KEY=your_groq_api_key
OLLAMA_MODEL=llama3.1:8b         # used when LLM_BACKEND=ollama
CONTEXT_MODE=hybrid              # hybrid | retrieval | section
AWS_ACCESS_KEY_ID=your_aws_key
AWS_SECRET_ACCESS_KEY=your_aws_secret
```

### 3. Run with Docker
```bash
docker-compose up --build
```
To use a local model from Docker, run Ollama on the host; containers reach it at `host.docker.internal:11434`.

### 4. Or run locally
```bash
pip install -r requirements-dev.txt
python src/ingest.py        # download filings + build the FAISS index
streamlit run src/app.py    # launch dashboard
pytest                      # offline unit tests
```

---

## Dashboard

1. **Signal Generator**: ask a question about one company's latest filing
2. **YoY Trend**: risk and sentiment across filing years
3. **Portfolio Dashboard**: sector-level heatmap and summary

---

## Author

**Dhruvi Gandhi**  
MS Applied Data Science, University of Chicago  
[GitHub](https://github.com/dhruvig1)
