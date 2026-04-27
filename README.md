# SEC Filing Analyzer 📊

An LLM-powered RAG pipeline for analyzing SEC filings — featuring semantic search, year-over-year trend analysis, and an interactive portfolio dashboard.

---

## Overview

This project ingests SEC EDGAR filings for 40 companies across 6 sectors, builds a vector store for semantic retrieval, and serves insights through a multi-tab Streamlit dashboard powered by Llama-3.3-70B via Groq API.

---

## Features

- **RAG Pipeline** — Retrieval-Augmented Generation over 120+ SEC filings (3,000+ chunks)
- **Semantic Search** — BERT embeddings (`all-MiniLM-L6-v2`) + FAISS vector store
- **Year-over-Year Trend Analysis** — Compare financial metrics across filing periods
- **Portfolio Dashboard** — 3-tab Streamlit UI for interactive exploration
- **Automated Ingestion** — Apache Airflow DAG for scheduled SEC EDGAR data pulls
- **Containerized** — Fully Dockerized for reproducible deployment

---

## Tech Stack

| Layer | Tools |
|---|---|
| LLM | Llama-3.3-70B via Groq API |
| Embeddings | `all-MiniLM-L6-v2` (BERT) |
| Vector Store | FAISS |
| Chunking & Retrieval | LangChain |
| Storage | AWS S3 |
| Orchestration | Apache Airflow |
| Frontend | Streamlit |
| Containerization | Docker |

---

## Project Structure

```
sec-filing-analyzer/
├── src/
│   ├── app.py          # Streamlit dashboard (3 tabs)
│   ├── ingest.py       # SEC EDGAR ingestion + S3 upload
│   └── rag.py          # RAG pipeline (embeddings, FAISS, LLM)
├── dags/
│   └── sec_dag.py      # Airflow DAG for scheduled ingestion
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## Data

- **Companies**: 40 across 6 sectors
- **Filings**: 120+ SEC filings
- **Chunks**: 3,000+ text chunks indexed in FAISS

---

## Evaluation Metrics

| Metric | Score |
|---|---|
| Backtesting Accuracy | 66.7% |
| Recall@5 | 0.70 |
| MRR | 0.70 |
| NDCG@5 | 0.869 |

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
GROQ_API_KEY=your_groq_api_key
AWS_ACCESS_KEY_ID=your_aws_key
AWS_SECRET_ACCESS_KEY=your_aws_secret
S3_BUCKET_NAME=your_bucket_name
```

### 3. Run with Docker
```bash
docker-compose up --build
```

### 4. Or run locally
```bash
pip install -r requirements.txt
python src/ingest.py       # ingest filings
streamlit run src/app.py   # launch dashboard
```

---

## Dashboard

The Streamlit app has 3 tabs:
1. **Search** — Ask natural language questions over SEC filings
2. **Trends** — Year-over-year financial metric comparisons
3. **Portfolio** — Cross-company sector-level analysis

---

## Author

**Dhruvi Gandhi**  
MS Applied Data Science, University of Chicago  
[GitHub](https://github.com/dhruvig1)
