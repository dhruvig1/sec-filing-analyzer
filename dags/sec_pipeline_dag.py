# dags/sec_pipeline_dag.py
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.email import EmailOperator
from datetime import datetime, timedelta
import sys, os

sys.path.insert(0, '/opt/airflow/src')

default_args = {
    "owner":          "dhruvi",
    "retries":        2,
    "retry_delay":    timedelta(minutes=5),
    "start_date":     datetime(2025, 1, 1),
}

def download_filings_task():
    from ingest import download_filings
    download_filings()

def build_embeddings_task():
    from ingest import load_and_chunk_filings, build_vector_store
    docs = load_and_chunk_filings()
    if docs:
        build_vector_store(docs)
    print(f"Built vector store with {len(docs)} sections")

def generate_signals_task():
    from rag import generate_signal
    tickers = ["JPM", "GS", "AAPL", "MSFT", "XOM"]
    signals = []
    for ticker in tickers:
        signal = generate_signal(ticker)
        signals.append(signal)
        print(f"{ticker}: {signal.get('risk_level')} | {signal.get('sentiment')}")
    
    # flag high risk tickers
    high_risk = [s["ticker"] for s in signals if s.get("risk_level") == "HIGH"]
    if high_risk:
        print(f"HIGH RISK DETECTED: {high_risk}")
    return signals

def sync_to_s3_task():
    import boto3, glob
    s3       = boto3.client("s3")
    BUCKET   = "sec-filings-dhruvig"
    data_dir = "/opt/airflow/data/sec-edgar-filings"
    
    for filepath in glob.glob(f"{data_dir}/**/*.txt", recursive=True):
        s3_key = filepath.replace(data_dir + "/", "")
        s3.upload_file(filepath, BUCKET, f"sec-edgar-filings/{s3_key}")
    print("Synced filings to S3")

with DAG(
    dag_id="sec_filing_pipeline",
    default_args=default_args,
    schedule_interval="0 9 * * 1",  # every Monday 9am
    catchup=False,
    description="Weekly SEC filing ingestion, embedding, and signal generation"
) as dag:

    download = PythonOperator(
        task_id="download_filings",
        python_callable=download_filings_task
    )

    embed = PythonOperator(
        task_id="build_embeddings",
        python_callable=build_embeddings_task
    )

    signals = PythonOperator(
        task_id="generate_signals",
        python_callable=generate_signals_task
    )

    s3_sync = PythonOperator(
        task_id="sync_to_s3",
        python_callable=sync_to_s3_task
    )

    # pipeline order
    download >> embed >> signals >> s3_sync