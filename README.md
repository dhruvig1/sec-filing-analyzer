# SEC Filing Analyzer

Who does this help?

A retrieval-augmented pipeline that reads SEC 10-K filings and answers analyst questions about a company's risks, with the source passages attached. It covers 40 large US companies across six sectors, runs with either a hosted model (gpt-oss-120b on Groq) or a local one (Qwen3-8B through Ollama), and comes with an evaluation suite that measures retrieval and generation quality instead of assuming them.


## What it does

The Streamlit app has three views:

- **Signal Generator.** Pick a company and ask a question ("How do export controls affect data center sales to China?"). The app retrieves the most relevant passages from that company's latest 10-K and returns a risk level, three specific risks, a one-sentence rationale, red flags quoted from the filing, and the list of passages it used. If the filing doesn't address the question, it says so instead of making something up.
- **YoY Trend.** The same analysis for each of a company's last three filings, so you can see how its stated risks change over time.
- **Portfolio Dashboard.** Several companies side by side, with a risk heatmap and a per-company breakdown.

## How it works

```
SEC EDGAR 10-Ks ──> section extraction ──> chunking ──> FAISS (dense) + BM25 (keyword)
                    (Item 1A, Item 7)      18,048 chunks            │
                                                                    │  filtered to one company
Analyst question + ticker ─────────────────────────────────────────>│  and one filing year
                                                                    ▼
                                                  hybrid retrieval (reciprocal rank fusion)
                                                                    │
                                                                    ▼
                                         LLM (Groq or local Ollama), schema-validated JSON
                                                                    │
                                                                    ▼
                                         risk level, key risks, red flags, sources
```

**Ingestion.** An Airflow DAG pulls 10-Ks from EDGAR weekly. Each `full-submission.txt` bundles the 10-K with exhibits and XBRL, so the extractor keeps only the 10-K (and the EX-13 annual report, where some companies put their MD&A), strips the HTML, and finds Item 1A (Risk Factors) and Item 7 (MD&A). Real filings make this harder than it sounds: tables of contents, cross-references ("see Item 1A"), running page headers, banks that skip "Item" headings entirely, and companies whose 10-K just points to the annual report. The extractor handles all of these and finds 244 of 244 sections across 122 filings.

**Retrieval.** Every search is restricted to one company and one filing year. There is deliberately no fallback to other years, because borrowing a later filing would leak future information into a point-in-time answer. Dense search uses all-MiniLM-L6-v2 embeddings in FAISS; keyword search uses BM25; the two rankings are merged with reciprocal rank fusion.

**Generation.** The LLM gets the retrieved passages and must return a JSON object matching a fixed schema. The local backend uses schema-constrained decoding, so the model can't produce anything else. Every output is validated, and an invalid one gets a single repair attempt with the specific errors before the app shows an error instead.

## Results

### Retrieval

I wrote 30 company-specific questions (five per sector) before looking at any results. For each question, the top five passages from every retriever were pooled and graded by an LLM judge: 2 = directly answers the question, 1 = on topic but partial, 0 = not relevant.

| Retriever | P@5 | Strict P@5 | MRR |
|---|---|---|---|
| Dense, MiniLM | 0.813 | 0.427 | 0.612 |
| BM25 | 0.813 | 0.373 | 0.625 |
| **Hybrid, MiniLM (used in the app)** | **0.893** | 0.433 | 0.681 |
| Dense, Qwen3-Embedding-0.6B | 0.840 | 0.447 | 0.756 |
| Hybrid, Qwen3-Embedding-0.6B | 0.880 | 0.447 | 0.706 |

P@5 counts passages graded 1 or 2; strict P@5 counts only 2s; MRR uses the first passage graded 2. Both retrievers see the same 30 questions, so I compared them with paired bootstrap tests rather than eyeballing the averages:

- Hybrid beats dense on P@5 by +0.080 (95% CI +0.027 to +0.140; better on 10 questions, worse on 2). This is why the app uses hybrid search.
- Qwen3-Embedding, a model 27 times larger than MiniLM, improves dense MRR by +0.143 (CI +0.027 to +0.260): it ranks the best passage higher.
- Hybrid with Qwen is not measurably better than hybrid with MiniLM on any metric. Since the app passes all five passages to the LLM, P@5 matters more than MRR, so I kept the much smaller model.
- Strict P@5 sits around 0.43 to 0.45 for every method. None of these changes increased how many passages directly answer the question; that is the next thing to work on.

### Hosted vs local generation

same prompt and same retrieved context for both models:

| | gpt-oss-120b (Groq) | Qwen3-8B (Ollama, laptop) |
|---|---|---|
| Valid against schema | 100% | 100% |
| Faithfulness, LLM judge (1 to 5) | 4.8 | 4.8 |
| Similarity of risks to filing text | 0.81 | 0.93 |
| Red flags quoted verbatim | 95% | 93% |
| Risk level matches hosted model | | 90% |
| Median latency | 1.3 s | 62 s |
| Cost per 1,000 signals | about $0.36 | $0 |

The local 8B model was as faithful as the hosted 120B model, with the filing text never leaving the machine; the cost is speed. 

### Checking the judge

An LLM judge is only useful if it agrees with people, so I graded 60 passages myself, blind to the judge's grades, in two rounds of 30. Agreement on whether a passage was relevant at all was 70% in both rounds, but Cohen's kappa on the full 0 to 2 scale was only 0.35 and 0.25. My own grading also shifted between rounds, which says the three-point scale is too fuzzy. Because of this, I treat the retrieval numbers as reliable for comparing retrievers (the same judge grades all of them on the same passages) rather than as absolute accuracy.

## What didn't work

- **Sentiment.** Both models gave nearly the same sentiment label for every filing (the hosted model said BEARISH 10 out of 10 times, the local one NEUTRAL 10 out of 10), because risk-factor sections read as alarming no matter the company. The field is still in the raw JSON but the app no longer shows it.
- **A bigger embedding model.** Covered above: real gains in ranking, none in what the app actually uses.

## Limitations

- The risk level is HIGH for most large companies, so it says little on its own; the specific risks and their sources are the useful part.
- The app answers from one filing at a time and has no notion of risks that changed between filings beyond the trend view.

## Running it

Requires Python 3.11 or later.

```bash
git clone https://github.com/dhruvig1/sec-filing-analyzer.git
cd sec-filing-analyzer
pip install -r requirements.txt
```

Create a `.env` file:

```
GROQ_API_KEY=your_key
SEC_USER_AGENT_NAME=Your Name
SEC_USER_AGENT_EMAIL=you@example.com
```

Then:

```bash
python src/ingest.py              # download filings, extract sections, build the index
streamlit run src/app.py          # open the app
```

For the local backend, install [Ollama](https://ollama.com), run `ollama pull qwen3:8b`, and it appears in the app's sidebar while `ollama serve` is running.

Evaluations:

```bash
python src/eval_passages.py run                          # retrieval benchmark
python src/eval_passages.py audit --n 30                 # sample passages to grade by hand
python src/compare_backends.py --backends groq ollama:qwen3:8b --max-cases 10 --judge
EMBEDDING_MODEL=Qwen/Qwen3-Embedding-0.6B python src/eval_passages.py run   # after building that index
pytest -q tests                                          # offline tests, no API calls
```

## Project layout

```
src/
  app.py               Streamlit app
  ingest.py            EDGAR download, section extraction, chunking, indexing
  text_utils.py        10-K cleaning and section finding
  retrieval.py         dense, BM25 and hybrid search
  rag.py               context building, prompting, output validation
  llm.py               hosted (Groq) and local (Ollama) backends with latency and cost tracking
  embeddings.py        embedding models, including Qwen's query instructions
  eval_passages.py     passage-level retrieval benchmark and judge audit
  compare_backends.py  hosted vs local generation comparison
  remote_embed.py      embed on a remote GPU, build the index locally
  config.py            paths, companies, settings
dags/                  Airflow DAG for scheduled ingestion
tests/                 offline tests with fake embeddings and a mocked LLM
```

## Stack

Python, FAISS, rank-bm25, sentence-transformers, LangChain (vector store and text splitting), Groq, Ollama, Streamlit, Plotly, Apache Airflow, Docker, pytest.