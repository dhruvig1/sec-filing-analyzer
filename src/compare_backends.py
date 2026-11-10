# src/compare_backends.py
"""Compare a local quantized model (Ollama) against a hosted model (Groq, default gpt-oss-120b) on the same filings.

Each backend gets the identical prompt and context for every (ticker, year), so
differences come from the model alone. For each output we record:

    json_valid      parsed as JSON
    schema_valid    correct keys, enum values, exactly 3 key_risks
    grounding       mean over key_risks of the max cosine similarity to any
                    sentence in the context (local MiniLM embeddings); a cheap,
                    reference-free proxy for faithfulness
    grounded_frac   share of key_risks with similarity >= --ground-threshold
    redflag_verbatim share of red_flags that appear verbatim in the context
                    (the prompt asks for verbatim phrases, so this checks it)
    judge_faithful  optional 1-5 LLM-as-judge score (--judge)
    latency_s, input/output tokens, cost_usd

and, across backends, how often they agree on risk_level and sentiment.

Usage (start Ollama first: `ollama serve` and `ollama pull llama3.1:8b`):

    python src/compare_backends.py --backends groq:openai/gpt-oss-120b ollama:llama3.1:8b
    python src/compare_backends.py --backends groq ollama:qwen2.5:7b --max-cases 10 --judge
"""
import argparse
import csv
import json
import os
import re
import sys
from collections import defaultdict
from statistics import mean, median

sys.path.append(os.path.dirname(__file__))

from config import RESULTS_DIR  # noqa: E402
from llm import complete, get_backend, get_model  # noqa: E402
from rag import DEFAULT_QUERY, build_context, run_llm, validate_signal  # noqa: E402

JUDGE_PROMPT = """You are grading whether an analyst's extracted risks are supported by a 10-K excerpt.

Excerpt:
{context}

Extracted output:
{output}

Score faithfulness from 1 to 5:
5 = every key risk and the rationale are directly supported by the excerpt
3 = mostly supported, some details are not in the excerpt
1 = largely unsupported or invented

Respond with JSON: {{"score": <int 1-5>, "unsupported": [<claims not found in the excerpt>]}}"""


def parse_backend_spec(spec):
    """'ollama:llama3.1:8b' -> ('ollama', 'llama3.1:8b'); 'groq' -> ('groq', default model)."""
    backend, _, model = spec.partition(":")
    backend = get_backend(backend)
    return backend, get_model(backend, model or None)


def split_sentences(text, min_len=30):
    pieces = re.split(r"(?<=[.!?;])\s+|\n+", text)
    return [p.strip() for p in pieces if len(p.strip()) >= min_len]


def _cosine_matrix(a, b):
    import numpy as np

    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    a = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
    b = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
    return a @ b.T


def grounding_scores(claims, context, embed_fn, threshold=0.5):
    """(mean max-similarity, fraction above threshold) of claims vs. context sentences."""
    sentences = split_sentences(context)
    claims = [c for c in claims if isinstance(c, str) and c.strip()]
    if not claims or not sentences:
        return 0.0, 0.0
    best = _cosine_matrix(embed_fn(claims), embed_fn(sentences)).max(axis=1)
    return float(best.mean()), float((best >= threshold).mean())


def _norm(s):
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s%$.]", " ", s.lower())).strip()


def redflag_verbatim_rate(flags, context):
    flags = [f for f in flags or [] if isinstance(f, str) and f.strip()]
    if not flags:
        return None
    ctx = _norm(context)
    return sum(_norm(f) in ctx for f in flags) / len(flags)


def judge(context, signal, backend, model):
    output = json.dumps({k: signal.get(k) for k in ("key_risks", "rationale", "red_flags")}, indent=2)
    res = complete([{"role": "user", "content": JUDGE_PROMPT.format(context=context, output=output)}],
                   backend=backend, model=model, temperature=0.0, max_tokens=300)
    try:
        return int(json.loads(res.text[res.text.index("{"): res.text.rindex("}") + 1])["score"])
    except (ValueError, KeyError, TypeError, json.JSONDecodeError):
        return None


def evaluate_output(signal, context, embed_fn, threshold):
    valid_json = "error" not in signal
    errors = validate_signal(signal) if valid_json else ["unparseable"]
    g_mean, g_frac = grounding_scores(signal.get("key_risks") or [], context, embed_fn, threshold) if valid_json else (0.0, 0.0)
    return {
        "json_valid": valid_json,
        "schema_valid": not errors,
        "schema_errors": "; ".join(errors),
        "grounding": round(g_mean, 3),
        "grounded_frac": round(g_frac, 3),
        "redflag_verbatim": redflag_verbatim_rate(signal.get("red_flags"), context) if valid_json else None,
    }


def default_cases(tickers=None):
    from evaluate import FILING_DATES

    return [(t, y) for t, years in FILING_DATES.items() if not tickers or t in tickers for y in years]


def run_comparison(cases, backends, query=DEFAULT_QUERY, context_mode="hybrid", embed_fn=None,
                   threshold=0.5, judge_spec=None, context_fn=build_context, llm_fn=run_llm):
    rows = []
    for ticker, year in cases:
        context, _ = context_fn(ticker, year, query, context_mode=context_mode)
        if not context:
            print(f"{ticker} {year}: no filing found, skipping")
            continue
        for backend, model in backends:
            try:
                signal, result = llm_fn(ticker, year, query, context, backend=backend, model=model)
            except Exception as e:
                print(f"{ticker} {year} [{backend}:{model}] failed: {e}")
                rows.append({"ticker": ticker, "year": year, "backend": backend, "model": model,
                             "json_valid": False, "schema_valid": False, "schema_errors": f"call failed: {e}"})
                continue
            row = {"ticker": ticker, "year": year, "backend": backend, "model": model,
                   "risk_level": signal.get("risk_level"), "sentiment": signal.get("sentiment"),
                   "latency_s": round(result.latency_s, 2), "input_tokens": result.input_tokens,
                   "output_tokens": result.output_tokens, "cost_usd": result.cost_usd}
            row.update(evaluate_output(signal, context, embed_fn, threshold))
            if judge_spec and row["json_valid"]:
                row["judge_faithful"] = judge(context, signal, *judge_spec)
            rows.append(row)
            print(f"{ticker} {year} [{backend}:{model}] {row['risk_level']}/{row['sentiment']} "
                  f"schema={row['schema_valid']} grounding={row['grounding']} {row['latency_s']}s")
    return rows


def summarize(rows):
    by_model = defaultdict(list)
    for r in rows:
        by_model[f"{r['backend']}:{r['model']}"].append(r)

    def avg(rs, key):
        vals = [r[key] for r in rs if r.get(key) is not None]
        return round(mean(vals), 3) if vals else None

    summary = {}
    for name, rs in by_model.items():
        lat = [r["latency_s"] for r in rs if r.get("latency_s") is not None]
        summary[name] = {
            "n": len(rs),
            "json_valid_pct": round(100 * mean(r["json_valid"] for r in rs), 1),
            "schema_valid_pct": round(100 * mean(r["schema_valid"] for r in rs), 1),
            "grounding": avg(rs, "grounding"),
            "grounded_frac": avg(rs, "grounded_frac"),
            "redflag_verbatim": avg(rs, "redflag_verbatim"),
            "judge_faithful": avg(rs, "judge_faithful"),
            "median_latency_s": round(median(lat), 2) if lat else None,
            "total_cost_usd": round(sum(r.get("cost_usd") or 0 for r in rs), 5),
        }

    # Agreement between each model and the first one listed (the reference).
    names = list(by_model)
    if len(names) > 1:
        ref = {(r["ticker"], r["year"]): r for r in by_model[names[0]] if r.get("schema_valid")}
        for name in names[1:]:
            pairs = [(ref[(r["ticker"], r["year"])], r) for r in by_model[name]
                     if r.get("schema_valid") and (r["ticker"], r["year"]) in ref]
            if pairs:
                summary[name]["risk_agreement_vs_ref"] = round(100 * mean(a["risk_level"] == b["risk_level"] for a, b in pairs), 1)
                summary[name]["sentiment_agreement_vs_ref"] = round(100 * mean(a["sentiment"] == b["sentiment"] for a, b in pairs), 1)
    return summary


def to_markdown(summary):
    cols = ["n", "json_valid_pct", "schema_valid_pct", "grounding", "grounded_frac", "redflag_verbatim",
            "judge_faithful", "risk_agreement_vs_ref", "sentiment_agreement_vs_ref", "median_latency_s", "total_cost_usd"]
    lines = ["| model | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
    for name, s in summary.items():
        lines.append(f"| {name} | " + " | ".join("—" if s.get(c) is None else str(s[c]) for c in cols) + " |")
    return "\n".join(lines)


def local_embedder():
    from langchain_huggingface import HuggingFaceEmbeddings
    from config import EMBEDDING_MODEL

    emb = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return emb.embed_documents


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--backends", nargs="+", default=["groq", "ollama"],
                   help="backend[:model] specs; the first is the reference for agreement")
    p.add_argument("--tickers", nargs="*", default=None)
    p.add_argument("--max-cases", type=int, default=None)
    p.add_argument("--context-mode", choices=["hybrid", "retrieval", "section"], default="hybrid")
    p.add_argument("--query", default=DEFAULT_QUERY)
    p.add_argument("--ground-threshold", type=float, default=0.5)
    p.add_argument("--judge", action="store_true",
                   help="add an LLM-as-judge faithfulness score (default judge: the Groq model; "
                        "note it may favour its own outputs)")
    p.add_argument("--judge-backend", default="groq")
    args = p.parse_args()

    backends = [parse_backend_spec(s) for s in args.backends]
    cases = default_cases(args.tickers)[: args.max_cases]
    judge_spec = parse_backend_spec(args.judge_backend) if args.judge else None

    rows = run_comparison(cases, backends, query=args.query, context_mode=args.context_mode,
                          embed_fn=local_embedder(), threshold=args.ground_threshold, judge_spec=judge_spec)
    summary = summarize(rows)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    fields = sorted({k for r in rows for k in r})
    with open(RESULTS_DIR / "backend_comparison.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    (RESULTS_DIR / "backend_comparison_summary.json").write_text(json.dumps(summary, indent=2))
    md = to_markdown(summary)
    (RESULTS_DIR / "backend_comparison.md").write_text(md + "\n")
    print("\n" + md)
    print(f"\nSaved per-filing results and summary to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
