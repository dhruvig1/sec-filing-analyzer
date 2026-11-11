# src/eval_passages.py
"""Passage-level retrieval evaluation, scored the way the app actually retrieves.

The older retrieval eval (evaluate.py retrieval) asks "did the right *companies*
come back for an unfiltered query?", using hand-picked company lists. With real
section text, most companies discuss most topics, so correct passages from
unlisted companies were scored as misses. The app never searches that way: it
filters to one company and one filing year, then asks a specific question.

This eval matches the app:

1. Company-specific questions (QUESTIONS below), written before looking at results.
2. Each retriever returns its top-k chunks for that company's latest filing.
   Retrievers: dense (FAISS, what the app uses), bm25 (keyword baseline),
   hybrid (reciprocal-rank fusion of the two).
3. The top-k from every retriever is pooled and each pooled passage is graded
   once by an LLM judge: 2 = directly answers, 1 = on topic / partial, 0 = not relevant.
   Grades are cached in results/judgments.jsonl, so reruns are free.
4. Metrics per retriever, averaged over questions, with 95% bootstrap CIs:
     P@k        share of top-k graded >= 1
     strict P@k share of top-k graded 2
     MRR        1 / rank of the first passage graded 2
     nDCG@k     gains 2^grade - 1; the ideal ranking comes from the pooled grades
5. Check the judge: `audit` samples passages for you to grade by hand,
   `agreement` reports judge-vs-human agreement and Cohen's kappa.

    python src/eval_passages.py run                    # all questions, all retrievers
    python src/eval_passages.py run --max-questions 5  # quick smoke test
    python src/eval_passages.py audit --n 30           # writes results/judge_audit.csv
    python src/eval_passages.py agreement              # after filling in human_grade
"""
import argparse
import csv
import json
import math
import os
import random
import sys
import time
from collections import Counter, defaultdict
from statistics import mean

sys.path.append(os.path.dirname(__file__))

from config import EMBEDDING_MODEL, RESULTS_DIR  # noqa: E402

K = 5
PASSAGE_CHARS = 1500     # chunks are ~1000 chars; cap what the judge sees
JUDGMENTS_FILE = "judgments.jsonl"
AUDIT_FILE = "judge_audit.csv"

# ── QUESTIONS ───────────────────────────────────────────────────
# Five per sector, phrased the way an analyst would ask about one company.
# Written before looking at retrieval output; don't tune them to the results.
QUESTIONS = [
    # Finance & Banking
    ("JPM",   "What drove changes in the provision and allowance for credit losses?"),
    ("GS",    "How do market volatility and client activity affect market-making revenues?"),
    ("BAC",   "How sensitive is net interest income to changes in interest rates?"),
    ("WFC",   "What regulatory consent orders or restrictions does the company operate under?"),
    ("AXP",   "What are the risks from Card Member loan credit losses and delinquencies?"),
    # Tech
    ("AAPL",  "What risks come from reliance on outsourcing partners and suppliers in Asia?"),
    ("MSFT",  "How is the company investing in AI infrastructure and datacenter capacity?"),
    ("NVDA",  "How do export controls affect sales of data center products to China?"),
    ("GOOGL", "What antitrust lawsuits and regulatory investigations does the company face?"),
    ("META",  "How could privacy and data protection regulation in Europe affect the business?"),
    # Insurance
    ("PGR",   "How do catastrophe losses and severe weather affect underwriting results?"),
    ("TRV",   "What drove catastrophe losses and prior-year reserve development?"),
    ("AIG",   "What risks arise from uncertainty in loss reserve estimates?"),
    ("MET",   "How do interest rate changes affect the investment portfolio and product margins?"),
    ("ALL",   "How does the company respond to rising auto insurance loss costs with rate increases?"),
    # Healthcare
    ("PFE",   "Which products face loss of exclusivity or patent expirations, and what is the revenue impact?"),
    ("LLY",   "What are the risks around manufacturing capacity and supply for key medicines?"),
    ("UNH",   "What drove changes in medical costs and the medical care ratio?"),
    ("ABBV",  "How is biosimilar competition affecting Humira revenue?"),
    ("JNJ",   "What product liability litigation does the company face?"),
    # Energy
    ("XOM",   "How do changes in oil and natural gas prices affect earnings?"),
    ("CVX",   "What climate change regulations and energy transition policies could affect the business?"),
    ("COP",   "What are the risks related to estimates of proved oil and gas reserves?"),
    ("SLB",   "How does customer capital spending affect demand for the company's services?"),
    ("PSX",   "How do refining margins and crack spreads affect results?"),
    # Consumer
    ("MCD",   "What risks come from dependence on franchisees?"),
    ("SBUX",  "What drove changes in comparable store sales?"),
    ("NKE",   "How is the company managing inventory levels and its wholesale and direct channels?"),
    ("WMT",   "How could tariffs and trade policy affect costs and merchandise sourcing?"),
    ("COST",  "How important are membership fees to the company's profitability?"),
]

JUDGE_PROMPT = """You are grading search results for an analyst reading {ticker}'s 10-K filed in {year}.

Question: {question}

Passage:
\"\"\"{passage}\"\"\"

Grade how useful this passage is for answering the question:
2 = directly addresses the question with specific information
1 = on the topic, but only partially or generally helpful
0 = not relevant: a different topic, boilerplate, a table of contents, or only a passing mention

Respond with JSON: {{"grade": <0, 1 or 2>, "reason": "<one short sentence>"}}"""


# ── RETRIEVERS (shared with the app, see retrieval.py) ──────────
from retrieval import (_bm25_cache, _company_cache, bm25_ranking, chunk_key, company_docs,  # noqa: E402,F401
                       dense_ranking, hybrid_ranking, latest_year, rrf_fuse, tokenize)

RETRIEVERS = {"dense": dense_ranking, "bm25": bm25_ranking, "hybrid": hybrid_ranking}


# ── JUDGE ───────────────────────────────────────────────────────
def load_judgments(path):
    cache = {}
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                cache[row["key"]] = row
    return cache


def judge_passage(question, ticker, year, passage, backend=None, model=None, retries=4):
    """Return (grade, reason). Retries with backoff on API errors or unparseable output."""
    from llm import complete

    prompt = JUDGE_PROMPT.format(ticker=ticker, year=year, question=question, passage=passage[:PASSAGE_CHARS])
    last = None
    for attempt in range(retries):
        try:
            res = complete([{"role": "user", "content": prompt}], backend=backend, model=model,
                           temperature=0.0, max_tokens=200, json_mode=True)
            data = json.loads(res.text[res.text.index("{"): res.text.rindex("}") + 1])
            grade = int(data["grade"])
            if grade in (0, 1, 2):
                return grade, str(data.get("reason", ""))[:300]
            last = f"grade out of range: {grade}"
        except Exception as e:  # rate limits, timeouts, bad JSON
            last = str(e)
        time.sleep(min(2 ** attempt, 20))
    raise RuntimeError(f"judge failed after {retries} attempts: {last}")


# ── METRICS ─────────────────────────────────────────────────────
def precision_at_k(grades, k=K, min_grade=1):
    return sum(g >= min_grade for g in grades[:k]) / k


def reciprocal_rank(grades, min_grade=2):
    for i, g in enumerate(grades, 1):
        if g >= min_grade:
            return 1.0 / i
    return 0.0


def dcg(grades, k=K):
    return sum((2 ** g - 1) / math.log2(i + 1) for i, g in enumerate(grades[:k], 1))


def ndcg_at_k(grades, pooled_grades, k=K):
    ideal = dcg(sorted(pooled_grades, reverse=True), k)
    return dcg(grades, k) / ideal if ideal > 0 else 0.0


def bootstrap_ci(values, n=2000, seed=0):
    """95% percentile bootstrap CI of the mean, resampling questions."""
    if len(values) < 2:
        return (None, None)
    rng = random.Random(seed)
    means = sorted(mean(rng.choices(values, k=len(values))) for _ in range(n))
    return round(means[int(0.025 * n)], 3), round(means[int(0.975 * n) - 1], 3)


def cohen_kappa(a, b):
    labels = sorted(set(a) | set(b))
    n = len(a)
    if n == 0:
        return None
    observed = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    expected = sum(ca[l] * cb[l] for l in labels) / (n * n)
    return 1.0 if expected == 1 else (observed - expected) / (1 - expected)


# ── RUN ─────────────────────────────────────────────────────────
def run_eval(vs, questions, retrievers=("dense", "bm25", "hybrid"), k=K, judge_fn=None,
             judgments_path=None, judge_label="judge"):
    """Returns (per_question_rows, summary). judge_fn(question, ticker, year, passage) -> (grade, reason)."""
    judgments_path = judgments_path or RESULTS_DIR / JUDGMENTS_FILE
    judgments_path.parent.mkdir(parents=True, exist_ok=True)
    cache = load_judgments(judgments_path)
    rows = []

    with open(judgments_path, "a") as log:
        for qid, (ticker, question) in enumerate(questions, 1):
            year = latest_year(vs, ticker)
            if year is None:
                print(f"Q{qid} {ticker}: not in index, skipping")
                continue
            ranked = {name: RETRIEVERS[name](vs, ticker, year, question)[:k] for name in retrievers}

            grades = {}
            for doc in {chunk_key(d): d for docs in ranked.values() for d in docs}.values():
                key = f"{judge_label}|{ticker}|{year}|{question}|{chunk_key(doc)}"
                if key not in cache:
                    grade, reason = judge_fn(question, ticker, year, doc.page_content)
                    cache[key] = {"key": key, "judge": judge_label, "ticker": ticker, "year": year,
                                  "question": question, "chunk_id": chunk_key(doc),
                                  "section": doc.metadata.get("section"), "grade": grade, "reason": reason,
                                  "passage": doc.page_content[:PASSAGE_CHARS]}
                    log.write(json.dumps(cache[key]) + "\n")
                    log.flush()
                grades[chunk_key(doc)] = cache[key]["grade"]

            pooled = list(grades.values())
            line = []
            for name, docs in ranked.items():
                g = [grades[chunk_key(d)] for d in docs]
                g += [0] * (k - len(g))  # fewer than k chunks counts as misses
                row = {"qid": qid, "ticker": ticker, "year": year, "question": question, "retriever": name,
                       "grades": g, "p_at_k": precision_at_k(g, k), "strict_p_at_k": precision_at_k(g, k, 2),
                       "mrr": reciprocal_rank(g), "ndcg_at_k": ndcg_at_k(g, pooled, k)}
                rows.append(row)
                line.append(f"{name}={g}")
            print(f"Q{qid:>2} {ticker:<5} {year} | " + "  ".join(line))

    return rows, summarize(rows, k)


def summarize(rows, k=K):
    by = defaultdict(list)
    for r in rows:
        by[r["retriever"]].append(r)
    summary = {}
    for name, rs in by.items():
        s = {"n_questions": len(rs)}
        for metric in ("p_at_k", "strict_p_at_k", "mrr", "ndcg_at_k"):
            vals = [r[metric] for r in rs]
            s[metric] = round(mean(vals), 3)
            s[f"{metric}_ci95"] = bootstrap_ci(vals)
        summary[name] = s
    return summary


def results_suffix(model=EMBEDDING_MODEL):
    """'' for the default MiniLM index (keeps the original file names), '__<model>' otherwise."""
    from config import DEFAULT_EMBEDDING_MODEL
    return "" if model == DEFAULT_EMBEDDING_MODEL else "__" + model.split("/")[-1].lower().replace(".", "_")


def to_markdown(summary, k=K, model=EMBEDDING_MODEL):
    head = (f"Embedding model (dense / hybrid): `{model}`\n\n"
            f"| retriever | n | P@{k} | strict P@{k} | MRR | nDCG@{k} |\n|---|---|---|---|---|---|")
    lines = [head]
    for name, s in summary.items():
        cell = lambda m: f"{s[m]:.3f} ({s[m + '_ci95'][0]}–{s[m + '_ci95'][1]})" if s[m + "_ci95"][0] is not None else f"{s[m]:.3f}"
        lines.append(f"| {name} | {s['n_questions']} | {cell('p_at_k')} | {cell('strict_p_at_k')} | "
                     f"{cell('mrr')} | {cell('ndcg_at_k')} |")
    return "\n".join(lines) + "\n\n95% bootstrap CIs over questions in parentheses."


def write_results(rows, summary, k=K, model=EMBEDDING_MODEL):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    sfx = results_suffix(model)
    with open(RESULTS_DIR / f"passage_eval_per_question{sfx}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (RESULTS_DIR / f"passage_eval_summary{sfx}.json").write_text(
        json.dumps({"embedding_model": model, "retrievers": summary}, indent=2))
    md = to_markdown(summary, k, model)
    (RESULTS_DIR / f"passage_eval{sfx}.md").write_text(md + "\n")
    return md


# ── JUDGE AUDIT ─────────────────────────────────────────────────
def write_audit(n=30, seed=0, judgments_path=None, audit_path=None, exclude_paths=()):
    """Sample judged passages (balanced across grades) for a human to grade blind.

    exclude_paths: earlier audit CSVs whose passages must not be sampled again, so a
    second round tests the judge on passages nobody has looked at yet.
    """
    judgments_path = judgments_path or RESULTS_DIR / JUDGMENTS_FILE
    audit_path = audit_path or RESULTS_DIR / AUDIT_FILE
    seen = set()
    for ep in exclude_paths:
        with open(ep, newline="") as f:
            seen |= {r["key"] for r in csv.DictReader(f)}
    rows = [r for r in load_judgments(judgments_path).values() if r["key"] not in seen]
    rng = random.Random(seed)
    by_grade = defaultdict(list)
    for r in rows:
        by_grade[r["grade"]].append(r)
    for g in by_grade:
        rng.shuffle(by_grade[g])
    sample = []
    while len(sample) < min(n, len(rows)):
        for g in sorted(by_grade):
            if by_grade[g] and len(sample) < n:
                sample.append(by_grade[g].pop())
    rng.shuffle(sample)
    with open(audit_path, "w", newline="") as f:
        w = csv.writer(f)
        # Blind: the judge's grade is not in this file; agreement looks it up by key.
        w.writerow(["key", "ticker", "question", "passage", "human_grade"])
        for r in sample:
            w.writerow([r["key"], r["ticker"], r["question"], r["passage"], ""])
    return audit_path, len(sample)


def audit_agreement(audit_path=None, judgments_path=None):
    audit_path = audit_path or RESULTS_DIR / AUDIT_FILE
    cache = load_judgments(judgments_path or RESULTS_DIR / JUDGMENTS_FILE)
    with open(audit_path, newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["human_grade"].strip() != "" and r["key"] in cache]
    if not rows:
        raise SystemExit(f"No human_grade values filled in {audit_path}")
    human = [int(r["human_grade"]) for r in rows]
    judge = [cache[r["key"]]["grade"] for r in rows]
    hb, jb = [h >= 1 for h in human], [j >= 1 for j in judge]
    return {
        "n": len(rows),
        "exact_agreement": round(mean(h == j for h, j in zip(human, judge)), 3),
        "binary_agreement": round(mean(h == j for h, j in zip(hb, jb)), 3),
        "cohen_kappa": round(cohen_kappa(human, judge), 3),
        "cohen_kappa_binary": round(cohen_kappa(hb, jb), 3),
        "judge_harsher": sum(j < h for h, j in zip(human, judge)),
        "judge_more_lenient": sum(j > h for h, j in zip(human, judge)),
    }


# ── CLI ─────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--retrievers", nargs="+", default=list(RETRIEVERS), choices=list(RETRIEVERS))
    r.add_argument("--k", type=int, default=K)
    r.add_argument("--max-questions", type=int, default=None)
    r.add_argument("--judge-backend", default=None, help="groq (default) or ollama")
    r.add_argument("--judge-model", default=None)
    a = sub.add_parser("audit")
    a.add_argument("--n", type=int, default=30)
    a.add_argument("--seed", type=int, default=0)
    a.add_argument("--exclude", nargs="*", default=[], help="earlier audit CSVs to exclude")
    sub.add_parser("agreement")
    args = p.parse_args()

    if args.cmd == "run":
        from llm import get_backend, get_model
        from rag import load_vectorstore

        backend = get_backend(args.judge_backend)
        model = get_model(backend, args.judge_model)
        judge_fn = lambda q, t, y, passage: judge_passage(q, t, y, passage, backend=backend, model=model)
        rows, summary = run_eval(load_vectorstore(), QUESTIONS[: args.max_questions], args.retrievers,
                                 args.k, judge_fn, judge_label=f"{backend}:{model}")
        print("\n" + write_results(rows, summary, args.k))
        print(f"\nJudge: {backend}:{model}. Embedding model: {EMBEDDING_MODEL}. Results in {RESULTS_DIR}/")
    elif args.cmd == "audit":
        target = RESULTS_DIR / AUDIT_FILE
        if target.exists() and str(target) not in map(str, args.exclude):
            raise SystemExit(f"{target} exists. Rename it (e.g. judge_audit_round1.csv) and pass it with --exclude.")
        path, n = write_audit(args.n, seed=args.seed, exclude_paths=args.exclude)
        print(f"Wrote {n} passages to {path}.\nFill in human_grade (0/1/2) using the same rubric as the judge, "
              f"then run: python src/eval_passages.py agreement")
    else:
        print(json.dumps(audit_agreement(), indent=2))


if __name__ == "__main__":
    main()
