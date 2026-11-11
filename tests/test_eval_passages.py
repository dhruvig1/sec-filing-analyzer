# tests/test_eval_passages.py
"""Offline tests for the passage-level eval: fake embeddings, rule-based judge, no API calls."""
import csv
import math
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import eval_passages as ev  # noqa: E402


# ── metric math ────────────────────────────────────────────────
def test_precision_mrr_ndcg_by_hand():
    g = [0, 2, 1, 0, 0]
    assert ev.precision_at_k(g) == 0.4
    assert ev.precision_at_k(g, min_grade=2) == 0.2
    assert ev.reciprocal_rank(g) == 0.5
    dcg = 3 / math.log2(3) + 1 / math.log2(4)
    ideal = 3 / math.log2(2) + 3 / math.log2(3) + 1 / math.log2(4)  # pooled has two 2s and a 1
    assert ev.ndcg_at_k(g, pooled_grades=[2, 2, 1, 0, 0, 0]) == pytest.approx(dcg / ideal)
    assert ev.ndcg_at_k([2, 2, 1, 0, 0], [2, 2, 1, 0]) == pytest.approx(1.0)
    assert ev.ndcg_at_k([0] * 5, [0, 0]) == 0.0


def test_rrf_rewards_agreement_between_rankers():
    class D:
        def __init__(self, cid):
            self.metadata, self.page_content = {"chunk_id": cid}, cid
    a, b, c = D("a"), D("b"), D("c")
    fused = ev.rrf_fuse([[a, b, c], [b, c, a]])
    assert [d.metadata["chunk_id"] for d in fused][0] == "b"  # ranked high by both


def test_cohen_kappa():
    assert ev.cohen_kappa([0, 1, 2, 2], [0, 1, 2, 2]) == 1.0
    assert ev.cohen_kappa([0, 0, 1, 1], [1, 1, 0, 0]) == pytest.approx(-1.0)


def test_bootstrap_ci_brackets_mean():
    lo, hi = ev.bootstrap_ci([0.2, 0.4, 0.6, 0.8, 1.0])
    assert lo <= 0.6 <= hi


# ── end to end on a tiny index ─────────────────────────────────
TOPICS = {
    "credit": "The provision for credit losses increased due to higher net charge-offs in card loans. ",
    "rates": "Net interest income is sensitive to changes in benchmark interest rates and deposit costs. ",
    "cyber": "Cyber attacks on our information systems could disrupt operations and expose data. ",
}


@pytest.fixture
def vs(tmp_path, monkeypatch):
    import ingest
    from langchain_community.embeddings import DeterministicFakeEmbedding

    monkeypatch.setattr(ingest, "INDEX_DIR", tmp_path / "faiss_index")
    ev._company_cache.clear(); ev._bm25_cache.clear()
    docs = []
    for ticker in ["JPM", "BAC"]:
        for year in [2025, 2026]:
            for topic, sentence in TOPICS.items():
                docs.append({"text": f"{ticker} {topic}. " + sentence * 12, "source": ticker,
                             "section": "RISK FACTORS", "file": "f", "year": year})
    return ingest.build_vector_store(docs, embeddings=DeterministicFakeEmbedding(size=32))


def rule_judge(calls):
    def judge(question, ticker, year, passage):
        calls.append(passage)
        if "credit losses" in passage:
            return 2, "about credit losses"
        return (1, "banking") if "interest" in passage else (0, "off topic")
    return judge


def test_retrievers_stay_inside_company_and_latest_year(vs):
    assert ev.latest_year(vs, "JPM") == 2026
    for name, fn in ev.RETRIEVERS.items():
        docs = fn(vs, "JPM", 2026, "credit losses charge-offs")
        assert docs, name
        assert all(d.metadata["source"] == "JPM" and d.metadata["year"] == 2026 for d in docs), name


def test_bm25_ranks_keyword_match_first(vs):
    top = ev.bm25_ranking(vs, "JPM", 2026, "provision for credit losses net charge-offs")[0]
    assert "credit losses" in top.page_content


def test_run_eval_pools_judgments_and_caches(vs, tmp_path):
    calls = []
    qs = [("JPM", "What drove the provision for credit losses?"), ("ZZZ", "not indexed")]
    path = tmp_path / "judgments.jsonl"
    rows, summary = ev.run_eval(vs, qs, judge_fn=rule_judge(calls), judgments_path=path)

    assert {r["retriever"] for r in rows} == {"dense", "bm25", "hybrid"} and all(r["ticker"] == "JPM" for r in rows)
    pooled = len({line for line in path.read_text().splitlines()})
    assert len(calls) == pooled  # each pooled passage judged exactly once
    bm25 = next(r for r in rows if r["retriever"] == "bm25")
    assert bm25["grades"][0] == 2 and bm25["mrr"] == 1.0
    assert set(summary) == {"dense", "bm25", "hybrid"} and summary["bm25"]["n_questions"] == 1

    calls.clear()
    ev.run_eval(vs, qs, judge_fn=rule_judge(calls), judgments_path=path)
    assert calls == []  # second run is free


def test_audit_is_blind_and_agreement_roundtrip(vs, tmp_path):
    jpath, apath = tmp_path / "judgments.jsonl", tmp_path / "audit.csv"
    ev.run_eval(vs, [("JPM", "What drove the provision for credit losses?")],
                judge_fn=rule_judge([]), judgments_path=jpath)
    _, n = ev.write_audit(n=6, judgments_path=jpath, audit_path=apath)
    with open(apath, newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == n and "judge_grade" not in rows[0] and "grade" not in rows[0]

    cache = ev.load_judgments(jpath)
    for r in rows:  # a human who agrees with the judge on everything
        r["human_grade"] = cache[r["key"]]["grade"]
    with open(apath, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    out = ev.audit_agreement(apath, jpath)
    assert out["n"] == n and out["exact_agreement"] == 1.0


def test_markdown_table_has_all_retrievers(vs, tmp_path):
    rows, summary = ev.run_eval(vs, [("JPM", "credit losses?"), ("BAC", "interest rates?")],
                                judge_fn=rule_judge([]), judgments_path=tmp_path / "j.jsonl")
    md = ev.to_markdown(summary)
    assert all(f"| {name} |" in md for name in ("dense", "bm25", "hybrid")) and "nDCG@5" in md


def test_second_audit_round_excludes_first(vs, tmp_path):
    jpath = tmp_path / "judgments.jsonl"
    ev.run_eval(vs, [("JPM", "credit losses?"), ("BAC", "interest rates?"), ("JPM", "cyber attacks?")],
                judge_fn=rule_judge([]), judgments_path=jpath)
    r1, r2 = tmp_path / "round1.csv", tmp_path / "round2.csv"
    ev.write_audit(n=4, judgments_path=jpath, audit_path=r1)
    ev.write_audit(n=4, seed=1, judgments_path=jpath, audit_path=r2, exclude_paths=[r1])
    keys = lambda p: {r["key"] for r in csv.DictReader(open(p, newline=""))}
    assert keys(r1) and keys(r2) and not keys(r1) & keys(r2)
