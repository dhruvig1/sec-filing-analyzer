# src/retrieval.py
"""Retrieval over the 10-K index: dense (FAISS), BM25 (keywords), and hybrid (both, fused).

Every method searches one company's filing for one year only. Retrieval never
falls back to other years: borrowing a later filing would leak future information
into a point-in-time signal (look-ahead bias).

The app and eval_passages.py both use this module, so the benchmark scores the
exact code the app runs.
"""
import hashlib
import re
from collections import defaultdict

from config import RETRIEVER

CANDIDATES = 20   # depth each method ranks before fusion
RRF_K = 60        # standard reciprocal-rank-fusion constant

_TOKEN = re.compile(r"[a-z0-9]+")
_STOP = set("""a an and are as at be by for from has have how in is it its of on or that the this to
was were what which with does do our we us their they them company""".split())


def tokenize(text):
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOP]


def chunk_key(doc):
    return doc.metadata.get("chunk_id") or hashlib.sha1(doc.page_content.encode()).hexdigest()[:12]


# ── corpus helpers ─────────────────────────────────────────────
def _all_docs(vs):
    return list(vs.docstore._dict.values())


def latest_year(vs, ticker):
    years = [d.metadata.get("year") for d in _all_docs(vs) if d.metadata.get("source") == ticker]
    years = [y for y in years if y]
    return max(years) if years else None


_company_cache = {}


def company_docs(vs, ticker, year):
    key = (id(vs), ticker, year)
    if key not in _company_cache:
        _company_cache[key] = [d for d in _all_docs(vs)
                               if d.metadata.get("source") == ticker and d.metadata.get("year") == year]
    return _company_cache[key]


# ── methods ────────────────────────────────────────────────────
def dense_ranking(vs, ticker, year, query, depth=CANDIDATES):
    """Exact metadata filter. LangChain's FAISS filters *after* the search, over the top
    fetch_k hits, so fetch_k covers the whole index to keep the filtered search exact."""
    flt = {"source": ticker.upper()}
    if year is not None:
        flt["year"] = int(year)
    fetch_k = max(depth * 20, getattr(getattr(vs, "index", None), "ntotal", 0))
    return vs.similarity_search(query, k=depth, filter=flt, fetch_k=fetch_k)


_bm25_cache = {}


def bm25_ranking(vs, ticker, year, query, depth=CANDIDATES):
    from rank_bm25 import BM25Okapi

    docs = company_docs(vs, ticker.upper(), year)
    if not docs:
        return []
    key = (id(vs), ticker.upper(), year)
    if key not in _bm25_cache:
        _bm25_cache[key] = BM25Okapi([tokenize(d.page_content) for d in docs])
    scores = _bm25_cache[key].get_scores(tokenize(query))
    order = sorted(range(len(docs)), key=lambda i: scores[i], reverse=True)
    return [docs[i] for i in order[:depth]]


def rrf_fuse(rankings, depth=CANDIDATES, k=RRF_K):
    """Reciprocal rank fusion: score(d) = sum over rankings of 1 / (k + rank)."""
    scores, by_key = defaultdict(float), {}
    for ranking in rankings:
        for rank, doc in enumerate(ranking, 1):
            key = chunk_key(doc)
            scores[key] += 1.0 / (k + rank)
            by_key[key] = doc
    return [by_key[key] for key in sorted(scores, key=scores.get, reverse=True)[:depth]]


def hybrid_ranking(vs, ticker, year, query, depth=CANDIDATES):
    return rrf_fuse([dense_ranking(vs, ticker, year, query), bm25_ranking(vs, ticker, year, query)], depth)


METHODS = {"dense": dense_ranking, "bm25": bm25_ranking, "hybrid": hybrid_ranking}


def retrieve(vs, ticker, query, year=None, k=5, method=None):
    """Top-k chunks for `query` from one company (and filing year, if given)."""
    method = (method or RETRIEVER).lower()
    if method not in METHODS:
        raise ValueError(f"Unknown retriever '{method}'. Use one of: {list(METHODS)}")
    if method != "dense" and not hasattr(vs, "docstore"):
        method = "dense"  # BM25 needs the stored chunks; plain vector stores only support dense
    if method != "dense" and year is None:
        year = latest_year(vs, ticker.upper())
    return METHODS[method](vs, ticker, year, query, depth=max(k, CANDIDATES))[:k]
