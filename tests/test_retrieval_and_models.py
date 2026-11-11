# tests/test_retrieval_and_models.py
"""Shared retrieval module, embedding-model config, and Ollama thinking handling (offline)."""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import config      # noqa: E402
import llm         # noqa: E402
import rag         # noqa: E402
import retrieval   # noqa: E402

TEXT = {
    "credit": "The provision for credit losses increased due to higher net charge-offs in card loans. ",
    "rates": "Net interest income is sensitive to changes in benchmark interest rates and deposit costs. ",
}


@pytest.fixture
def vs(tmp_path, monkeypatch):
    import ingest
    from langchain_community.embeddings import DeterministicFakeEmbedding

    monkeypatch.setattr(ingest, "INDEX_DIR", tmp_path / "idx")
    retrieval._company_cache.clear(); retrieval._bm25_cache.clear()
    docs = [{"text": f"{t} {topic}. " + sent * 12, "source": t, "section": "RISK FACTORS", "file": "f", "year": y}
            for t in ["C", "COP"] for y in [2025, 2026] for topic, sent in TEXT.items()]
    return ingest.build_vector_store(docs, embeddings=DeterministicFakeEmbedding(size=32))


def test_app_retrieval_defaults_to_hybrid(vs, monkeypatch):
    called = []
    monkeypatch.setitem(retrieval.METHODS, "hybrid", lambda *a, **k: called.append("hybrid") or [])
    rag.retrieve_context(vs, "C", "credit losses", year=2026)
    assert called == ["hybrid"] and config.RETRIEVER == "hybrid"


@pytest.mark.parametrize("method", ["dense", "bm25", "hybrid"])
def test_every_method_stays_in_company_and_year(vs, method):
    docs = rag.retrieve_context(vs, "c", "provision for credit losses charge-offs", year=2026, k=4, method=method)
    assert docs and len(docs) <= 4
    assert all(d.metadata["source"] == "C" and d.metadata["year"] == 2026 for d in docs)


def test_bm25_and_hybrid_use_latest_year_when_none_given(vs):
    docs = rag.retrieve_context(vs, "C", "interest rates", year=None, method="bm25")
    assert docs and all(d.metadata["year"] == 2026 for d in docs)


def test_unknown_method_and_plain_vectorstore_fallback():
    class Plain:  # no docstore: BM25 impossible, falls back to dense
        def similarity_search(self, q, k, filter, fetch_k):
            self.filter = filter
            return []
    p = Plain()
    assert rag.retrieve_context(p, "MS", "q", year=2025) == [] and p.filter == {"source": "MS", "year": 2025}
    with pytest.raises(ValueError):
        rag.retrieve_context(p, "MS", "q", method="nope")


def test_index_dir_per_embedding_model():
    assert config.index_dir_for("all-MiniLM-L6-v2").name == "faiss_index"  # existing index untouched
    assert config.index_dir_for("Qwen/Qwen3-Embedding-0.6B").name == "faiss_index__qwen3-embedding-0_6b"


def test_qwen_embedder_adds_instruction_to_queries_only(monkeypatch):
    import types
    import embeddings

    seen = []

    class FakeST:
        def __init__(self, name, **kw):
            self.name = name
            self.device = kw.get("device", "cpu")
        def encode(self, texts, **kw):
            seen.append((self.name, list(texts), kw))
            return [[1.0, 0.0] for _ in texts]

    monkeypatch.setitem(sys.modules, "sentence_transformers", types.SimpleNamespace(SentenceTransformer=FakeST))
    fake_torch = types.SimpleNamespace(backends=types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: False)),
                                       cuda=types.SimpleNamespace(is_available=lambda: False))
    monkeypatch.setitem(sys.modules, "torch", fake_torch)  # don't load real PyTorch next to FAISS in tests
    embeddings.get_embeddings.cache_clear()

    qwen = embeddings.get_embeddings("Qwen/Qwen3-Embedding-0.6B")
    assert qwen.embed_query("credit losses?") == [1.0, 0.0]
    assert seen[-1][1][0].startswith("Instruct: ") and seen[-1][1][0].endswith("Query: credit losses?")
    qwen.embed_documents(["passage text"])
    assert seen[-1][1] == ["passage text"] and seen[-1][2]["normalize_embeddings"]  # no instruction on documents

    mini = embeddings.get_embeddings("all-MiniLM-L6-v2")
    mini.embed_query("credit losses?")
    assert seen[-1][1] == ["credit losses?"]
    embeddings.get_embeddings.cache_clear()


def test_qwen3_thinking_off_and_think_tags_stripped(monkeypatch):
    sent = {}

    class Resp:
        def raise_for_status(self): pass
        def json(self):
            return {"message": {"content": "<think>hmm {not json}</think>\n{\"ok\": 1}"}, "eval_count": 5}

    monkeypatch.setattr(llm.requests, "post", lambda url, json, timeout: sent.update(json) or Resp())
    res = llm.complete([{"role": "user", "content": "x"}], backend="ollama")
    assert res.model == "qwen3:8b" and sent["think"] is False and res.text == '{"ok": 1}'

    sent.clear()
    llm.complete([{"role": "user", "content": "x"}], backend="ollama", model="llama3.1:8b")
    assert "think" not in sent  # non-thinking models get no think flag


def test_answerable_field_in_schema_prompt_and_validator():
    assert "answerable" in rag.SIGNAL_SCHEMA["required"]
    assert rag.SIGNAL_SCHEMA["properties"]["answerable"] == {"type": "boolean"}
    assert '"answerable"' in rag.USER_TEMPLATE and "NOT already stated in key_risks" in rag.USER_TEMPLATE
    base = {"ticker": "BLK", "risk_level": "LOW", "sentiment": "NEUTRAL",
            "key_risks": ["a", "b", "c"], "rationale": "r", "red_flags": []}
    assert rag.validate_signal(base) == []
    assert rag.validate_signal({**base, "answerable": False}) == []
    assert rag.validate_signal({**base, "answerable": "no"}) == ["answerable must be true or false"]
