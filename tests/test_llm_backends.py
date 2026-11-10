import json
from types import SimpleNamespace

import pytest

import llm
from rag import parse_signal, validate_signal

VALID = {"ticker": "AAPL", "risk_level": "HIGH", "sentiment": "BEARISH",
         "key_risks": ["a", "b", "c"], "rationale": "r", "red_flags": []}


def test_backend_selection(monkeypatch):
    monkeypatch.setenv("LLM_BACKEND", "ollama")
    monkeypatch.delenv("OLLAMA_MODEL", raising=False)
    assert llm.get_backend() == "ollama" and llm.get_model("ollama") == "llama3.1:8b"
    with pytest.raises(ValueError):
        llm.get_backend("openai")


def test_ollama_request_and_usage(monkeypatch):
    captured = {}

    def fake_post(url, json=None, timeout=None):
        captured.update(url=url, payload=json)
        return SimpleNamespace(raise_for_status=lambda: None, json=lambda: {
            "message": {"content": '{"ok": 1}'}, "prompt_eval_count": 1200, "eval_count": 150})

    monkeypatch.setattr(llm.requests, "post", fake_post)
    res = llm.complete([{"role": "user", "content": "hi"}], backend="ollama", model="qwen2.5:7b")
    assert captured["url"].endswith("/api/chat")
    assert captured["payload"]["format"] == "json" and captured["payload"]["stream"] is False
    assert captured["payload"]["options"]["num_ctx"] >= 8192
    assert (res.backend, res.model, res.input_tokens, res.output_tokens, res.cost_usd) == \
           ("ollama", "qwen2.5:7b", 1200, 150, 0.0)


def test_groq_cost_and_json_mode(monkeypatch):
    calls = {}

    class FakeCompletions:
        def create(self, **kw):
            calls.update(kw)
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="{}"))],
                                   usage=SimpleNamespace(prompt_tokens=1_000_000, completion_tokens=1_000_000))

    monkeypatch.setattr(llm, "_groq_client", SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions())))
    res = llm.complete([{"role": "user", "content": "hi"}], backend="groq")
    assert calls["response_format"] == {"type": "json_object"}
    assert calls["model"] == "llama-3.3-70b-versatile"
    assert res.cost_usd == pytest.approx(llm.PRICING["groq"]["input"] + llm.PRICING["groq"]["output"])


def test_ollama_unreachable_gives_clear_error(monkeypatch):
    def boom(*a, **k):
        raise llm.requests.ConnectionError()
    monkeypatch.setattr(llm.requests, "post", boom)
    with pytest.raises(RuntimeError, match="ollama serve"):
        llm.complete([{"role": "user", "content": "hi"}], backend="ollama")


def test_parse_and_validate():
    assert parse_signal("Sure! " + json.dumps(VALID) + " done") == VALID
    assert parse_signal("no json here") is None
    assert validate_signal(VALID) == []
    assert validate_signal({**VALID, "risk_level": "SEVERE", "key_risks": ["a"]})
