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
    assert llm.get_backend() == "ollama" and llm.get_model("ollama") == llm.DEFAULT_MODELS["ollama"]
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
    monkeypatch.delenv("GROQ_MODEL", raising=False)
    monkeypatch.delenv("GROQ_PRICE_INPUT_PER_M", raising=False)
    monkeypatch.delenv("GROQ_PRICE_OUTPUT_PER_M", raising=False)
    calls = {}

    class FakeCompletions:
        def create(self, **kw):
            calls.update(kw)
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="{}"))],
                                   usage=SimpleNamespace(prompt_tokens=1_000_000, completion_tokens=1_000_000))

    monkeypatch.setattr(llm, "_groq_client", SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions())))
    res = llm.complete([{"role": "user", "content": "hi"}], backend="groq")
    assert calls["response_format"] == {"type": "json_object"}
    assert calls["model"] == "openai/gpt-oss-120b"
    # reasoning model: bigger budget + explicit effort
    assert calls["max_tokens"] >= llm.REASONING_MIN_TOKENS
    assert calls["extra_body"]["reasoning_effort"] == "low"
    assert res.cost_usd == pytest.approx(0.15 + 0.60)


def test_non_reasoning_groq_model_keeps_budget(monkeypatch):
    calls = {}

    class FakeCompletions:
        def create(self, **kw):
            calls.update(kw)
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="{}"))], usage=None)

    monkeypatch.setattr(llm, "_groq_client", SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions())))
    llm.complete([{"role": "user", "content": "hi"}], backend="groq", model="some-other-model", max_tokens=600)
    assert calls["max_tokens"] == 600 and "extra_body" not in calls


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
