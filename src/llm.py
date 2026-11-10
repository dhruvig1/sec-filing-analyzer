# src/llm.py
"""Swappable inference layer: hosted Llama-3.3-70B on Groq, or a local quantized model via Ollama.

Select the backend with environment variables (or pass backend=/model= explicitly):

    LLM_BACKEND=groq      GROQ_MODEL=llama-3.3-70b-versatile    (needs GROQ_API_KEY)
    LLM_BACKEND=ollama    OLLAMA_MODEL=llama3.1:8b               (needs `ollama serve`)

Ollama's default tags (e.g. llama3.1:8b, qwen2.5:7b) are 4-bit quantized (Q4_K_M),
so an 8B model runs in roughly 5 GB of RAM on a laptop. Filing text never leaves
the machine when LLM_BACKEND=ollama.
"""
import os
import time
from dataclasses import dataclass

import requests

DEFAULT_MODELS = {
    "groq": "llama-3.3-70b-versatile",
    "ollama": "llama3.1:8b",
}

# USD per 1M tokens. Groq's published on-demand rate for llama-3.3-70b-versatile
# (verify at https://groq.com/pricing). Local inference has no per-token charge.
PRICING = {
    "groq": {
        "input": float(os.getenv("GROQ_PRICE_INPUT_PER_M", "0.59")),
        "output": float(os.getenv("GROQ_PRICE_OUTPUT_PER_M", "0.79")),
    },
    "ollama": {"input": 0.0, "output": 0.0},
}


@dataclass
class LLMResult:
    text: str
    backend: str
    model: str
    latency_s: float
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def cost_usd(self) -> float:
        p = PRICING.get(self.backend, {"input": 0.0, "output": 0.0})
        return (self.input_tokens * p["input"] + self.output_tokens * p["output"]) / 1_000_000


def get_backend(backend=None) -> str:
    backend = (backend or os.getenv("LLM_BACKEND", "groq")).lower()
    if backend not in DEFAULT_MODELS:
        raise ValueError(f"Unknown LLM_BACKEND '{backend}'. Use one of: {list(DEFAULT_MODELS)}")
    return backend


def get_model(backend: str, model=None) -> str:
    return model or os.getenv(f"{backend.upper()}_MODEL", DEFAULT_MODELS[backend])


_groq_client = None


def _groq_complete(messages, model, temperature, max_tokens, json_mode) -> LLMResult:
    global _groq_client
    if _groq_client is None:  # created lazily so importing this module never needs an API key
        from groq import Groq
        _groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    kwargs = dict(model=model, messages=messages, temperature=temperature, max_tokens=max_tokens)
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    start = time.perf_counter()
    resp = _groq_client.chat.completions.create(**kwargs)
    latency = time.perf_counter() - start
    usage = getattr(resp, "usage", None)
    return LLMResult(
        text=resp.choices[0].message.content,
        backend="groq",
        model=model,
        latency_s=latency,
        input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
        output_tokens=getattr(usage, "completion_tokens", 0) or 0,
    )


def _ollama_complete(messages, model, temperature, max_tokens, json_mode) -> LLMResult:
    host = os.getenv("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
            # Ollama's default context window is small and silently truncates long
            # prompts; filing excerpts need more room.
            "num_ctx": int(os.getenv("OLLAMA_NUM_CTX", "8192")),
        },
    }
    if json_mode:
        payload["format"] = "json"
    start = time.perf_counter()
    try:
        resp = requests.post(f"{host}/api/chat", json=payload,
                             timeout=float(os.getenv("OLLAMA_TIMEOUT", "300")))
    except requests.ConnectionError as e:
        raise RuntimeError(
            f"Could not reach Ollama at {host}. Start it with `ollama serve` "
            f"and pull the model with `ollama pull {model}`."
        ) from e
    latency = time.perf_counter() - start
    resp.raise_for_status()
    data = resp.json()
    return LLMResult(
        text=data["message"]["content"],
        backend="ollama",
        model=model,
        latency_s=latency,
        input_tokens=data.get("prompt_eval_count", 0) or 0,
        output_tokens=data.get("eval_count", 0) or 0,
    )


_BACKENDS = {"groq": _groq_complete, "ollama": _ollama_complete}


def complete(messages, backend=None, model=None, temperature=0.2, max_tokens=600, json_mode=True) -> LLMResult:
    """Run one chat completion on the selected backend and return text plus timing/usage."""
    backend = get_backend(backend)
    model = get_model(backend, model)
    return _BACKENDS[backend](messages, model, temperature, max_tokens, json_mode)
