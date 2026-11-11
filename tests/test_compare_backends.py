import numpy as np

from compare_backends import (grounding_scores, parse_backend_spec, redflag_verbatim_rate,
                              run_comparison, summarize, to_markdown)
from llm import LLMResult

CONTEXT = ("The company relies on a single supplier in Taiwan for advanced chips. "
           "Net charge-offs rose to 2.1% of loans in commercial real estate. "
           "A material weakness in internal controls was identified.")
WORDS = ["supplier", "taiwan", "chips", "charge-offs", "real", "estate", "weakness", "controls", "dragons"]


def bow_embed(texts):  # deterministic bag-of-words "embedding" for tests
    return np.array([[t.lower().count(w) for w in WORDS] + [0.01] for t in texts], dtype=float)


def test_backend_spec_keeps_model_tag():
    assert parse_backend_spec("ollama:llama3.1:8b") == ("ollama", "llama3.1:8b")
    assert parse_backend_spec("groq:openai/gpt-oss-120b") == ("groq", "openai/gpt-oss-120b")


def test_grounding_separates_supported_from_invented():
    supported, _ = grounding_scores(["Dependence on a Taiwan chips supplier"], CONTEXT, bow_embed)
    invented, frac = grounding_scores(["Exposure to dragons"], CONTEXT, bow_embed)
    assert supported > 0.5 > invented and frac == 0.0


def test_redflag_verbatim():
    assert redflag_verbatim_rate(["material weakness in internal controls", "going concern"], CONTEXT) == 0.5
    assert redflag_verbatim_rate([], CONTEXT) is None


def test_end_to_end_comparison_with_fake_models():
    outputs = {
        "openai/gpt-oss-120b": {"ticker": "AAPL", "risk_level": "HIGH", "sentiment": "BEARISH",
                "key_risks": ["single Taiwan chips supplier", "charge-offs in real estate", "weakness in controls"],
                "rationale": "x", "red_flags": ["material weakness in internal controls"]},
        "small": {"ticker": "AAPL", "risk_level": "MEDIUM", "sentiment": "BEARISH",
                  "key_risks": ["dragons", "dragons", "chips supplier"], "rationale": "x", "red_flags": []},
    }

    def fake_llm(ticker, year, query, context, backend=None, model=None):
        return dict(outputs[model]), LLMResult("{}", backend, model, 1.5 if model == "small" else 0.4, 2000, 200)

    rows = run_comparison([("AAPL", 2024), ("MSFT", 2024)], [("groq", "openai/gpt-oss-120b"), ("ollama", "small")],
                          embed_fn=bow_embed, context_fn=lambda *a, **k: (CONTEXT, []), llm_fn=fake_llm)
    s = summarize(rows)
    assert s["groq:openai/gpt-oss-120b"]["grounding"] > s["ollama:small"]["grounding"]
    assert s["ollama:small"]["risk_agreement_vs_ref"] == 0.0
    assert s["ollama:small"]["sentiment_agreement_vs_ref"] == 100.0
    assert s["ollama:small"]["total_cost_usd"] == 0.0 and s["groq:openai/gpt-oss-120b"]["total_cost_usd"] > 0
    assert "| groq:openai/gpt-oss-120b |" in to_markdown(s)
