# src/rag.py
"""Risk-signal generation over 10-K filings.

Context for the LLM is built in one of three modes (CONTEXT_MODE env var or the
context_mode argument):

    hybrid    (default) Item 1A excerpt + query-specific retrieved chunks
    retrieval only the retrieved chunks for the user's query
    section   only the Item 1A excerpt (the original behaviour; use it to reproduce
              earlier backtest numbers)

How chunks are retrieved is a separate setting (RETRIEVER, see retrieval.py):
dense (FAISS), bm25, or hybrid (default: dense + BM25 fused). Don't confuse the
"hybrid" context mode above with hybrid retrieval.

Retrieval is filtered to the exact ticker AND filing year. There is deliberately
no fallback to other years: borrowing a later filing's text would leak future
information into a point-in-time signal (look-ahead bias).
"""
import json
import os
from functools import lru_cache

from dotenv import load_dotenv

from config import EMBEDDING_MODEL, FILINGS_DIR, INDEX_DIR
from llm import LLMResult, complete
from text_utils import clean_filing_text, extract_risk_section

load_dotenv()

DEFAULT_QUERY = "What are the major risk factors and investment outlook?"
SECTION_CHARS = {"section": 4000, "hybrid": 2500}
RETRIEVED_CHARS = 3000
MAX_ATTEMPTS = 2  # first try + one repair retry (app / backtest path)

RISK_LEVELS = {"LOW", "MEDIUM", "HIGH"}
SENTIMENTS = {"BULLISH", "NEUTRAL", "BEARISH"}

# JSON Schema for the signal. Backends that support structured outputs (Ollama) are
# constrained to it while decoding, so small local models can't return e.g. key_risks
# as objects instead of strings. validate_signal() still checks every backend's output.
SIGNAL_SCHEMA = {
    "type": "object",
    "properties": {
        "ticker": {"type": "string"},
        "risk_level": {"type": "string", "enum": sorted(RISK_LEVELS)},
        "sentiment": {"type": "string", "enum": sorted(SENTIMENTS)},
        "key_risks": {"type": "array", "items": {"type": "string"}, "minItems": 3, "maxItems": 3},
        "rationale": {"type": "string"},
        "red_flags": {"type": "array", "items": {"type": "string"}},
        "answerable": {"type": "boolean"},
    },
    "required": ["ticker", "answerable", "risk_level", "sentiment", "key_risks", "rationale", "red_flags"],
}

SYSTEM_PROMPT = (
    "You are a senior financial analyst reviewing a specific SEC 10-K filing. "
    "Use only the filing text you are given. Respond with a single JSON object and nothing else."
)

USER_TEMPLATE = """Extract SPECIFIC risks mentioned in THIS filing only. Do not use generic risks like "market volatility" or "regulatory changes" unless they are explicitly discussed with specific details in the text below.

Analyst question: {query}

Return a JSON object with exactly these keys:
- "ticker": string
- "risk_level": one of "LOW", "MEDIUM", "HIGH"
- "sentiment": one of "BULLISH", "NEUTRAL", "BEARISH"
- "answerable": true if the excerpt addresses the analyst question, false if it does not (for example, the question is about a different company or a topic the filing doesn't cover)
- "key_risks": list of 3 SPECIFIC, distinct risks with details from the filing. Focus them on the analyst question when answerable; otherwise give the 3 most important risks in the excerpt. Never repeat a risk.
- "rationale": one sentence referencing specific numbers or events from the filing
- "red_flags": list of short warning phrases copied verbatim from the filing that are NOT already stated in key_risks (e.g. going-concern doubts, material weaknesses, restatements, covenant breaches, regulatory orders); empty list if none

Filing year: {year}
Company: {ticker}

Filing excerpt:
{context}"""


# ── VECTOR STORE ────────────────────────────────────────────────
@lru_cache(maxsize=1)
def load_vectorstore():
    from langchain_community.vectorstores import FAISS

    from embeddings import get_embeddings

    if not INDEX_DIR.exists():
        raise FileNotFoundError(
            f"No index for EMBEDDING_MODEL={EMBEDDING_MODEL} at {INDEX_DIR}. "
            f"Build it with: python src/ingest.py --skip-download")
    # The index is built locally by ingest.py, so deserializing it is safe.
    return FAISS.load_local(str(INDEX_DIR), get_embeddings(EMBEDDING_MODEL), allow_dangerous_deserialization=True)


def retrieve_context(vectorstore, ticker, query, year=None, k=5, method=None):
    """Top-k chunks for `query` from exactly this ticker (and filing year, if given).

    method: dense | bm25 | hybrid (default: RETRIEVER setting, hybrid). See retrieval.py.
    """
    from retrieval import retrieve

    return retrieve(vectorstore, ticker, query, year=year, k=k, method=method)


# ── RAW FILINGS ─────────────────────────────────────────────────
def find_filing(ticker, year):
    from config import parse_filing_path

    for path in sorted((FILINGS_DIR / ticker / "10-K").glob("*/full-submission.txt")):
        _, _, file_year = parse_filing_path(path)
        if file_year == year:
            return path
    return None


def get_context_from_file(ticker, year, max_chars=6000):
    """Item 1A excerpt from the raw filing (cleaned of HTML), with fallbacks."""
    path = find_filing(ticker, year)
    if path is None:
        return None
    with open(path, "r", errors="ignore") as f:
        text = clean_filing_text(f.read())

    section = extract_risk_section(text, max_chars=max_chars)
    if section:
        return section

    idx = text.upper().find("RISK FACTORS", len(text) // 20)  # skip the table of contents
    if idx != -1 and len(text) - idx > 500:
        return text[idx: idx + max_chars]

    print(f"No risk section found for {ticker} {year}, using raw text")
    return text[5000: 5000 + max_chars]


def build_context(ticker, year, query, context_mode=None, k=5, vectorstore=None):
    """Return (context_text, sources) for the given mode."""
    mode = (context_mode or os.getenv("CONTEXT_MODE", "hybrid")).lower()
    if mode not in {"hybrid", "retrieval", "section"}:
        raise ValueError(f"Unknown context_mode '{mode}'")

    parts, sources = [], []

    if mode in {"section", "hybrid"}:
        excerpt = get_context_from_file(ticker, year)
        if excerpt:
            parts.append("[Item 1A Risk Factors excerpt]\n" + excerpt[: SECTION_CHARS[mode]])
            sources.append({"type": "item_1a_excerpt", "year": year})

    if mode in {"retrieval", "hybrid"}:
        try:
            vs = vectorstore or load_vectorstore()
            docs = retrieve_context(vs, ticker, query, year=year, k=k)
        except Exception as e:  # index not built yet, etc.
            print(f"Retrieval unavailable ({e}); continuing without retrieved chunks")
            docs = []
        budget = RETRIEVED_CHARS if mode == "hybrid" else RETRIEVED_CHARS + SECTION_CHARS["hybrid"]
        used = 0
        for d in docs:
            chunk = d.page_content.strip()
            if parts and chunk[:200] in parts[0]:  # skip chunks already in the excerpt
                continue
            if used + len(chunk) > budget:
                break
            parts.append(f"[Retrieved: {d.metadata.get('section', 'filing')}]\n{chunk}")
            sources.append({"type": "retrieved_chunk", "year": d.metadata.get("year"),
                            "section": d.metadata.get("section"),
                            "chunk_id": d.metadata.get("chunk_id")})
            used += len(chunk)

    return "\n\n".join(parts), sources


# ── OUTPUT PARSING ──────────────────────────────────────────────
def parse_signal(raw):
    try:
        start, end = raw.index("{"), raw.rindex("}") + 1
        return json.loads(raw[start:end])
    except (ValueError, json.JSONDecodeError):
        return None


def validate_signal(sig):
    """List of schema problems (empty list means the output is valid)."""
    if not isinstance(sig, dict):
        return ["not a JSON object"]
    errors = []
    if sig.get("risk_level") not in RISK_LEVELS:
        errors.append(f"risk_level={sig.get('risk_level')!r}")
    if sig.get("sentiment") not in SENTIMENTS:
        errors.append(f"sentiment={sig.get('sentiment')!r}")
    risks = sig.get("key_risks")
    if not isinstance(risks, list) or len(risks) != 3 or not all(isinstance(r, str) for r in risks):
        errors.append("key_risks must be a list of 3 strings")
    if not isinstance(sig.get("rationale"), str):
        errors.append("rationale missing")
    if not isinstance(sig.get("red_flags"), list):
        errors.append("red_flags must be a list")
    if "answerable" in sig and not isinstance(sig["answerable"], bool):
        errors.append("answerable must be true or false")
    return errors


# ── SIGNAL GENERATION ───────────────────────────────────────────
def build_messages(ticker, year, query, context):
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_TEMPLATE.format(
            query=query, year=year, ticker=ticker, context=context)},
    ]


def normalize_signal(sig):
    """Fix harmless formatting slips ("high", " Bearish") so they don't fail validation."""
    if isinstance(sig, dict):
        for key in ("risk_level", "sentiment"):
            if isinstance(sig.get(key), str):
                sig[key] = sig[key].strip().upper()
    return sig


def run_llm(ticker, year, query, context, backend=None, model=None, max_attempts=1):
    """Call the LLM on a prepared context. Returns (signal_dict, LLMResult).

    With max_attempts > 1, an unparseable or schema-invalid answer is sent back to
    the model with the specific errors and it gets another try. The default is a
    single attempt so compare_backends.py measures each model's first-try validity;
    generate_signal() (the app path) passes MAX_ATTEMPTS.

    The returned LLMResult sums latency and tokens over all attempts, so cost
    numbers include retries.
    """
    messages = build_messages(ticker, year, query, context)
    results, signal, errors = [], None, []
    for _ in range(max_attempts):
        result = complete(messages, backend=backend, model=model, json_mode=SIGNAL_SCHEMA)
        results.append(result)
        signal = normalize_signal(parse_signal(result.text))
        errors = ["could not parse JSON"] if signal is None else validate_signal(signal)
        if not errors:
            break
        messages = messages + [
            {"role": "assistant", "content": result.text},
            {"role": "user", "content": "Your response was invalid: " + "; ".join(errors)
                + ". Return only the corrected JSON object with exactly the required keys."},
        ]

    last = results[-1]
    total = LLMResult(
        text=last.text, backend=last.backend, model=last.model,
        latency_s=sum(r.latency_s for r in results),
        input_tokens=sum(r.input_tokens for r in results),
        output_tokens=sum(r.output_tokens for r in results),
    )
    if signal is None:
        signal = {"ticker": ticker, "raw_response": last.text, "error": "Could not parse JSON"}
    signal["_meta"] = {
        "backend": total.backend,
        "model": total.model,
        "latency_s": round(total.latency_s, 2),
        "input_tokens": total.input_tokens,
        "output_tokens": total.output_tokens,
        "attempts": len(results),
        "schema_errors": errors,
    }
    return signal, total


def generate_signal(ticker, query=DEFAULT_QUERY, year=None, backend=None, model=None, context_mode=None):
    query = query or DEFAULT_QUERY
    if not year:
        from trend import get_filings_by_year
        filings = get_filings_by_year(ticker)
        if not filings:
            return {"error": f"No filing data found for {ticker}"}
        year = filings[-1]["year"]  # most recent

    print(f"Retrieving context for {ticker} ({year})...")
    context, sources = build_context(ticker, year, query, context_mode=context_mode)
    if not context:
        return {"error": f"No filing found for {ticker} {year}"}

    print(f"Calling LLM for {ticker}...")
    signal, _ = run_llm(ticker, year, query, context, backend=backend, model=model,
                        max_attempts=MAX_ATTEMPTS)
    if signal["_meta"]["schema_errors"] and "error" not in signal:
        # Don't let a half-valid signal (e.g. risk_level="SEVERE") reach the app or the backtest.
        signal["error"] = "LLM output failed schema validation: " + "; ".join(signal["_meta"]["schema_errors"])
    signal["year"] = year
    signal["_meta"]["context_mode"] = (context_mode or os.getenv("CONTEXT_MODE", "hybrid")).lower()
    signal["_meta"]["sources"] = sources
    return signal


if __name__ == "__main__":
    for ticker, year in [("AAPL", 2024), ("JPM", 2025)]:
        print(f"\n{'=' * 50} {ticker} ({year})")
        print(json.dumps(generate_signal(ticker, year=year), indent=2))
