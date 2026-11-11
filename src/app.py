# src/app.py
import os
import re
import sys

import plotly.graph_objects as go
import requests
import streamlit as st

sys.path.append(os.path.dirname(__file__))
from config import SECTORS, TICKERS
from llm import DEFAULT_MODELS
from rag import DEFAULT_QUERY, generate_signal
from trend import get_filings_by_year, risk_to_score

RISK_COLORS = {"LOW": "green", "MEDIUM": "orange", "HIGH": "red"}
SENTIMENT_NOTE = ("Sentiment is still in the raw JSON but not shown: in testing it barely varied "
                  "with the filing (each model gave nearly the same label every time).")

# Page Configuration
st.set_page_config(page_title="SEC Filing Analyzer", layout="wide")
st.title("SEC Filing Analyzer")
st.caption("Risk signals from SEC 10-K filings: hybrid retrieval (BM25 + embeddings) + LLM, with sources")
st.divider()


@st.cache_data(ttl=60, show_spinner=False)
def ollama_available():
    host = os.getenv("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
    try:
        return requests.get(host, timeout=0.5).ok
    except requests.RequestException:
        return False


@st.cache_data(show_spinner=False, max_entries=500)
def _cached_signal(ticker, query, year, backend, model):
    return generate_signal(ticker, query, year=year, backend=backend, model=model)


def cached_signal(ticker, query, year, backend, model):
    """Same inputs give the same signal, so repeated clicks are instant and free.
    Errors (rate limits, Ollama not running) are not kept, so a retry can succeed."""
    sig = _cached_signal(ticker, query, year, backend, model)
    if "error" in sig:
        _cached_signal.clear()
    return sig


def norm(text):
    return re.sub(r"\W+", " ", str(text).lower()).strip()


def distinct_red_flags(sig):
    """Hide red flags that just restate a key risk."""
    risks = [norm(r) for r in sig.get("key_risks", [])]
    return [f for f in sig.get("red_flags", []) if norm(f) and not any(norm(f) in r or r in norm(f) for r in risks)]


def show_sources(sig):
    sources = [s for s in sig.get("_meta", {}).get("sources", []) if s.get("type") == "retrieved_chunk"]
    excerpt = any(s.get("type") == "item_1a_excerpt" for s in sig.get("_meta", {}).get("sources", []))
    with st.expander(f"Sources ({len(sources)} retrieved passages{' + Item 1A excerpt' if excerpt else ''})"):
        for s in sources:
            st.write(f"`{s.get('chunk_id') or 'chunk'}` · {s.get('section', '').title()} · filed {s.get('year')}")
        meta = sig.get("_meta", {})
        if meta.get("latency_s") is not None:
            st.caption(f"{meta.get('backend')}:{meta.get('model')} · {meta.get('latency_s')}s · "
                       f"{meta.get('input_tokens', 0)} in / {meta.get('output_tokens', 0)} out tokens")


def show_not_answerable(sig):
    if sig.get("answerable") is False:
        st.warning("The retrieved filing passages don't address this question. "
                   "Showing the most important risks they do contain instead.")


# ── Sidebar: model backend ──────────────────────────────────────
with st.sidebar:
    st.subheader("Model")
    local_ok = ollama_available()
    options = ["Hosted: Groq"] + (["Local: Ollama"] if local_ok else [])
    default_local = os.getenv("LLM_BACKEND", "groq") == "ollama" and local_ok
    backend_label = st.radio("Inference backend", options, index=1 if default_local else 0)
    BACKEND = "groq" if backend_label.startswith("Hosted") else "ollama"
    MODEL = st.text_input("Model name", value=os.getenv(f"{BACKEND.upper()}_MODEL", DEFAULT_MODELS[BACKEND]))
    if BACKEND == "ollama":
        st.caption("Runs on this machine; filing text never leaves it. Slower: about a minute per signal.")
    elif not os.getenv("GROQ_API_KEY"):
        st.error("GROQ_API_KEY is not set.")
    if not local_ok:
        st.caption("Local models appear here when Ollama is running (`ollama serve`).")
LLM = {"backend": BACKEND, "model": MODEL}

tab1, tab2, tab3 = st.tabs(["Signal Generator", "YoY Trend", "Portfolio Dashboard"])

# ════════════════════════════════════════════════════════════════
# TAB 1: SIGNAL GENERATOR
# ════════════════════════════════════════════════════════════════
with tab1:
    col_side, col_main = st.columns([1, 2])

    with col_side:
        st.subheader("Configuration")
        ticker = st.selectbox("Select Company", TICKERS, key="tab1_ticker")
        query = st.text_area("Analysis Query", value=DEFAULT_QUERY, height=100, key="tab1_query")
        run = st.button("Generate Signal", type="primary", use_container_width=True, key="tab1_run")
        st.caption("Data: SEC EDGAR 10-K filings (last 3 years). Uses the most recent filing.")

    with col_main:
        if not run:
            st.info("Select a company and click **Generate Signal** to begin.")
        else:
            with st.spinner(f"Analyzing {ticker}'s 10-K..."):
                signal = cached_signal(ticker, query, None, **LLM)

            if "error" in signal:
                st.error(signal["error"])
            else:
                risk = signal.get("risk_level", "N/A")
                n_sources = sum(s.get("type") == "retrieved_chunk" for s in signal.get("_meta", {}).get("sources", []))
                m1, m2, m3 = st.columns(3)
                m1.metric("Risk Level", risk)
                m2.metric("Filing Year", signal.get("year", "N/A"))
                m3.metric("Passages Retrieved", n_sources)
                show_not_answerable(signal)
                st.divider()

                left, right = st.columns(2)
                with left:
                    st.subheader("Key Risks")
                    for i, r in enumerate(signal.get("key_risks", []), 1):
                        st.warning(f"**{i}.** {r}")
                    st.subheader("Analyst Rationale")
                    st.info(signal.get("rationale", "N/A"))
                with right:
                    st.subheader("Red Flags")
                    flags = distinct_red_flags(signal)
                    if flags:
                        for f in flags:
                            st.error(f)
                    else:
                        st.success("No additional red flags beyond the key risks")
                    show_sources(signal)
                    with st.expander("Raw signal JSON"):
                        st.json(signal)
                st.caption(SENTIMENT_NOTE)

# ════════════════════════════════════════════════════════════════
# TAB 2: YoY TREND
# ════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Year-over-Year Risk Trend")
    t2_col1, t2_col2 = st.columns([1, 3])

    with t2_col1:
        trend_ticker = st.selectbox("Select Company", TICKERS, key="tab2_ticker")
        trend_query = st.text_area("Analysis Query", value=DEFAULT_QUERY, height=100, key="tab2_query")
        run_trend = st.button("Generate Trend", type="primary", use_container_width=True, key="tab2_run")

    with t2_col2:
        if not run_trend:
            st.info("Select a company and click **Generate Trend** to compare its filings year by year.")
        else:
            years = [f["year"] for f in get_filings_by_year(trend_ticker)]
            trend_data = []
            with st.spinner(f"Analyzing {trend_ticker} across {len(years)} filings..."):
                for y in years:
                    trend_data.append(cached_signal(trend_ticker, trend_query, y, **LLM))
            ok = [t for t in trend_data if "error" not in t]

            if not ok:
                st.error("No trend data found." + (f" ({trend_data[0]['error']})" if trend_data else ""))
            else:
                fig = go.Figure(go.Scatter(
                    x=[t["year"] for t in ok], y=[risk_to_score(t.get("risk_level")) for t in ok],
                    mode="lines+markers+text", text=[t.get("risk_level") for t in ok], textposition="top center",
                    line=dict(color="#EF553B", width=3), marker=dict(size=12), name="Risk level"))
                fig.update_layout(
                    title=f"{trend_ticker}: risk level by filing year",
                    xaxis=dict(title="Filing year", tickvals=[t["year"] for t in ok]),
                    yaxis=dict(title="Risk level", tickvals=[1, 2, 3], ticktext=["LOW", "MEDIUM", "HIGH"],
                               range=[0.5, 3.5]),
                    height=380, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="white"))
                st.plotly_chart(fig, use_container_width=True)

                st.subheader("Key Risks by Year")
                for t in trend_data:
                    if "error" in t:
                        st.error(f"{t.get('year', '')}: {t['error']}")
                        continue
                    with st.expander(f"**{t['year']}**: {t.get('risk_level')} risk"):
                        show_not_answerable(t)
                        for i, r in enumerate(t.get("key_risks", []), 1):
                            st.write(f"{i}. {r}")
                        st.caption(f"Rationale: {t.get('rationale', 'N/A')}")
                        show_sources(t)

# ════════════════════════════════════════════════════════════════
# TAB 3: PORTFOLIO DASHBOARD
# ════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("Portfolio-Level Risk Dashboard")
    st.caption("Analyze risk across several companies at once")
    col_left, col_right = st.columns([1, 3])

    with col_left:
        sector_filter = st.selectbox("Filter by Sector", ["All Sectors"] + list(SECTORS), key="tab3_sector")
        available = TICKERS if sector_filter == "All Sectors" else SECTORS[sector_filter]
        selected = st.multiselect("Select Portfolio Companies", available, default=available[:3],
                                  max_selections=8, key="tab3_tickers")
        portfolio_query = st.text_area("Analysis Query", value=DEFAULT_QUERY, height=100, key="tab3_query")
        run_portfolio = st.button("Analyze Portfolio", type="primary", use_container_width=True, key="tab3_run")

    with col_right:
        if not run_portfolio:
            st.info("Select companies and click **Analyze Portfolio**.")
        elif selected:
            signals = {}
            progress = st.progress(0, text="Analyzing portfolio...")
            for i, t in enumerate(selected):
                signals[t] = cached_signal(t, portfolio_query, None, **LLM)
                progress.progress((i + 1) / len(selected), text=f"Analyzed {i + 1}/{len(selected)} companies")
            progress.empty()

            ok = {t: s for t, s in signals.items() if "error" not in s}
            failed = {t: s["error"] for t, s in signals.items() if "error" in s}
            if failed:
                st.error("Could not analyze: " + "; ".join(f"{t} ({e})" for t, e in failed.items()))

            if ok:
                tickers = list(ok)
                levels = [ok[t].get("risk_level", "N/A") for t in tickers]
                fig = go.Figure(go.Heatmap(
                    z=[[risk_to_score(l) for l in levels]], x=tickers, y=["Risk level"],
                    text=[levels], texttemplate="%{text}", textfont={"size": 14, "color": "white"},
                    zmin=1, zmax=3, showscale=False,
                    colorscale=[[0.0, "#1a7f37"], [0.5, "#e67e22"], [1.0, "#c0392b"]]))  # LOW green -> HIGH red
                fig.update_layout(title="Portfolio Risk Heatmap", height=200, xaxis=dict(title="Company"),
                                  plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                                  font=dict(color="white"), margin=dict(t=40, b=40))
                st.plotly_chart(fig, use_container_width=True)

                st.divider()
                st.subheader("Portfolio Summary")
                cols = st.columns(4)
                for col, level in zip(cols, ["HIGH", "MEDIUM", "LOW"]):
                    names = [t for t in tickers if ok[t].get("risk_level") == level]
                    col.metric(f"{level.title()} Risk", len(names))
                    col.caption(", ".join(names) if names else "None")
                scores = [risk_to_score(l) for l in levels]
                cols[3].metric("Avg Risk Score", f"{sum(scores) / len(scores):.1f} / 3")

                st.divider()
                st.subheader("Company Breakdown")
                for t in tickers:
                    sig = ok[t]
                    level = sig.get("risk_level", "N/A")
                    with st.expander(f"**{t}**: :{RISK_COLORS.get(level, 'gray')}[{level}] risk · filed {sig.get('year')}"):
                        show_not_answerable(sig)
                        for i, r in enumerate(sig.get("key_risks", []), 1):
                            st.write(f"{i}. {r}")
                        st.caption(f"Rationale: {sig.get('rationale', 'N/A')}")
                        for f in distinct_red_flags(sig):
                            st.error(f)
                        show_sources(sig)
                st.caption(SENTIMENT_NOTE)
