# src/app.py
import streamlit as st
import plotly.graph_objects as go
import sys, os
sys.path.append(os.path.dirname(__file__))
from rag import generate_signal
from trend import generate_trend, sentiment_to_score, risk_to_score
import plotly.graph_objects as go
import plotly.express as px

# Page Configuration
st.set_page_config(
    page_title="SEC Filing Analyzer",
    #page_icon="🏦",
    layout="wide"
)

st.title("SEC Filing Analyzer")
st.caption("Extracts investment signals from 10-K filings using RAG + LLM")
st.divider()

TICKERS = [
    "JPM", "BAC", "GS", "MS", "WFC", "C", "BLK", "AXP",
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA",
    "PGR", "TRV", "MET", "AIG", "PRU", "ALL",
    "JNJ", "UNH", "PFE", "ABBV", "CVS",
    "XOM", "CVX", "COP", "SLB",
    "WMT", "HD", "NKE", "MCD", "SBUX"
]

SECTORS = {
    "Finance & Banking": ["JPM", "BAC", "GS", "MS", "WFC", "C", "BLK", "AXP"],
    "Tech":              ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA"],
    "Insurance":         ["PGR", "TRV", "MET", "AIG", "PRU", "ALL"],
    "Healthcare":        ["JNJ", "UNH", "PFE", "ABBV", "CVS"],
    "Energy":            ["XOM", "CVX", "COP", "SLB"],
    "Consumer":          ["WMT", "HD", "NKE", "MCD", "SBUX"]
}


# Tabs
tab1, tab2, tab3 = st.tabs(["Signal Generator", "YoY Trend", "Portfolio Dashboard"])

# ════════════════════════════════════════════════════════════════
# TAB 1: SIGNAL GENERATOR
# ════════════════════════════════════════════════════════════════
with tab1:
    col_side, col_main = st.columns([1, 2])

    with col_side:
        st.subheader("Configuration")
        ticker = st.selectbox("Select Company", TICKERS, key="tab1_ticker")
        query  = st.text_area("Analysis Query",
                    value="What are the major risk factors and investment outlook?",
                    height=100, key="tab1_query")
        run    = st.button("Generate Signal", type="primary",
                    use_container_width=True, key="tab1_run")
        st.caption("Data sourced from SEC EDGAR 10-K filings (last 3 years)")

    with col_main:
        if not run:
            st.info("Select a company and click **Generate Signal** to begin.")

        if run:
            with st.spinner(f"Analyzing {ticker} SEC filings..."):
                signal = generate_signal(ticker, query)

            if "error" in signal and "raw_response" not in signal:
                st.error(f" {signal['error']}")
            else:
                risk      = signal.get("risk_level", "N/A")
                sent      = signal.get("sentiment", "N/A")
                n_risks   = len(signal.get("key_risks", []))
                risk_color = {"LOW": "green", "MEDIUM": "orange", "HIGH": "red"}.get(risk, "gray")
                sent_color = {"BULLISH": "green", "NEUTRAL": "gray", "BEARISH": "red"}.get(sent, "gray")

                m1, m2, m3 = st.columns(3)
                m1.metric("Risk Level", risk)
                m2.metric("Sentiment",  sent)
                m3.metric("Key Risks",  n_risks)
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
                    flags = signal.get("red_flags", [])
                    if flags:
                        for f in flags:
                            st.error(f" {f}")
                    else:
                        st.success("No major red flags detected")
                    st.subheader("Raw Signal JSON")
                    st.json(signal)

                st.divider()
                st.markdown(
                    f"### Overall Signal: :{sent_color}[**{sent}**] | "
                    f"Risk: :{risk_color}[**{risk}**]"
                )

# ════════════════════════════════════════════════════════════════
# TAB 2: YoY TREND
# ════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Year-over-Year Risk & Sentiment Trend")

    t2_col1, t2_col2 = st.columns([1, 3])

    with t2_col1:
        trend_ticker = st.selectbox("Select Company", TICKERS, key="tab2_ticker")
        trend_query  = st.text_area("Analysis Query",
                        value="What are the major risk factors and investment outlook?",
                        height=100, key="tab2_query")
        run_trend    = st.button("Generate Trend", type="primary",
                        use_container_width=True, key="tab2_run")

    with t2_col2:
        if not run_trend:
            st.info("Select a company and click **Generate Trend** to see YoY analysis.")

        if run_trend:
            with st.spinner(f"Analyzing {trend_ticker} across all filing years..."):
                trend_data = generate_trend(trend_ticker, trend_query)

            if not trend_data:
                st.error("No trend data found.")
            else:
                years      = [t["year"] for t in trend_data]
                sentiments = [sentiment_to_score(t.get("sentiment", "NEUTRAL")) for t in trend_data]
                risks      = [risk_to_score(t.get("risk_level", "MEDIUM")) for t in trend_data]
                sent_labels = [t.get("sentiment", "N/A") for t in trend_data]
                risk_labels = [t.get("risk_level", "N/A") for t in trend_data]

                # CHART
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=years, y=sentiments,
                    mode="lines+markers+text",
                    name="Sentiment",
                    text=sent_labels,
                    textposition="top center",
                    line=dict(color="#00CC96", width=3),
                    marker=dict(size=12)
                ))
                fig.add_trace(go.Scatter(
                    x=years, y=risks,
                    mode="lines+markers+text",
                    name="Risk Level",
                    text=risk_labels,
                    textposition="bottom center",
                    line=dict(color="#EF553B", width=3),
                    marker=dict(size=12),
                    yaxis="y2"
                ))
                fig.update_layout(
                    title=f"{trend_ticker} — Risk & Sentiment Trend",
                    xaxis=dict(title="Filing Year", tickvals=years),
                    yaxis=dict(
                        title="Sentiment",
                        tickvals=[-1, 0, 1],
                        ticktext=["BEARISH", "NEUTRAL", "BULLISH"]
                    ),
                    yaxis2=dict(
                        title="Risk Level",
                        tickvals=[1, 2, 3],
                        ticktext=["LOW", "MEDIUM", "HIGH"],
                        overlaying="y",
                        side="right"
                    ),
                    legend=dict(x=0.01, y=0.99),
                    height=420,
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="white")
                )
                st.plotly_chart(fig, use_container_width=True)

                # YoY RISK TABLE 
                st.subheader("Key Risks by Year")
                for t in trend_data:
                    with st.expander(f"**{t['year']}** — {t.get('sentiment')} | {t.get('risk_level')}"):
                        for i, r in enumerate(t.get("key_risks", []), 1):
                            st.write(f"{i}. {r}")
                        st.caption(f"Rationale: {t.get('rationale', 'N/A')}")

# ════════════════════════════════════════════════════════════════
# TAB 3: PORTFOLIO DASHBOARD
# ════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("Portfolio-Level Risk Dashboard")
    st.caption("Analyze risk across your entire portfolio at once")

    col_left, col_right = st.columns([1, 3])

    with col_left:
        sector_filter = st.selectbox(
            "Filter by Sector",
            ["All Sectors"] + list(SECTORS.keys()),
            key="tab3_sector"
        )
        
        # filter tickers by sector
        if sector_filter == "All Sectors":
            available_tickers = TICKERS
        else:
            available_tickers = SECTORS[sector_filter]
        
        selected_tickers = st.multiselect(
            "Select Portfolio Companies",
            available_tickers,
            default=available_tickers[:3],
            key="tab3_tickers"
        )
        portfolio_query = st.text_area(
            "Analysis Query",
            value="What are the major risk factors and investment outlook?",
            height=100,
            key="tab3_query"
        )
        run_portfolio = st.button(
            "Analyze Portfolio",
            type="primary",
            use_container_width=True,
            key="tab3_run"
        )

    with col_right:
        if not run_portfolio:
            st.info("Select companies and click **Analyze Portfolio**.")

        if run_portfolio and selected_tickers:
            all_signals = []
            progress    = st.progress(0, text="Analyzing portfolio...")

            for i, t in enumerate(selected_tickers):
                with st.spinner(f"Analyzing {t}..."):
                    sig = generate_signal(t, portfolio_query)
                    sig["ticker"] = t
                    all_signals.append(sig)
                progress.progress(
                    (i + 1) / len(selected_tickers),
                    text=f"Analyzed {i+1}/{len(selected_tickers)} companies"
                )

            progress.empty()

            # HEATMAP
            risk_map  = {"LOW": 1, "MEDIUM": 2, "HIGH": 3}
            sent_map  = {"BEARISH": -1, "NEUTRAL": 0, "BULLISH": 1}

            tickers_list  = [s["ticker"] for s in all_signals]
            risk_scores   = [risk_map.get(s.get("risk_level", "MEDIUM"), 2) for s in all_signals]
            sent_scores   = [sent_map.get(s.get("sentiment", "NEUTRAL"), 0) for s in all_signals]
            risk_labels   = [s.get("risk_level", "N/A") for s in all_signals]
            sent_labels   = [s.get("sentiment",  "N/A") for s in all_signals]

            # heatmap matrix: rows = metrics, cols = tickers
            fig_heat = go.Figure(data=go.Heatmap(
                z=[risk_scores, sent_scores],
                x=tickers_list,
                y=["Risk Level", "Sentiment"],
                text=[[risk_labels[i] for i in range(len(tickers_list))],
                      [sent_labels[i] for i in range(len(tickers_list))]],
                texttemplate="%{text}",
                textfont={"size": 14, "color": "white"},
                colorscale=[
                    [0.0, "#1a472a"],   # dark green (low risk / bullish)
                    [0.5, "#e67e22"],   # orange (medium)
                    [1.0, "#c0392b"],   # red (high risk / bearish)
                ],
                showscale=False
            ))
            fig_heat.update_layout(
                title="Portfolio Risk Heatmap",
                height=280,
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="white"),
                xaxis=dict(title="Company"),
                yaxis=dict(title="")
            )
            st.plotly_chart(fig_heat, use_container_width=True)

            # SUMMARY METRICS ROW 
            st.divider()
            st.subheader("Portfolio Summary")

            high_risk   = [s["ticker"] for s in all_signals if s.get("risk_level") == "HIGH"]
            medium_risk = [s["ticker"] for s in all_signals if s.get("risk_level") == "MEDIUM"]
            low_risk    = [s["ticker"] for s in all_signals if s.get("risk_level") == "LOW"]
            bearish     = [s["ticker"] for s in all_signals if s.get("sentiment")  == "BEARISH"]
            bullish     = [s["ticker"] for s in all_signals if s.get("sentiment")  == "BULLISH"]
            neutral     = [s["ticker"] for s in all_signals if s.get("sentiment")  == "NEUTRAL"]
            avg_risk   = round(sum(risk_scores) / len(risk_scores), 2)

            m1, m2, m3, m4 = st.columns(4)
            #m1.metric("High Risk Companies",  len(high_risk),  
            #          delta=", ".join(high_risk) if high_risk else "None")
            #m2.metric("Bearish Companies",    len(bearish),    
            #          delta=", ".join(bearish)   if bearish  else "None")
            #m3.metric("Bullish Companies",    len(bullish),    
            #          delta=", ".join(bullish)   if bullish  else "None")
            #m4.metric("Avg Risk Score",       f"{avg_risk}/3")

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("High Risk",    len(high_risk),
                      delta=", ".join(high_risk) if high_risk else "None")
            m2.metric("Medium Risk",  len(medium_risk),
                      delta=", ".join(medium_risk[:3]) if medium_risk else "None")
            m3.metric("Low Risk",     len(low_risk),
                      delta=", ".join(low_risk) if low_risk else "None")
            m4.metric("⚖️ Avg Risk Score", f"{avg_risk}/3")

            s1, s2, s3 = st.columns(3)
            s1.metric("Bullish", len(bullish),
                      delta=", ".join(bullish) if bullish else "None")
            s2.metric("Neutral", len(neutral),
                      delta=", ".join(neutral[:3]) if neutral else "None")
            s3.metric("Bearish", len(bearish),
                      delta=", ".join(bearish) if bearish else "None")

            # PER COMPANY BREAKDOWN
            st.divider()
            st.subheader("Company Breakdown")

            for sig in all_signals:
                risk      = sig.get("risk_level", "N/A")
                sent      = sig.get("sentiment",  "N/A")
                rc        = {"LOW": "green", "MEDIUM": "orange", "HIGH": "red"}.get(risk, "gray")
                sc        = {"BULLISH": "green", "NEUTRAL": "gray", "BEARISH": "red"}.get(sent, "gray")

                with st.expander(
                    f"**{sig['ticker']}** — :{rc}[{risk}] | :{sc}[{sent}]"
                ):
                    for i, r in enumerate(sig.get("key_risks", []), 1):
                        st.write(f"{i}. {r}")
                    st.caption(f"Rationale: {sig.get('rationale', 'N/A')}")
                    if sig.get("red_flags"):
                        for f in sig["red_flags"]:
                            st.error(f"{f}")