import streamlit as st
import requests
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import json, os

# ── Config ────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NIFTY50 · Stoic-HMM Regime Detector",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

API_URL = os.getenv("API_URL", "http://localhost:8000")

REGIME_COLORS = {
    "DESIRE":   "#D4AC0D",
    "FEAR":     "#C0392B",
    "PLEASURE": "#1A7A4A",
    "DISTRESS": "#6C3483",
}
REGIME_ICONS = {"DESIRE": "🔥", "FEAR": "⚠️", "PLEASURE": "✨", "DISTRESS": "📉"}

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .stApp { background-color: #000000; color: #F5F0E8; }
    .main .block-container { padding-top: 1rem; }
    [data-testid="stSidebar"] { background-color: #0D1117; }
    .metric-card {
        background: #0D1117;
        border: 1px solid #1E2A3A;
        border-radius: 10px;
        padding: 16px 20px;
        text-align: center;
    }
    .regime-badge {
        display: inline-block;
        padding: 6px 18px;
        border-radius: 20px;
        font-weight: bold;
        font-size: 1.1rem;
        letter-spacing: 2px;
    }
    h1, h2, h3 { color: #C9A84C !important; }
    .stSlider > div > div { background-color: #1E2A3A; }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("# ⚖️ NIFTY 50 · Stoic-HMM Regime Detector")
st.markdown("*Detecting market emotional regimes using Wundt's Valence-Arousal model and Hidden Markov Machines*")
st.divider()

# ── Sidebar: Manual Input ─────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📊 Input Market Signals")
    st.markdown("Adjust the sliders to match current NIFTY 50 fundamentals:")
    st.divider()

    pe_median  = st.slider("Median PE Ratio",        10.0, 80.0, 31.0, 0.5,
                            help="Current median P/E across NIFTY 50")
    eps_growth = st.slider("EPS Growth Rate",        -0.3,  0.5,  0.10, 0.01,
                            format="%.2f",
                            help="YoY EPS growth (e.g. 0.12 = 12%)")
    de_median  = st.slider("Median Debt/Equity",      1.0, 12.0,  6.5, 0.1,
                            help="Median D/E ratio across NIFTY 50")
    pat_change = st.slider("PAT Margin Change (pp)", -10.0, 10.0,  0.0, 0.5,
                            help="YoY change in PAT margin in percentage points")
    roe_median = st.slider("Median ROE (%)",          5.0, 35.0, 16.0, 0.5,
                            help="Median Return on Equity")
    ebitda_vol = st.slider("EBITDA Margin Volatility", 1.0, 20.0, 8.0, 0.5,
                            help="Cross-sectional std dev of EBITDA margins")
    mc_growth  = st.slider("MCap Growth Rate",       -0.3,  0.5,  0.10, 0.01,
                            format="%.2f",
                            help="YoY total market cap growth")
    roe_change = st.slider("ROE Change (pp)",        -10.0, 10.0,  0.0, 0.5,
                            help="YoY change in median ROE")

    st.divider()
    run_btn = st.button("🔍 DETECT REGIME", use_container_width=True, type="primary")

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🎯 Live Detection", "📜 Historical Regimes", "🗺️ Wundt Plane"])

# ── TAB 1: Live Detection ─────────────────────────────────────────────────────
with tab1:
    if run_btn:
        payload = {
            "pe_median": pe_median, "eps_growth": eps_growth,
            "de_median": de_median, "pat_change": pat_change,
            "roe_median": roe_median, "ebitda_vol": ebitda_vol,
            "mc_growth": mc_growth, "roe_change": roe_change,
        }
        try:
            resp = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
            result = resp.json()

            regime = result["regime"]
            color  = REGIME_COLORS[regime]
            icon   = REGIME_ICONS[regime]

            # Big regime display
            st.markdown(f"""
            <div style='text-align:center; padding:30px; background:#0D1117;
                        border:2px solid {color}; border-radius:16px; margin-bottom:20px'>
                <div style='font-size:3rem'>{icon}</div>
                <div class='regime-badge' style='background:{color}22; color:{color};
                     border:1px solid {color}; margin-top:8px'>
                    {regime}
                </div>
                <div style='color:#F5F0E8; margin-top:12px; font-size:0.95rem'>
                    {result['description']}
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Metrics row
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.metric("Valence", f"{result['valence']:+.3f}",
                          delta="Positive" if result['valence'] > 0 else "Negative")
            with c2:
                st.metric("Arousal", f"{result['arousal']:+.3f}",
                          delta="High" if result['arousal'] > 0 else "Low")
            with c3:
                st.metric("Confidence", f"{result['confidence']*100:.1f}%")
            with c4:
                dominant = max(result['probabilities'], key=result['probabilities'].get)
                st.metric("Top Regime", dominant)

            # Probability bar chart
            probs = result["probabilities"]
            fig_bar = go.Figure(go.Bar(
                x=list(probs.keys()),
                y=[v*100 for v in probs.values()],
                marker_color=[REGIME_COLORS[k] for k in probs.keys()],
                text=[f"{v*100:.1f}%" for v in probs.values()],
                textposition='auto',
            ))
            fig_bar.update_layout(
                title="Regime Posterior Probabilities",
                paper_bgcolor="#000000", plot_bgcolor="#0D1117",
                font_color="#F5F0E8", title_font_color="#C9A84C",
                yaxis_title="Probability (%)", xaxis_title="",
                showlegend=False, height=300,
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        except Exception as e:
            st.error(f"API Error: {e}. Make sure the backend is running at {API_URL}")
    else:
        st.info("👈 Adjust the sliders in the sidebar and click **DETECT REGIME**")

# ── TAB 2: Historical Regimes ─────────────────────────────────────────────────
with tab2:
    try:
        hist_resp = requests.get(f"{API_URL}/history", timeout=10)
        hist_data = hist_resp.json()["data"]
        df = pd.DataFrame(hist_data)

        # Timeline chart
        fig_timeline = go.Figure()
        for regime, color in REGIME_COLORS.items():
            mask = df["regime"] == regime
            if mask.any():
                fig_timeline.add_trace(go.Bar(
                    x=df[mask]["year"], y=[1]*mask.sum(),
                    name=regime, marker_color=color,
                    text=df[mask]["regime"], textposition='inside',
                    hovertemplate="<b>%{x}</b><br>Regime: %{text}<br>Valence: %{customdata[0]}<br>Arousal: %{customdata[1]}",
                    customdata=df[mask][["valence","arousal"]].values,
                ))
        fig_timeline.update_layout(
            barmode='stack', title="Regime Timeline — NIFTY 50 (FY2015–FY2024)",
            paper_bgcolor="#000000", plot_bgcolor="#0D1117",
            font_color="#F5F0E8", title_font_color="#C9A84C",
            showlegend=True, height=200, yaxis_visible=False,
            legend=dict(bgcolor="#0D1117", bordercolor="#1E2A3A"),
        )
        st.plotly_chart(fig_timeline, use_container_width=True)

        # Probabilities stacked area
        fig_area = go.Figure()
        for regime, color in REGIME_COLORS.items():
            col = f"{regime.lower()}_prob"
            fig_area.add_trace(go.Scatter(
                x=df["year"], y=df[col]*100,
                mode='lines', stackgroup='one',
                name=regime, line_color=color,
                fillcolor=color.replace("#","rgba(").rstrip(")") + ",0.4)" if False else color,
            ))
        fig_area.update_layout(
            title="Posterior Probability by Year",
            paper_bgcolor="#000000", plot_bgcolor="#0D1117",
            font_color="#F5F0E8", title_font_color="#C9A84C",
            yaxis_title="Probability (%)", height=350,
            legend=dict(bgcolor="#0D1117", bordercolor="#1E2A3A"),
        )
        st.plotly_chart(fig_area, use_container_width=True)

        # Data table
        st.markdown("#### 📋 Full Historical Data")
        display_df = df[["year","regime","valence","arousal","confidence"]].copy()
        display_df.columns = ["Year","Regime","Valence","Arousal","Confidence"]
        display_df["Confidence"] = display_df["Confidence"].apply(lambda x: f"{x*100:.1f}%")
        st.dataframe(display_df, use_container_width=True, hide_index=True)

    except Exception as e:
        st.error(f"Could not load history from API: {e}")

# ── TAB 3: Wundt Plane ────────────────────────────────────────────────────────
with tab3:
    try:
        hist_resp = requests.get(f"{API_URL}/history", timeout=10)
        hist_data = hist_resp.json()["data"]
        df = pd.DataFrame(hist_data)

        fig_wundt = go.Figure()

        # Quadrant shading
        for x0,x1,y0,y1,name,color in [
            (0,3.5,0,3,   "DESIRE",   REGIME_COLORS["DESIRE"]),
            (-3.5,0,0,3,  "FEAR",     REGIME_COLORS["FEAR"]),
            (-3.5,0,-3,0, "DISTRESS", REGIME_COLORS["DISTRESS"]),
            (0,3.5,-3,0,  "PLEASURE", REGIME_COLORS["PLEASURE"]),
        ]:
            fig_wundt.add_shape(type="rect", x0=x0,x1=x1,y0=y0,y1=y1,
                fillcolor=color, opacity=0.08, line_width=0)
            fig_wundt.add_annotation(x=(x0+x1)/2, y=(y0+y1)/2, text=name,
                font=dict(color=color, size=13), showarrow=False, opacity=0.6)

        # Trajectory arrows
        for i in range(len(df)-1):
            c = REGIME_COLORS[df.iloc[i]["regime"]]
            fig_wundt.add_annotation(
                x=df.iloc[i+1]["valence"], y=df.iloc[i+1]["arousal"],
                ax=df.iloc[i]["valence"],  ay=df.iloc[i]["arousal"],
                xref='x', yref='y', axref='x', ayref='y',
                arrowhead=2, arrowcolor=c, arrowwidth=2,
            )

        # Points
        for _, row in df.iterrows():
            c = REGIME_COLORS[row["regime"]]
            fig_wundt.add_trace(go.Scatter(
                x=[row["valence"]], y=[row["arousal"]],
                mode="markers+text",
                text=[row["year"]], textposition="top center",
                marker=dict(size=14, color=c, line=dict(color="white", width=1.5)),
                hovertemplate=f"<b>{row['year']}</b><br>Regime: {row['regime']}<br>"
                              f"Valence: {row['valence']}<br>Arousal: {row['arousal']}",
                showlegend=False,
            ))

        # Axis lines
        fig_wundt.add_hline(y=0, line_color="#C9A84C", line_width=0.8, opacity=0.5)
        fig_wundt.add_vline(x=0, line_color="#C9A84C", line_width=0.8, opacity=0.5)

        fig_wundt.update_layout(
            title="Wundt Valence-Arousal Plane — NIFTY 50 Emotional Trajectory",
            paper_bgcolor="#000000", plot_bgcolor="#000000",
            font_color="#F5F0E8", title_font_color="#C9A84C",
            xaxis=dict(title="VALENCE  (Positive ↔ Negative)", range=[-3.5,3.5],
                       gridcolor="#1E2A3A", zerolinecolor="#C9A84C"),
            yaxis=dict(title="AROUSAL  (High ↔ Low)", range=[-3,3],
                       gridcolor="#1E2A3A", zerolinecolor="#C9A84C"),
            height=600, showlegend=False,
        )
        st.plotly_chart(fig_wundt, use_container_width=True)

        st.markdown("""
        **How to read this chart:**
        - Each dot = one fiscal year on the Wundt emotional plane
        - Arrows = direction of emotional drift year-over-year
        - **Top-right** (Desire) = PE exuberance + EPS momentum
        - **Top-left** (Fear) = leverage stress + profit decline
        - **Bottom-right** (Pleasure) = quality complacency
        - **Bottom-left** (Distress) = grinding erosion
        """)

    except Exception as e:
        st.error(f"Could not load Wundt plane data: {e}")
