# frontend/streamlit-app/app.py
"""
Lumina Quant Lab 2.0 - Main Streamlit Dashboard
"""

import os
from datetime import datetime, timedelta

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

# Page configuration
st.set_page_config(
    page_title="Lumina Quant Lab",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# API configuration
API_URL = os.getenv("API_URL", "http://localhost:8000")

# Custom CSS
st.markdown(
    """
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        background: linear-gradient(90deg, #1f77b4, #ff7f0e);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #262730;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 3px solid #1f77b4;
    }
    .stAlert {
        margin-top: 1rem;
    }
</style>
""",
    unsafe_allow_html=True,
)

# Header
st.markdown(
    '<div class="main-header">🚀 Lumina Quant Lab 2.0</div>', unsafe_allow_html=True
)
st.markdown("**Advanced Quantitative Trading Platform with AI/ML**")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.markdown("### Navigation")

    st.markdown("### Quick Actions")

    ticker_input = st.text_input("Ticker Symbol", "AAPL", key="sidebar_ticker")

    if st.button("🔍 Quick Analysis", width="stretch"):
        st.session_state["quick_ticker"] = ticker_input
        st.rerun()

    st.markdown("---")
    st.markdown("### System Status")

    # Health check
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        if response.status_code == 200:
            st.success("✅ API Online")
        else:
            st.error("❌ API Error")
    except:
        st.error("❌ API Offline")

# Main content
st.header("Welcome to Lumina Quant Lab")

# Overview metrics
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Available Tickers", "500+", "+50")
with col2:
    st.metric("ML Models", "3", "+1")
with col3:
    st.metric("Features", "100+", "✓")
with col4:
    st.metric("Backtests Run", "1,234", "+89")

st.markdown("---")

# Quick start guide
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📚 Quick Start Guide")

    st.markdown("""
    ### Getting Started

    1. **Data Explorer** - Explore historical price data and 100+ technical indicators
    2. **ML Models** - Train LSTM, Transformer, or XGBoost models for price prediction
    3. **Backtesting** - Test your strategies on historical data
    4. **Risk Analysis** - Analyze portfolio risk with VaR, CVaR, and more
    5. **Sentiment** - Track market sentiment from news and social media

    ### Key Features

    - ✅ Real-time data collection from multiple sources
    - ✅ Advanced ML models with attention mechanisms
    - ✅ Professional backtesting engine
    - ✅ Comprehensive risk analytics
    - ✅ Multi-source sentiment analysis
    """)

with col2:
    st.subheader("🎯 Recent Activity")

    st.info("**Latest Backtest**\nMomentum Strategy\nSharpe: 1.85")
    st.success("**Model Trained**\nAAPL LSTM\nVal Loss: 2.34")
    st.warning("**Risk Alert**\nPortfolio VaR\nIncreased 5%")

st.markdown("---")

# Sample chart
st.subheader("📊 Market Overview")

try:
    # Fetch data for sample tickers
    tickers = ["AAPL", "MSFT", "GOOGL"]

    fig = go.Figure()

    for ticker in tickers:
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=90)

            response = requests.get(
                f"{API_URL}/api/v2/data/{ticker}/prices",
                params={
                    "start_date": start_date.isoformat(),
                    "end_date": end_date.isoformat(),
                },
                timeout=10,
            )

            if response.status_code == 200:
                data = response.json()
                df = pd.DataFrame(data["data"])
                df["time"] = pd.to_datetime(df["time"])

                # Normalize to 100 for comparison
                df["normalized"] = (df["close"] / df["close"].iloc[0]) * 100

                fig.add_trace(
                    go.Scatter(
                        x=df["time"], y=df["normalized"], name=ticker, mode="lines"
                    )
                )
        except:
            pass

    fig.update_layout(
        title="90-Day Performance (Normalized to 100)",
        xaxis_title="Date",
        yaxis_title="Normalized Price",
        hovermode="x unified",
        template="plotly_dark",
        height=400,
    )

    st.plotly_chart(fig, width="stretch")

except Exception as e:
    st.error(f"Error loading market data: {e}")
    st.info("Make sure the API is running and accessible")

# Footer
st.markdown("---")
col1, col2, col3 = st.columns([1, 1, 1])

with col1:
    st.markdown("**Lumina Quant Lab v2.0**")
with col2:
    st.markdown("📚 [Documentation](http://localhost:8000/docs)")
with col3:
    st.markdown("💬 [GitHub](https://github.com/franjavi-upct-es/lumina_project/tree/main)")

# Session state management
if "quick_ticker" in st.session_state:
    st.sidebar.success(f"Analyzing {st.session_state['quick_ticker']}...")
