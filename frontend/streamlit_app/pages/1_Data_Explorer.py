# frontend/streamlit-app/pages/1_Data_Explorer.py
"""
Data Explorer Page - Explore historical price data and features
"""

import os
from datetime import datetime, timedelta

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
from plotly.subplots import make_subplots

# Page config
st.set_page_config(
    page_title="Data Explorer - Lumina",
    page_icon="📊",
    layout="wide",
)

# API configuration
API_URL = os.getenv("API_URL", "http://localhost:8000")

# Custom CSS
st.markdown(
    """
<style>
    .metric-card {
        background-color: #262730;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 3px solid #1d77b4;
    }
    .feature-category {
        padding: 0.5rem;
        border-radius: 0.25rem;
        background-color: #1e1e1e;
        margin: 0.25rem 0;
    }
</style>
    """,
    unsafe_allow_html=True,
)

# Header
st.title("📊 Data Explorer")
st.markdown("Explore historical price data and engineered features")

# Sidebar controls
with st.sidebar:
    st.markdown("### 🎛️ Controls")

    # Ticker selection
    ticker = st.text_input(
        "Ticker Symbol",
        value=st.session_state.get("quick_ticker", "AAPL"),
        help="Enter stock ticker (e.g., AAPL, GOOGL, TSLA)",
    ).upper()

    # Date range
    st.markdown("### 📅 Date Range")
    end_date = st.date_input("End Date", datetime.now())

    date_range = st.selectbox(
        "Quick Select",
        ["1 Week", "1 Month", "3 Months", "6 Months", "1 Year", "2 Years", "5 Years", "Custom"],
    )

    if date_range == "Custom":
        start_date = st.date_input("Start Date", value=datetime.now() - timedelta(days=365))
    else:
        days_map = {
            "1 Week": 7,
            "1 Month": 30,
            "3 Months": 90,
            "6 Months": 180,
            "1 Year": 365,
            "2 Years": 730,
            "5 Years": 1825,
        }
        start_date = end_date - timedelta(days=days_map[date_range])

    # Interval selection
    interval = st.selectbox(
        "Interval",
        ["1d", "1h", "15m", "30m", "1wk", "1mo"],
        index=0,
        help="Data granularity",
    )

    # Feature options
    st.markdown("### 🔬 Feature Options")
    show_features = st.checkbox("Show Features", value=False)

    if show_features:
        feature_categories = st.multiselect(
            "Feature Categories",
            ["price", "volume", "volatility", "momentum", "trend", "statistical"],
            default=["price", "momentum"],
        )

    # Actions
    st.markdown("### ⚡ Actions")
    if st.button("🔄 Refresh Data", width="stretch"):
        st.rerun()

    if st.button("💾 Export Data", width="stretch"):
        st.session_state["export_requested"] = True

# Main content
try:
    # Fetch price data
    with st.spinner(f"Loading data for {ticker}..."):
        response = requests.get(
            f"{API_URL}/api/v2/data/{ticker}/prices",
            params={
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "interval": interval,
            },
            timeout=30,
        )

    if response.status_code == 200:
        data = response.json()
        df = pd.DataFrame(data["data"])

        if len(df) == 0:
            st.error(f"No data found for {ticker}")
            st.stop()

        df["time"] = pd.to_datetime(df["time"])
        df = df.sort_values(by="time")

        # Summary metrics
        st.markdown("### 📈 Summary")
        col1, col2, col3, col4, col5 = st.columns(5)

        current_price = df["close"].iloc[-1]
        prev_price = df["close"].iloc[0]
        price_change = current_price - prev_price
        price_change_pct = (price_change / prev_price) * 100

        with col1:
            st.metric(
                "Current Price",
                f"${current_price:.2f}",
                f"{price_change_pct:.2f}%",
            )

        with col2:
            st.metric("High", f"${df['high'].max():.2f}")

        with col3:
            st.metric("Low", f"${df['low'].min():.2f}")

        with col4:
            avg_volume = df["volume"].mean()
            st.metric("Avg Volume", f"{avg_volume / 1e6:.1f}M")

        with col5:
            volatility = df["close"].pct_change().std() * (252**0.5)
            st.metric("Volatility (Ann.)", f"{volatility * 100:.1f}%")

        # Price chart
        st.markdown("### 📊 Price Chart")

        chart_type = st.radio(
            "Chart Type",
            ["Candlestick", "Line", "Area"],
            horizontal=True,
        )

        fig = make_subplots(
            rows=2,
            cols=1,
            row_heights=[0.7, 0.3],
            vertical_spacing=0.05,
            subplot_titles=(f"{ticker} Price", "Volume"),
        )

        # Price chart
        if chart_type == "Candlestick":
            fig.add_trace(
                go.Candlestick(
                    x=df["time"],
                    open=df["open"],
                    high=df["high"],
                    low=df["low"],
                    close=df["close"],
                    name="OHLC",
                ),
                row=1,
                col=1,
            )
        elif chart_type == "Line":
            fig.add_trace(
                go.Scatter(
                    x=df["time"],
                    y=df["close"],
                    mode="lines",
                    name="Close",
                    line=dict(color="#1f77b4", width=2),
                ),
                row=1,
                col=1,
            )
        else:  # Area
            fig.add_trace(
                go.Scatter(
                    x=df["time"],
                    y=df["close"],
                    fill="tozeroy",
                    name="Close",
                    line=dict(color="#1f77b4"),
                ),
                row=1,
                col=1,
            )

        # Volume chart
        colors = [
            "green" if close >= open_ else "red" for close, open_ in zip(df["close"], df["open"])
        ]

        fig.add_trace(
            go.Bar(
                x=df["time"],
                y=df["volume"],
                marker_color=colors,
                opacity=0.5,
            ),
            row=2,
            col=1,
        )

        # Layout
        fig.update_layout(
            height=700,
            showlegend=False,
            hovermode="x unified",
            template="plotly_dark",
            xaxis_rangeslider_visible=False,
        )

        fig.update_xaxes(title_text="Date", row=2, col=1)
        fig.update_yaxes(title_text="Price ($)", row=1, col=1)
        fig.update_yaxes(title_text="Volume", row=2, col=1)

        st.plotly_chart(fig, width="stretch")

        # Statistics
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 📊 Price Statistics")

            returns = df["close"].pct_change().dropna()

            stats_df = pd.DataFrame(
                {
                    "Metric": [
                        "Mean",
                        "Median",
                        "Std Dev",
                        "Min",
                        "Max",
                        "Range",
                        "Skewness",
                        "Kurtosis",
                    ],
                    "Value": [
                        f"${df['close'].mean():.2f}",
                        f"${df['close'].median():.2f}",
                        f"${df['close'].std():.2f}",
                        f"${df['close'].min():.2f}",
                        f"${df['close'].max():.2f}",
                        f"${df['close'].max() - df['close'].min():.2f}",
                        f"{returns.skew():.4f}",
                        f"{returns.kurt():.4f}",
                    ],
                }
            )
            st.dataframe(stats_df, hide_index=True, width="stretch")

        with col2:
            st.markdown("### 📈 Returns Analysis")

            daily_return = returns.mean() * 100
            annual_return = (1 + returns.mean()) ** 252 - 1
            sharpe = (returns.mean() / returns.std()) * (252**0.5) if returns.std() > 0 else 0

            returns_df = pd.DataFrame(
                {
                    "Metric": [
                        "Daily Return",
                        "Annual Return",
                        "Volatility",
                        "Sharpe Ratio",
                        "Max Gain",
                        "Max Loss",
                        "Win Rate",
                        "Data Points",
                    ],
                    "Value": [
                        f"{daily_return:.4f}%",
                        f"{annual_return * 100:.2f}%",
                        f"{returns.std() * 100:.4f}%",
                        f"{sharpe:.2f}",
                        f"{returns.max() * 100:.2f}%",
                        f"{returns.min() * 100:.2f}%",
                        f"{(returns > 0).mean() * 100:.1f}%",
                        f"{len(df)}",
                    ],
                }
            )
            st.dataframe(returns_df, hide_index=True, width="stretch")

        # Returns distribution
        st.markdown("### 📊 Returns Distribution")

        fig_hist = px.histogram(
            returns * 100,
            nbins=50,
            title="Daily Returns Distribution",
            labels={"value": "Return (%)", "count": "Frequency"},
            template="plotly_dark",
        )

        fig_hist.add_vline(
            x=returns.mean() * 100,
            line_dash="dash",
            line_color="green",
            annotation_text="Mean",
        )

        st.plotly_chart(fig_hist, width="stretch")

        # Features section
        if show_features:
            st.markdown("---")
            st.markdown("### 🔬 Engineered Features")

            with st.spinner("Computing features..."):
                feature_response = requests.get(
                    f"{API_URL}/api/v2/data/{ticker}/features",
                    params={
                        "start_date": start_date.isoformat(),
                        "end_date": end_date.isoformat(),
                        "categories": ",".join(feature_categories),
                        "include_data": True,
                    },
                    timeout=60,
                )

            if feature_response.status_code == 200:
                feature_data = feature_response.json()

                # Show feature count by category
                st.markdown("#### Feature Categories")

                cols = st.columns(len(feature_categories))
                for idx, (col, cat) in enumerate(zip(cols, feature_categories)):
                    with col:
                        count = len(feature_data["categories"].get(cat, []))
                        st.metric(cat.title(), count)

                # Feature data
                if feature_data.get("data"):
                    features_df = pd.DataFrame(feature_data["data"])
                    features_df["time"] = pd.to_datetime(features_df["time"])

                    # Select features to plot
                    st.markdown("#### Feature Visualization")

                    available_features = [
                        f
                        for f in features_df.columns
                        if f not in ["time", "ticker", "source", "collected_at"]
                    ]

                    selected_features = st.multiselect(
                        "Select features to plot",
                        available_features[:20],  # Show first 20
                        default=available_features[:3]
                        if len(available_features) >= 3
                        else available_features,
                    )

                    if selected_features:
                        # Plot selected features
                        fig_features = make_subplots(
                            rows=len(selected_features),
                            cols=1,
                            subplot_titles=selected_features,
                            vertical_spacing=0.05,
                        )

                        for idx, feature in enumerate(selected_features, 1):
                            fig_features.add_trace(
                                go.Scatter(
                                    x=features_df["time"],
                                    y=features_df[feature],
                                    name=feature,
                                    mode="lines",
                                ),
                                row=idx,
                                col=1,
                            )

                        fig_features.update_layout(
                            height=300 * len(selected_features),
                            showlegend=False,
                            template="plotly_dark",
                        )

                        st.plotly_chart(fig_features, width="stretch")

                    # Feature correlation
                    if len(selected_features) > 1:
                        st.markdown("#### Feature Correlation")

                        corr_df = features_df[selected_features].corr()

                        fig_corr = px.imshow(
                            corr_df,
                            text_auto=True,
                            aspect="auto",
                            color_continuous_scale="RdBu_r",
                            title="Feature Correlation Matrix",
                            template="plotly_dark",
                        )

                        st.plotly_chart(fig_corr, width="stretch")

        # Raw data table
        with st.expander("📋 View Raw Data"):
            st.dataframe(
                df[["time", "open", "high", "low", "close", "volume"]].tail(100),
                width="stretch",
            )

        # Export functionality
        if st.session_state.get("export_requested", False):
            st.markdown("### 💾 Export Data")

            export_format = st.radio("Format", ["CSV", "JSON", "Excel"], horizontal=True)

            if export_format == "CSV":
                csv = df.to_csv(index=False)
                st.download_button(
                    "Download CSV",
                    csv,
                    f"{ticker}_data.csv",
                    "text/csv",
                    width="stretch",
                )
            elif export_format == "JSON":
                json_str = df.to_json(orient="records", date_format="iso")
                st.download_button(
                    "Download JSON",
                    json_str,
                    f"{ticker}_data.json",
                    "application/json",
                    width="stretch",
                )
            else:  # Excel
                st.info("Excel export requires pandas with openpyxl")

            st.session_state["export_requested"] = False

        # Company info
        with st.expander("ℹ️ Company Information"):
            with st.spinner("Loading company info..."):
                try:
                    info_response = requests.get(f"{API_URL}/api/v2/data/{ticker}/info", timeout=10)

                    if info_response.status_code == 200:
                        info = info_response.json()

                        col1, col2 = st.columns(2)

                        with col1:
                            st.markdown(f"**Name:** {info.get('name', 'N/A')}")
                            st.markdown(f"**Sector:** {info.get('sector', 'N/A')}")
                            st.markdown(f"**Industry:** {info.get('industry', 'N/A')}")
                            st.markdown(f"**Country:** {info.get('country', 'N/A')}")

                        with col2:
                            market_cap = info.get("market_cap", 0)
                            st.markdown(
                                f"**Market Cap:** ${market_cap / 1e9:.2f}B"
                                if market_cap
                                else "**Market Cap:** N/A"
                            )
                            st.markdown(f"**P/E Ratio:** {info.get('pe_ratio', 'N/A')}")
                            st.markdown(f"**Beta:** {info.get('beta', 'N/A')}")
                            div_yield = info.get("dividend_yield", 0)
                            st.markdown(
                                f"**Div Yield:** {div_yield * 100:.2f}%"
                                if div_yield
                                else "**Div Yield:** N/A"
                            )

                        if info.get("description"):
                            st.markdown("**Description:**")
                            st.markdown(info["description"][:500] + "...")
                    else:
                        st.warning("Company information not available")
                except Exception as e:
                    st.error(f"Error loading company info: {e}")

    elif response.status_code == 404:
        st.error(f"❌ Ticker '{ticker}' not found")
        st.info("💡 Try a different ticker symbol")
    else:
        st.error(f"❌ Error loading data: {response.status_code}")
        st.code(response.text)

except requests.exceptions.ConnectionError:
    st.error("❌ Cannot connect to API")
    st.info(f"Make sure the API is running at {API_URL}")
    st.code("cd backend && uvicorn api.main:app --reload")

except Exception as e:
    st.error(f"❌ An error occurred: {e}")
    st.exception(e)

# Footer
st.markdown("---")
st.markdown(
    f"**Data Explorer** | API: {API_URL} | "
    f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
)
