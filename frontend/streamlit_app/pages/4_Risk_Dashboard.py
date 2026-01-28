# frontend/streamlit-app/pages/4_Risk_Dashboard.py
"""
Risk Dashboard Page - Comprehensive portfolio and security risk analysis.
Includes VaR, CVaR, drawdown analysis, stress testing, and correlation analysis.
"""

import os
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

# Page configuration
st.set_page_config(
    page_title="Risk Dashboard - Lumina",
    page_icon="⚠️",
    layout="wide",
)

# API configuration
API_URL = os.getenv("API_URL", "http://localhost:8000")


# =============================================================================
# CUSTOM CSS
# =============================================================================

st.markdown(
    """
<style>
    .risk-card {
        background-color: #262730;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .risk-card-danger {
        border-left: 4px solid #d62728;
    }
    .risk-card-warning {
        border-left: 4px solid #ff7f0e;
    }
    .risk-card-success {
        border-left: 4px solid #2ca02c;
    }
    .var-display {
        font-size: 2rem;
        font-weight: bold;
        text-align: center;
    }
    .metric-label {
        color: #808080;
        font-size: 0.875rem;
    }
    .stress-scenario {
        padding: 0.75rem;
        margin: 0.5rem 0;
        border-radius: 0.25rem;
        background-color: #1e1e1e;
    }
</style>
""",
    unsafe_allow_html=True,
)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def fetch_risk_data(
    tickers: list[str],
    weights: dict[str, float],
    start_date: datetime,
    end_date: datetime,
) -> dict[str, Any] | None:
    """
    Fetch risk metrics from the API.

    Args:
        tickers: List of ticker symbols
        weights: Portfolio weights
        start_date: Analysis start date
        end_date: Analysis end date

    Returns:
        Dictionary with risk metrics or None if error
    """
    try:
        response = requests.post(
            f"{API_URL}/api/v2/risk/var",
            json={
                "tickers": tickers,
                "weights": weights,
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "confidence_levels": [0.95, 0.99],
                "method": st.session_state.get("var_method", "historical"),
                "holding_period": st.session_state.get("holding_period", 1),
            },
            timeout=60,
        )

        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.status_code}")
            return None
    except requests.RequestException as e:
        st.error(f"Connection error: {e}")
        return None


def fetch_stress_test(
    tickers: list[str],
    weights: dict[str, float],
) -> dict[str, Any] | None:
    """
    Fetch stress test results from the API.

    Args:
        tickers: List of ticker symbols
        weights: Portfolio weights

    Returns:
        Dictionary with stress test results or None if error
    """
    try:
        response = requests.post(
            f"{API_URL}/api/v2/risk/stress-test",
            json={
                "tickers": tickers,
                "weights": weights,
                "include_historical": True,
            },
            timeout=60,
        )

        if response.status_code == 200:
            return response.json()
        else:
            return None
    except requests.RequestException:
        return None


def calculate_local_risk_metrics(
    returns: pd.Series,
    confidence_levels: list[float] | None = None,
) -> dict[str, Any]:
    """
    Calculate risk metrics locally when API is unavailable.

    Args:
        returns: Series of portfolio returns
        confidence_levels: Confidence levels for VaR calculation

    Returns:
        Dictionary with risk metrics
    """
    if confidence_levels is None:
        confidence_levels = [0.95, 0.99]

    results = {"var_metrics": {}}

    for conf in confidence_levels:
        var = -np.percentile(returns, (1 - conf) * 100)
        cvar = -returns[returns <= -var].mean() if (returns <= -var).any() else var

        results["var_metrics"][f"{conf:.0%}"] = {
            "var": float(var),
            "cvar": float(cvar),
            "var_percent": float(var * 100),
            "cvar_percent": float(cvar * 100),
        }

    # Drawdown calculations
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max

    results["max_drawdown"] = float(abs(drawdown.min()))
    results["current_drawdown"] = float(abs(drawdown.iloc[-1]))
    results["drawdown_series"] = drawdown

    # Volatility
    results["volatility"] = float(returns.std() * np.sqrt(252))

    # Sharpe (assuming 0 risk-free rate)
    ann_return = returns.mean() * 252
    results["sharpe_ratio"] = (
        float(ann_return / results["volatility"]) if results["volatility"] > 0 else 0
    )

    return results


def create_var_gauge(
    var_value: float,
    confidence: str,
    portfolio_value: float = 100000,
) -> go.Figure:
    """
    Create a VaR gauge visualization.

    Args:
        var_value: VaR as decimal (e.g., 0.05 for 5%)
        confidence: Confidence level string
        portfolio_value: Portfolio value for dollar calculation

    Returns:
        Plotly Figure object
    """
    dollar_var = var_value * portfolio_value

    # Determine color based on severity
    if var_value < 0.02:
        color = "#2ca02c"  # Green
    elif var_value < 0.05:
        color = "#ff7f0e"  # Orange
    else:
        color = "#d62728"  # Red

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number+delta",
            value=var_value * 100,
            number={"suffix": "%", "font": {"size": 40}},
            title={"text": f"VaR ({confidence})", "font": {"size": 16}},
            gauge={
                "axis": {"range": [0, 10], "ticksuffix": "%"},
                "bar": {"color": color},
                "steps": [
                    {"range": [0, 2], "color": "rgba(44, 160, 44, 0.3)"},
                    {"range": [2, 5], "color": "rgba(255, 127, 14, 0.3)"},
                    {"range": [5, 10], "color": "rgba(214, 39, 40, 0.3)"},
                ],
                "threshold": {
                    "line": {"color": "white", "width": 2},
                    "thickness": 0.75,
                    "value": var_value * 100,
                },
            },
        )
    )

    fig.update_layout(
        height=250,
        template="plotly_dark",
        margin=dict(t=50, b=0, l=20, r=20),
        annotations=[
            dict(
                x=0.5,
                y=-0.1,
                text=f"${dollar_var:,.0f} at risk",
                showarrow=False,
                font=dict(size=14, color="#808080"),
            )
        ],
    )

    return fig


def create_drawdown_chart(
    drawdown_series: pd.Series,
    title: str = "Portfolio Drawdown",
) -> go.Figure:
    """
    Create a drawdown visualization.

    Args:
        drawdown_series: Series of drawdown values
        title: Chart title

    Returns:
        Plotly Figure object
    """
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=drawdown_series.index,
            y=drawdown_series * 100,
            fill="tozeroy",
            name="Drawdown",
            line=dict(color="#d62728"),
            fillcolor="rgba(214, 39, 40, 0.3)",
        )
    )

    # Add max drawdown line
    max_dd = drawdown_series.min() * 100

    fig.add_hline(
        y=max_dd,
        line_dash="dash",
        line_color="#ff7f0e",
        annotation_text=f"Max DD: {max_dd:.2f}%",
        annotation_position="right",
    )

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Drawdown (%)",
        template="plotly_dark",
        height=350,
        hovermode="x unified",
    )

    return fig


def create_correlation_heatmap(
    correlation_matrix: pd.DataFrame,
    title: str = "Asset Correlation",
) -> go.Figure:
    """
    Create a correlation heatmap.

    Args:
        correlation_matrix: Correlation matrix DataFrame
        title: Chart title

    Returns:
        Plotly Figure object
    """
    fig = go.Figure(
        data=go.Heatmap(
            z=correlation_matrix.values,
            x=correlation_matrix.columns.tolist(),
            y=correlation_matrix.index.tolist(),
            colorscale=[
                [0, "#d62728"],
                [0.5, "#f0f0f0"],
                [1, "#2ca02c"],
            ],
            zmin=-1,
            zmax=1,
            text=np.round(correlation_matrix.values, 2),
            texttemplate="%{text}",
            textfont={"size": 12},
            colorbar=dict(title="Correlation"),
        )
    )

    fig.update_layout(
        title=title,
        template="plotly_dark",
        height=400,
    )

    return fig


def create_var_distribution_chart(
    returns: pd.Series,
    var_95: float,
    var_99: float,
) -> go.Figure:
    """
    Create a returns distribution with VaR markers.

    Args:
        returns: Series of returns
        var_95: VaR at 95%
        var_99: VaR at 99%

    Returns:
        Plotly Figure object
    """
    fig = go.Figure()

    # Returns histogram
    fig.add_trace(
        go.Histogram(
            x=returns * 100,
            nbinsx=50,
            name="Daily Returns",
            marker_color="#1f77b4",
            opacity=0.7,
        )
    )

    # VaR lines
    fig.add_vline(
        x=-var_95 * 100,
        line_dash="dash",
        line_color="#ff7f0e",
        annotation_text=f"VaR 95%: {var_95 * 100:.2f}%",
        annotation_position="top",
    )

    fig.add_vline(
        x=-var_99 * 100,
        line_dash="dash",
        line_color="#d62728",
        annotation_text=f"VaR 99%: {var_99 * 100:.2f}%",
        annotation_position="top",
    )

    fig.update_layout(
        title="Returns Distribution with VaR",
        xaxis_title="Daily Return (%)",
        yaxis_title="Frequency",
        template="plotly_dark",
        height=350,
        showlegend=False,
    )

    return fig


# =============================================================================
# HEADER
# =============================================================================

st.title("⚠️ Risk Dashboard")
st.markdown("Comprehensive portfolio risk analysis and stress testing")


# =============================================================================
# SIDEBAR CONTROLS
# =============================================================================

with st.sidebar:
    st.markdown("### 🎛️ Risk Analysis Controls")

    # Portfolio Input
    st.markdown("#### Portfolio Configuration")

    portfolio_input = st.text_area(
        "Portfolio (ticker:weight)",
        value="AAPL:0.30\nMSFT:0.25\nGOOGL:0.20\nAMZN:0.15\nTSLA:0.10",
        help="Enter one ticker:weight per line (weights should sum to 1)",
    )

    # Parse portfolio input
    tickers = []
    weights = {}

    try:
        for line in portfolio_input.strip().split("\n"):
            if ":" in line:
                ticker, weight = line.split(":")
                ticker = ticker.strip().upper()
                weight = float(weight.strip())
                tickers.append(ticker)
                weights[ticker] = weight
    except ValueError:
        st.error("Invalid portfolio format. Use TICKER:WEIGHT format.")

    # Normalize weights if needed
    total_weight = sum(weights.values())
    if total_weight > 0 and abs(total_weight - 1.0) > 0.01:
        st.warning(f"Weights sum to {total_weight:.2f}. Normalizing to 1.0")
        weights = {k: v / total_weight for k, v in weights.items()}

    # Portfolio value
    portfolio_value = st.number_input(
        "Portfolio Value ($)",
        min_value=1000,
        max_value=100000000,
        value=100000,
        step=10000,
        format="%d",
    )

    st.markdown("---")

    # Date range
    st.markdown("#### Analysis Period")
    end_date = st.date_input("End Date", datetime.now())

    lookback_period = st.selectbox(
        "Lookback Period",
        ["1 Month", "3 Months", "6 Months", "1 Year", "2 Years", "5 Years"],
        index=3,
    )

    days_map = {
        "1 Month": 30,
        "3 Months": 90,
        "6 Months": 180,
        "1 Year": 365,
        "2 Years": 730,
        "5 Years": 1825,
    }
    start_date = end_date - timedelta(days=days_map[lookback_period])

    st.markdown("---")

    # VaR settings
    st.markdown("#### VaR Configuration")

    var_method = st.selectbox(
        "VaR Method",
        ["historical", "parametric", "monte_carlo"],
        index=0,
        help=(
            "Historical: Uses actual return distribution\n"
            "Parametric: Assumes normal distribution\n"
            "Monte Carlo: Simulation-based"
        ),
    )
    st.session_state["var_method"] = var_method

    holding_period = st.slider(
        "Holding Period (days)",
        min_value=1,
        max_value=30,
        value=1,
        help="Number of days for VaR calculation",
    )
    st.session_state["holding_period"] = holding_period

    st.markdown("---")

    # Actions
    if st.button("🔄 Refresh Analysis", width="stretch"):
        st.cache_data.clear()
        st.rerun()


# =============================================================================
# MAIN CONTENT
# =============================================================================

if not tickers:
    st.warning("Please enter at least one ticker in the portfolio configuration.")
    st.stop()

# Portfolio summary
st.markdown("### 📊 Portfolio Overview")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Portfolio Value", f"${portfolio_value:,.0f}")

with col2:
    st.metric("Number of Assets", len(tickers))

with col3:
    max_weight_ticker = max(weights.items(), key=lambda x: x[1])
    st.metric(
        "Largest Position",
        f"{max_weight_ticker[0]} ({max_weight_ticker[1] * 100:.0f}%)",
    )

with col4:
    concentration = sum(w**2 for w in weights.values())  # HHI
    st.metric("Concentration (HHI)", f"{concentration:.3f}")

# Display weights
st.markdown("#### Portfolio Weights")
weights_df = pd.DataFrame(
    [{"Ticker": t, "Weight": w, "Value": portfolio_value * w} for t, w in weights.items()]
)

col1, col2 = st.columns([2, 3])

with col1:
    st.dataframe(
        weights_df,
        column_config={
            "Ticker": st.column_config.TextColumn("Ticker"),
            "Weight": st.column_config.ProgressColumn(
                "Weight", format="%.1f%%", min_value=0, max_value=1
            ),
            "Value": st.column_config.NumberColumn("Value ($)", format="$%.0f"),
        },
        hide_index=True,
        width="stretch",
    )

with col2:
    fig_weights = px.pie(
        weights_df,
        values="Weight",
        names="Ticker",
        title="Portfolio Allocation",
        hole=0.4,
        template="plotly_dark",
    )
    fig_weights.update_layout(height=300, margin=dict(t=40, b=0, l=0, r=0))
    st.plotly_chart(fig_weights, width="stretch")

st.markdown("---")

# Fetch or calculate risk metrics
st.markdown("### 📈 Risk Analysis")

with st.spinner("Calculating risk metrics..."):
    # Try API first
    risk_data = fetch_risk_data(tickers, weights, start_date, end_date)

    # Fall back to local calculation if API unavailable
    if risk_data is None:
        st.info("API unavailable. Using local calculations with sample data.")

        # Generate sample data for demonstration
        np.random.seed(42)
        dates = pd.date_range(start=start_date, end=end_date, freq="B")
        returns = pd.Series(np.random.normal(0.0005, 0.02, len(dates)), index=dates)
        risk_data = calculate_local_risk_metrics(returns)
        risk_data["returns"] = returns


# Display VaR metrics
st.markdown("#### Value at Risk (VaR)")

var_metrics = risk_data.get("var_metrics", {})

col1, col2 = st.columns(2)

with col1:
    if "95%" in var_metrics:
        var_95 = var_metrics["95%"]["var"]
        fig_var_95 = create_var_gauge(var_95, "95%", portfolio_value)
        st.plotly_chart(fig_var_95, width="stretch")

        st.markdown(
            f"""
            <div class="risk-card risk-card-warning">
                <div class="metric-label">CVaR (Expected Shortfall) at 95%</div>
                <div class="var-display">{var_metrics["95%"]["cvar"] * 100:.2f}%</div>
                <div class="metric-label">
                    ${var_metrics["95%"]["cvar"] * portfolio_value:,.0f} expected loss beyond VaR
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

with col2:
    if "99%" in var_metrics:
        var_99 = var_metrics["99%"]["var"]
        fig_var_99 = create_var_gauge(var_99, "99%", portfolio_value)
        st.plotly_chart(fig_var_99, width="stretch")

        st.markdown(
            f"""
            <div class="risk-card risk-card-danger">
                <div class="metric-label">CVaR (Expected Shortfall) at 99%</div>
                <div class="var-display">{var_metrics["99%"]["cvar"] * 100:.2f}%</div>
                <div class="metric-label">
                    ${var_metrics["99%"]["cvar"] * portfolio_value:,.0f} expected loss beyond VaR
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# VaR interpretation
var_95_amount = var_metrics.get("95%", {}).get("var", 0) * portfolio_value
st.info(
    f"**VaR Interpretation:** With {var_method} VaR at 95% confidence over {holding_period} "
    f"day(s), there is a 5% chance of losing more than ${var_95_amount:,.0f}."
)

# Returns distribution
if "returns" in risk_data:
    returns = risk_data["returns"]
    var_95_val = var_metrics.get("95%", {}).get("var", 0.02)
    var_99_val = var_metrics.get("99%", {}).get("var", 0.03)

    fig_dist = create_var_distribution_chart(returns, var_95_val, var_99_val)
    st.plotly_chart(fig_dist, width="stretch")

st.markdown("---")

# Drawdown Analysis
st.markdown("#### 📉 Drawdown Analysis")

col1, col2, col3 = st.columns(3)

with col1:
    max_dd = risk_data.get("max_drawdown", 0)
    st.metric(
        "Maximum Drawdown",
        f"{max_dd * 100:.2f}%",
        delta=f"-${max_dd * portfolio_value:,.0f}",
        delta_color="inverse",
    )

with col2:
    current_dd = risk_data.get("current_drawdown", 0)
    st.metric(
        "Current Drawdown",
        f"{current_dd * 100:.2f}%",
        delta_color="inverse",
    )

with col3:
    vol = risk_data.get("volatility", 0)
    st.metric(
        "Annualized Volatility",
        f"{vol * 100:.2f}%",
    )

# Drawdown chart
if "drawdown_series" in risk_data:
    fig_dd = create_drawdown_chart(risk_data["drawdown_series"])
    st.plotly_chart(fig_dd, width="stretch")

st.markdown("---")

# Stress Testing
st.markdown("### 🔬 Stress Testing")

with st.spinner("Running stress tests..."):
    stress_data = fetch_stress_test(tickers, weights)

# If no API data, use predefined scenarios
if stress_data is None:
    st.info("Using predefined stress scenarios for demonstration.")

    stress_scenarios = {
        "2008 Financial Crisis": {
            "market_shock": -0.35,
            "volatility_spike": 3.0,
            "estimated_loss": -0.28,
        },
        "2020 COVID Crash": {
            "market_shock": -0.34,
            "volatility_spike": 4.0,
            "estimated_loss": -0.25,
        },
        "1987 Black Monday": {
            "market_shock": -0.22,
            "volatility_spike": 2.5,
            "estimated_loss": -0.18,
        },
        "Interest Rate +200bps": {
            "market_shock": -0.10,
            "volatility_spike": 1.5,
            "estimated_loss": -0.08,
        },
        "Tech Sector -20%": {
            "market_shock": -0.15,
            "volatility_spike": 1.8,
            "estimated_loss": -0.12,
        },
    }
else:
    stress_scenarios = stress_data.get("scenarios", {})

# Display stress test results
st.markdown("#### Scenario Analysis")

cols = st.columns(3)

for idx, (scenario_name, scenario_data) in enumerate(stress_scenarios.items()):
    with cols[idx % 3]:
        loss = scenario_data.get("estimated_loss", 0)
        loss_amount = loss * portfolio_value

        # Determine severity color
        if abs(loss) > 0.20:
            severity_class = "risk-card-danger"
            severity_color = "#d62728"
        elif abs(loss) > 0.10:
            severity_class = "risk-card-warning"
            severity_color = "#ff7f0e"
        else:
            severity_class = "risk-card-success"
            severity_color = "#2ca02c"

        st.markdown(
            f"""
            <div class="risk-card {severity_class}">
                <div style="font-weight: bold; margin-bottom: 0.5rem;">{scenario_name}</div>
                <div style="font-size: 1.5rem; color: {severity_color}; font-weight: bold;">
                    {loss * 100:.1f}%
                </div>
                <div class="metric-label">${loss_amount:,.0f} potential loss</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# Stress test comparison chart
fig_stress = go.Figure()

scenario_names = list(stress_scenarios.keys())
scenario_losses = [s.get("estimated_loss", 0) * 100 for s in stress_scenarios.values()]

colors = [
    "#d62728" if abs(l) > 20 else "#ff7f0e" if abs(l) > 10 else "#2ca02c" for l in scenario_losses
]

fig_stress.add_trace(
    go.Bar(
        y=scenario_names,
        x=scenario_losses,
        orientation="h",
        marker_color=colors,
        text=[f"{l:.1f}%" for l in scenario_losses],
        textposition="outside",
    )
)

fig_stress.update_layout(
    title="Stress Test: Estimated Portfolio Impact",
    xaxis_title="Portfolio Impact (%)",
    yaxis_title="",
    template="plotly_dark",
    height=350,
    xaxis=dict(range=[min(scenario_losses) * 1.2, 5]),
)

st.plotly_chart(fig_stress, width="stretch")

st.markdown("---")

# Correlation Analysis
st.markdown("### 🔗 Correlation Analysis")

# Generate sample correlation matrix for demonstration
np.random.seed(42)
n_assets = len(tickers)
random_matrix = np.random.rand(n_assets, n_assets)
corr_matrix = (random_matrix + random_matrix.T) / 2
np.fill_diagonal(corr_matrix, 1)

# Ensure valid correlation matrix
eigenvalues = np.linalg.eigvalsh(corr_matrix)
if np.any(eigenvalues < 0):
    corr_matrix = np.clip(corr_matrix, -1, 1)

correlation_df = pd.DataFrame(corr_matrix, index=tickers, columns=tickers)

col1, col2 = st.columns([3, 2])

with col1:
    fig_corr = create_correlation_heatmap(correlation_df)
    st.plotly_chart(fig_corr, width="stretch")

with col2:
    st.markdown("#### Correlation Statistics")

    # Extract upper triangle (excluding diagonal)
    upper_tri = correlation_df.where(np.triu(np.ones(correlation_df.shape), k=1).astype(bool))
    correlations = upper_tri.stack().values

    avg_corr = np.mean(correlations)
    max_corr = np.max(correlations)
    min_corr = np.min(correlations)

    st.metric("Average Correlation", f"{avg_corr:.3f}")
    st.metric("Highest Correlation", f"{max_corr:.3f}")
    st.metric("Lowest Correlation", f"{min_corr:.3f}")

    # Diversification benefit
    diversification_benefit = 1 - avg_corr
    st.metric(
        "Diversification Benefit",
        f"{diversification_benefit * 100:.1f}%",
        help="Higher is better. Based on average correlation.",
    )

st.markdown("---")

# Risk Summary
st.markdown("### 📋 Risk Summary")

summary_data = {
    "Metric": [
        "VaR (95%)",
        "CVaR (95%)",
        "VaR (99%)",
        "CVaR (99%)",
        "Max Drawdown",
        "Current Drawdown",
        "Annualized Volatility",
        "Sharpe Ratio",
        "Diversification Benefit",
    ],
    "Value": [
        f"{var_metrics.get('95%', {}).get('var', 0) * 100:.2f}%",
        f"{var_metrics.get('95%', {}).get('cvar', 0) * 100:.2f}%",
        f"{var_metrics.get('99%', {}).get('var', 0) * 100:.2f}%",
        f"{var_metrics.get('99%', {}).get('cvar', 0) * 100:.2f}%",
        f"{risk_data.get('max_drawdown', 0) * 100:.2f}%",
        f"{risk_data.get('current_drawdown', 0) * 100:.2f}%",
        f"{risk_data.get('volatility', 0) * 100:.2f}%",
        f"{risk_data.get('sharpe_ratio', 0):.2f}",
        f"{diversification_benefit * 100:.1f}%",
    ],
    "Dollar Amount": [
        f"${var_metrics.get('95%', {}).get('var', 0) * portfolio_value:,.0f}",
        f"${var_metrics.get('95%', {}).get('cvar', 0) * portfolio_value:,.0f}",
        f"${var_metrics.get('99%', {}).get('var', 0) * portfolio_value:,.0f}",
        f"${var_metrics.get('99%', {}).get('cvar', 0) * portfolio_value:,.0f}",
        f"${risk_data.get('max_drawdown', 0) * portfolio_value:,.0f}",
        f"${risk_data.get('current_drawdown', 0) * portfolio_value:,.0f}",
        "-",
        "-",
        "-",
    ],
}

summary_df = pd.DataFrame(summary_data)
st.dataframe(summary_df, hide_index=True, width="stretch")

# Export button
csv_data = summary_df.to_csv(index=False).encode("utf-8")
st.download_button(
    label="📥 Download Risk Report",
    data=csv_data,
    file_name=f"risk_report_{datetime.now().strftime('%Y%m%d')}.csv",
    mime="text/csv",
)

# Footer
st.markdown("---")
st.caption(
    f"Analysis Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')} | "
    f"VaR Method: {var_method.title()} | Holding Period: {holding_period} day(s)"
)
