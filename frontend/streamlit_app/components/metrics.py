# frontend/streamlit-app/components/metrics.py
"""
Reusable metric components for Lumina Quant Lab dashboard.
Provides standardized metric displays and KPI visualizations.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
import streamlit as st

# =============================================================================
# DATA CLASSES
# =============================================================================


@dataclass
class MetricConfig:
    """Configuration for a single metric display."""

    label: str
    value: float | str
    delta: float | str | None = None
    delta_color: str = "normal"  # 'normal', 'inverse', 'off'
    format_str: str = "{:.2f}"
    prefix: str = ""
    suffix: str = ""
    help_text: str | None = None


@dataclass
class RiskMetrics:
    """Container for risk-related metrics."""

    var_95: float
    var_99: float
    cvar_95: float
    cvar_99: float
    max_drawdown: float
    current_drawdown: float
    volatility: float
    beta: float | None = None
    sharpe: float | None = None
    sortino: float | None = None


@dataclass
class PerformanceMetrics:
    """Container for performance-related metrics."""

    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    num_trades: int


@dataclass
class SentimentMetrics:
    """Container for sentiment-related metrics."""

    overall_score: float
    news_score: float | None = None
    reddit_score: float | None = None
    twitter_score: float | None = None
    volume: int | None = None
    trend: str | None = None  # 'bullish', 'bearish', 'neutral'


# =============================================================================
# FORMATTING UTILITIES
# =============================================================================


def format_currency(value: float, decimals: int = 2) -> str:
    """
    Format a value as currency.

    Args:
        value: Numeric value
        decimals: Number of decimal places

    Returns:
        Formatted currency string
    """
    if abs(value) >= 1e9:
        return f"${value / 1e9:.{decimals}f}B"
    elif abs(value) >= 1e6:
        return f"${value / 1e6:.{decimals}f}M"
    elif abs(value) >= 1e3:
        return f"${value / 1e3:.{decimals}f}K"
    else:
        return f"${value:,.{decimals}f}"


def format_percentage(value: float, decimals: int = 2) -> str:
    """
    Format a value as percentage.

    Args:
        value: Numeric value (0.15 for 15%)
        decimals: Number of decimal places

    Returns:
        Formatted percentage string
    """
    return f"{value * 100:.{decimals}f}%"


def format_number(value: float, decimals: int = 2) -> str:
    """
    Format a large number with abbreviations.

    Args:
        value: Numeric value
        decimals: Number of decimal places

    Returns:
        Formatted number string
    """
    if abs(value) >= 1e9:
        return f"{value / 1e9:.{decimals}f}B"
    elif abs(value) >= 1e6:
        return f"{value / 1e6:.{decimals}f}M"
    elif abs(value) >= 1e3:
        return f"{value / 1e3:.{decimals}f}K"
    else:
        return f"{value:,.{decimals}f}"


def format_ratio(value: float, decimals: int = 2) -> str:
    """
    Format a ratio value.

    Args:
        value: Numeric ratio value
        decimals: Number of decimal places

    Returns:
        Formatted ratio string
    """
    return f"{value:.{decimals}f}"


# =============================================================================
# METRIC CARD COMPONENTS
# =============================================================================


def render_metric_card(
    label: str,
    value: str | float,
    delta: str | float | None = None,
    delta_color: str = "normal",
    help_text: str | None = None,
) -> None:
    """
    Render a single metric card using Streamlit's native metric component.

    Args:
        label: Metric label
        value: Metric value
        delta: Optional delta/change value
        delta_color: Color mode for delta ('normal', 'inverse', 'off')
        help_text: Optional help tooltip text
    """
    st.metric(
        label=label,
        value=value,
        delta=delta,
        delta_color=delta_color,
        help=help_text,
    )


def render_metric_row(metrics: list[MetricConfig], columns: int | None = None) -> None:
    """
    Render a row of metrics in columns.

    Args:
        metrics: List of MetricConfig objects
        columns: Number of columns (defaults to len(metrics))
    """
    num_cols = columns or len(metrics)
    cols = st.columns(num_cols)

    for idx, metric in enumerate(metrics):
        with cols[idx % num_cols]:
            # Format the value
            if isinstance(metric.value, (int, float)):
                formatted_value = (
                    f"{metric.prefix}{metric.format_str.format(metric.value)}{metric.suffix}"
                )
            else:
                formatted_value = f"{metric.prefix}{metric.value}{metric.suffix}"

            # Format the delta
            formatted_delta = None
            if metric.delta is not None:
                if isinstance(metric.delta, (int, float)):
                    formatted_delta = f"{metric.delta:+.2f}"
                else:
                    formatted_delta = str(metric.delta)

            render_metric_card(
                label=metric.label,
                value=formatted_value,
                delta=formatted_delta,
                delta_color=metric.delta_color,
                help_text=metric.help_text,
            )


def render_styled_metric_card(
    label: str,
    value: str,
    delta: str | None = None,
    color: str = "#1f77b4",
    icon: str | None = None,
) -> None:
    """
    Render a styled metric card with custom CSS.

    Args:
        label: Metric label
        value: Metric value
        delta: Optional delta string
        color: Border/accent color
        icon: Optional emoji icon
    """
    delta_html = (
        f'<span style="color: {"#2ca02c" if delta and delta.startswith("+") else "#d62728"}">{delta}</span>'
        if delta
        else ""
    )
    icon_html = f'<span style="font-size: 1.5rem; margin-right: 8px;">{icon}</span>' if icon else ""

    st.markdown(
        f"""
        <div style="
            background-color: #262730;
            padding: 1rem;
            border-radius: 0.5rem;
            border-left: 4px solid {color};
            margin-bottom: 0.5rem;
        ">
            <div style="color: #808080; font-size: 0.875rem; margin-bottom: 0.25rem;">
                {icon_html}{label}
            </div>
            <div style="font-size: 1.5rem; font-weight: bold; color: white;">
                {value} {delta_html}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# =============================================================================
# RISK METRICS DISPLAY
# =============================================================================


def render_risk_metrics(
    metrics: RiskMetrics,
    portfolio_value: float | None = None,
) -> None:
    """
    Render a comprehensive risk metrics display.

    Args:
        metrics: RiskMetrics object
        portfolio_value: Optional portfolio value for dollar amounts
    """
    st.markdown("### 📊 Risk Metrics")

    # VaR Section
    st.markdown("#### Value at Risk (VaR)")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        var_95_pct = format_percentage(metrics.var_95)
        var_95_amount = (
            format_currency(portfolio_value * metrics.var_95) if portfolio_value else "N/A"
        )
        render_metric_card("VaR (95%)", var_95_pct, help_text=f"Dollar amount: {var_95_amount}")

    with col2:
        var_99_pct = format_percentage(metrics.var_99)
        var_99_amount = (
            format_currency(portfolio_value * metrics.var_99) if portfolio_value else "N/A"
        )
        render_metric_card("VaR (99%)", var_99_pct, help_text=f"Dollar amount: {var_99_amount}")

    with col3:
        cvar_95_pct = format_percentage(metrics.cvar_95)
        render_metric_card("CVaR (95%)", cvar_95_pct, help_text="Expected Shortfall at 95%")

    with col4:
        cvar_99_pct = format_percentage(metrics.cvar_99)
        render_metric_card("CVaR (99%)", cvar_99_pct, help_text="Expected Shortfall at 99%")

    # Drawdown Section
    st.markdown("#### Drawdown Analysis")
    col1, col2, col3 = st.columns(3)

    with col1:
        render_metric_card(
            "Max Drawdown",
            format_percentage(metrics.max_drawdown),
            delta_color="inverse",
            help_text="Maximum peak-to-trough decline",
        )

    with col2:
        render_metric_card(
            "Current Drawdown",
            format_percentage(metrics.current_drawdown),
            delta_color="inverse",
            help_text="Current distance from peak",
        )

    with col3:
        render_metric_card(
            "Volatility (Ann.)",
            format_percentage(metrics.volatility),
            help_text="Annualized standard deviation of returns",
        )

    # Risk-Adjusted Metrics
    if any([metrics.beta, metrics.sharpe, metrics.sortino]):
        st.markdown("#### Risk-Adjusted Metrics")
        col1, col2, col3 = st.columns(3)

        with col1:
            if metrics.beta is not None:
                render_metric_card(
                    "Beta", format_ratio(metrics.beta), help_text="Market sensitivity"
                )

        with col2:
            if metrics.sharpe is not None:
                render_metric_card(
                    "Sharpe Ratio",
                    format_ratio(metrics.sharpe),
                    help_text="Return per unit of risk",
                )

        with col3:
            if metrics.sortino is not None:
                render_metric_card(
                    "Sortino Ratio",
                    format_ratio(metrics.sortino),
                    help_text="Return per unit of downside risk",
                )


def render_var_summary(
    var_metrics: dict[str, dict[str, float]],
    portfolio_value: float = 100000,
) -> None:
    """
    Render a VaR summary with multiple confidence levels.

    Args:
        var_metrics: Dict of confidence_level -> {var, cvar, ...}
        portfolio_value: Portfolio value for dollar calculations
    """
    st.markdown("### 💰 Value at Risk Summary")

    cols = st.columns(len(var_metrics))

    for idx, (confidence, metrics) in enumerate(var_metrics.items()):
        with cols[idx]:
            var_pct = metrics.get("var", 0) * 100
            cvar_pct = metrics.get("cvar", 0) * 100
            var_amount = portfolio_value * metrics.get("var", 0)

            st.markdown(
                f"""
                <div style="
                    background-color: #1e1e1e;
                    padding: 1rem;
                    border-radius: 0.5rem;
                    text-align: center;
                    border: 1px solid #333;
                ">
                    <div style="font-size: 1.25rem; font-weight: bold; color: #1f77b4;">
                        {confidence} Confidence
                    </div>
                    <hr style="border-color: #333;">
                    <div style="margin: 0.5rem 0;">
                        <span style="color: #808080;">VaR:</span>
                        <span style="color: #d62728; font-weight: bold;"> {var_pct:.2f}%</span>
                    </div>
                    <div style="margin: 0.5rem 0;">
                        <span style="color: #808080;">CVaR:</span>
                        <span style="color: #ff7f0e; font-weight: bold;"> {cvar_pct:.2f}%</span>
                    </div>
                    <div style="margin: 0.5rem 0;">
                        <span style="color: #808080;">$ at Risk:</span>
                        <span style="color: white; font-weight: bold;"> {format_currency(var_amount)}</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )


# =============================================================================
# PERFORMANCE METRICS DISPLAY
# =============================================================================


def render_performance_metrics(metrics: PerformanceMetrics) -> None:
    """
    Render a comprehensive performance metrics display.

    Args:
        metrics: PerformanceMetrics object
    """
    st.markdown("### 📈 Performance Metrics")

    # Returns Section
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        delta_color = "normal" if metrics.total_return >= 0 else "inverse"
        render_metric_card(
            "Total Return",
            format_percentage(metrics.total_return),
            delta_color=delta_color,
        )

    with col2:
        render_metric_card(
            "Annualized Return",
            format_percentage(metrics.annualized_return),
        )

    with col3:
        render_metric_card(
            "Volatility",
            format_percentage(metrics.volatility),
        )

    with col4:
        render_metric_card(
            "Max Drawdown",
            format_percentage(metrics.max_drawdown),
            delta_color="inverse",
        )

    # Risk-Adjusted Metrics
    st.markdown("#### Risk-Adjusted Metrics")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        color = "normal" if metrics.sharpe_ratio >= 0 else "inverse"
        render_metric_card("Sharpe Ratio", format_ratio(metrics.sharpe_ratio), delta_color=color)

    with col2:
        color = "normal" if metrics.sortino_ratio >= 0 else "inverse"
        render_metric_card("Sortino Ratio", format_ratio(metrics.sortino_ratio), delta_color=color)

    with col3:
        render_metric_card(
            "Profit Factor",
            format_ratio(metrics.profit_factor),
            help_text="Gross profit / Gross loss",
        )

    with col4:
        render_metric_card(
            "Win Rate",
            format_percentage(metrics.win_rate),
        )

    # Trading Statistics
    st.markdown("#### Trading Statistics")
    col1, col2 = st.columns(2)

    with col1:
        render_metric_card("Total Trades", str(metrics.num_trades))


def render_performance_summary_card(
    metrics: PerformanceMetrics, title: str = "Strategy Performance"
) -> None:
    """
    Render a compact performance summary card.

    Args:
        metrics: PerformanceMetrics object
        title: Card title
    """
    return_color = "#2ca02c" if metrics.total_return >= 0 else "#d62728"

    st.markdown(
        f"""
        <div style="
            background-color: #262730;
            padding: 1.5rem;
            border-radius: 0.5rem;
            border-left: 4px solid {return_color};
        ">
            <h4 style="margin: 0 0 1rem 0; color: white;">{title}</h4>
            <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 1rem;">
                <div>
                    <span style="color: #808080;">Return:</span>
                    <span style="color: {return_color}; font-weight: bold; font-size: 1.25rem;">
                        {format_percentage(metrics.total_return)}
                    </span>
                </div>
                <div>
                    <span style="color: #808080;">Sharpe:</span>
                    <span style="color: white; font-weight: bold;">
                        {format_ratio(metrics.sharpe_ratio)}
                    </span>
                </div>
                <div>
                    <span style="color: #808080;">Max DD:</span>
                    <span style="color: #d62728; font-weight: bold;">
                        {format_percentage(metrics.max_drawdown)}
                    </span>
                </div>
                <div>
                    <span style="color: #808080;">Win Rate:</span>
                    <span style="color: white; font-weight: bold;">
                        {format_percentage(metrics.win_rate)}
                    </span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# =============================================================================
# SENTIMENT METRICS DISPLAY
# =============================================================================


def render_sentiment_metrics(metrics: SentimentMetrics) -> None:
    """
    Render sentiment metrics display.

    Args:
        metrics: SentimentMetrics object
    """
    st.markdown("### 📰 Sentiment Analysis")

    # Overall Sentiment
    sentiment_color = (
        "#2ca02c"
        if metrics.overall_score > 0.1
        else "#d62728"
        if metrics.overall_score < -0.1
        else "#808080"
    )

    sentiment_label = (
        "Bullish"
        if metrics.overall_score > 0.1
        else "Bearish"
        if metrics.overall_score < -0.1
        else "Neutral"
    )

    st.markdown(
        f"""
        <div style="
            background-color: #262730;
            padding: 1.5rem;
            border-radius: 0.5rem;
            text-align: center;
            border: 2px solid {sentiment_color};
            margin-bottom: 1rem;
        ">
            <div style="font-size: 2rem; color: {sentiment_color}; font-weight: bold;">
                {metrics.overall_score:.2f}
            </div>
            <div style="font-size: 1.25rem; color: {sentiment_color};">
                {sentiment_label}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Source breakdown
    if any([metrics.news_score, metrics.reddit_score, metrics.twitter_score]):
        st.markdown("#### By Source")
        cols = st.columns(3)

        sources = [
            ("📰 News", metrics.news_score),
            ("🔴 Reddit", metrics.reddit_score),
            ("🐦 Twitter", metrics.twitter_score),
        ]

        for idx, (label, score) in enumerate(sources):
            with cols[idx]:
                if score is not None:
                    render_styled_metric_card(
                        label=label,
                        value=f"{score:.2f}",
                        color="#1f77b4",
                    )

    # Volume and trend
    if metrics.volume is not None or metrics.trend is not None:
        col1, col2 = st.columns(2)

        with col1:
            if metrics.volume is not None:
                render_metric_card("Mention Volume", format_number(metrics.volume))

        with col2:
            if metrics.trend is not None:
                trend_emoji = (
                    "📈"
                    if metrics.trend == "bullish"
                    else "📉"
                    if metrics.trend == "bearish"
                    else "➡️"
                )
                render_metric_card("Trend", f"{trend_emoji} {metrics.trend.title()}")


def render_fear_greed_gauge(score: float, title: str = "Fear & Greed Index") -> None:
    """
    Render a fear and greed index gauge.

    Args:
        score: Score from 0 (extreme fear) to 100 (extreme greed)
        title: Display title
    """
    # Determine sentiment zone
    if score < 25:
        zone = "Extreme Fear"
        zone_color = "#d62728"
    elif score < 45:
        zone = "Fear"
        zone_color = "#ff7f0e"
    elif score < 55:
        zone = "Neutral"
        zone_color = "#808080"
    elif score < 75:
        zone = "Greed"
        zone_color = "#90EE90"
    else:
        zone = "Extreme Greed"
        zone_color = "#2ca02c"

    st.markdown(
        f"""
        <div style="
            background-color: #262730;
            padding: 1.5rem;
            border-radius: 0.5rem;
            text-align: center;
        ">
            <div style="color: #808080; margin-bottom: 0.5rem;">{title}</div>
            <div style="font-size: 3rem; font-weight: bold; color: {zone_color};">
                {score:.0f}
            </div>
            <div style="font-size: 1.25rem; color: {zone_color};">
                {zone}
            </div>
            <div style="
                margin-top: 1rem;
                height: 10px;
                background: linear-gradient(to right, #d62728, #ff7f0e, #808080, #90EE90, #2ca02c);
                border-radius: 5px;
                position: relative;
            ">
                <div style="
                    position: absolute;
                    left: {score}%;
                    top: -5px;
                    width: 3px;
                    height: 20px;
                    background: white;
                    border-radius: 2px;
                "></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# =============================================================================
# CALCULATION UTILITIES
# =============================================================================


def calculate_risk_metrics(
    returns: pd.Series, benchmark_returns: pd.Series | None = None
) -> RiskMetrics:
    """
    Calculate risk metrics from a returns series.

    Args:
        returns: Series of portfolio returns
        benchmark_returns: Optional benchmark returns for beta calculation

    Returns:
        RiskMetrics object
    """
    # VaR calculations
    var_95 = -np.percentile(returns, 5)
    var_99 = -np.percentile(returns, 1)

    # CVaR (Expected Shortfall)
    cvar_95 = -returns[returns <= -var_95].mean() if (returns <= -var_95).any() else var_95
    cvar_99 = -returns[returns <= -var_99].mean() if (returns <= -var_99).any() else var_99

    # Drawdown calculations
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max

    max_drawdown = abs(drawdown.min())
    current_drawdown = abs(drawdown.iloc[-1])

    # Volatility
    volatility = returns.std() * np.sqrt(252)

    # Beta calculation
    beta = None
    if benchmark_returns is not None and len(benchmark_returns) == len(returns):
        covariance = np.cov(returns, benchmark_returns)[0, 1]
        benchmark_variance = np.var(benchmark_returns)
        beta = covariance / benchmark_variance if benchmark_variance > 0 else 1.0

    # Sharpe ratio (assuming risk-free rate of 0)
    sharpe = (returns.mean() * 252) / volatility if volatility > 0 else 0

    # Sortino ratio
    downside_returns = returns[returns < 0]
    downside_std = (
        downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else volatility
    )
    sortino = (returns.mean() * 252) / downside_std if downside_std > 0 else 0

    return RiskMetrics(
        var_95=var_95,
        var_99=var_99,
        cvar_95=cvar_95,
        cvar_99=cvar_99,
        max_drawdown=max_drawdown,
        current_drawdown=current_drawdown,
        volatility=volatility,
        beta=beta,
        sharpe=sharpe,
        sortino=sortino,
    )


def calculate_performance_metrics(
    returns: pd.Series,
    trades: pd.DataFrame | None = None,
) -> PerformanceMetrics:
    """
    Calculate performance metrics from returns and trades.

    Args:
        returns: Series of portfolio returns
        trades: Optional DataFrame with trade data

    Returns:
        PerformanceMetrics object
    """
    # Total return
    total_return = (1 + returns).prod() - 1

    # Annualized return (assuming daily returns)
    num_years = len(returns) / 252
    annualized_return = (1 + total_return) ** (1 / num_years) - 1 if num_years > 0 else 0

    # Volatility
    volatility = returns.std() * np.sqrt(252)

    # Sharpe ratio
    sharpe_ratio = annualized_return / volatility if volatility > 0 else 0

    # Sortino ratio
    downside_returns = returns[returns < 0]
    downside_std = (
        downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else volatility
    )
    sortino_ratio = annualized_return / downside_std if downside_std > 0 else 0

    # Max drawdown
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = abs(drawdown.min())

    # Trading metrics
    if trades is not None and len(trades) > 0:
        winning_trades = trades[trades["pnl"] > 0]
        losing_trades = trades[trades["pnl"] < 0]

        win_rate = len(winning_trades) / len(trades) if len(trades) > 0 else 0
        gross_profit = winning_trades["pnl"].sum() if len(winning_trades) > 0 else 0
        gross_loss = abs(losing_trades["pnl"].sum()) if len(losing_trades) > 0 else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")
        num_trades = len(trades)
    else:
        win_rate = (returns > 0).mean()
        profit_factor = (
            abs(returns[returns > 0].sum() / returns[returns < 0].sum())
            if (returns < 0).any()
            else float("inf")
        )
        num_trades = 0

    return PerformanceMetrics(
        total_return=total_return,
        annualized_return=annualized_return,
        volatility=volatility,
        sharpe_ratio=sharpe_ratio,
        sortino_ratio=sortino_ratio,
        max_drawdown=max_drawdown,
        win_rate=win_rate,
        profit_factor=profit_factor if not np.isinf(profit_factor) else 999.99,
        num_trades=num_trades,
    )
