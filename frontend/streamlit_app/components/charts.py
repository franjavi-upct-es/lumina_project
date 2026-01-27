# frontend/streamlit-app/components/charts.py
"""
Reusable chart components for Lumina Quant Lab dashboard.
Provides standardized Plotly visualizations for financial data analysis.
"""

from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# =============================================================================
# CONFIGURATION
# =============================================================================

# Default Plotly template for consistent styling
DEFAULT_TEMPLATE = "plotly_dark"

# Color palette for consistent visualizations
COLORS = {
    "primary": "#1f77b4",
    "secondary": "#ff7f0e",
    "success": "#2ca02c",
    "danger": "#d62728",
    "warning": "#ffbb00",
    "info": "#17a2b8",
    "purple": "#9467bd",
    "pink": "#e377c2",
    "gray": "#7f7f7f",
    "cyan": "#17becf",
}

# Gradient colors for heatmaps
HEATMAP_COLORS = [
    [0, "#d62728"],  # Red (negative)
    [0.5, "#f0f0f0"],  # White/Gray (neutral)
    [1, "#2ca02c"],  # Green (positive)
]


# =============================================================================
# PRICE CHARTS
# =============================================================================


def create_candlestick_chart(
    df: pd.DataFrame,
    title: str = "Price Chart",
    show_volume: bool = True,
    height: int = 600,
) -> go.Figure:
    """
    Create an interactive candlestick chart with optional volume bars.

    Args:
        df: DataFrame with columns [time, open, high, low, close, volume]
        title: Chart title
        show_volume: Whether to show volume subplot
        height: Chart height in pixels

    Returns:
        Plotly Figure object
    """
    if show_volume and "volume" in df.columns:
        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.7, 0.3],
            subplot_titles=(title, "Volume"),
        )

        # Candlestick chart
        fig.add_trace(
            go.Candlestick(
                x=df["time"],
                open=df["open"],
                high=df["high"],
                low=df["low"],
                close=df["close"],
                name="OHLC",
                increasing_line_color=COLORS["success"],
                decreasing_line_color=COLORS["danger"],
            ),
            row=1,
            col=1,
        )

        # Volume bars with color based on price direction
        colors = [
            COLORS["success"] if close >= open_ else COLORS["danger"]
            for close, open_ in zip(df["close"], df["open"], strict=False)
        ]

        fig.add_trace(
            go.Bar(
                x=df["time"],
                y=df["volume"],
                name="Volume",
                marker_color=colors,
                opacity=0.7,
            ),
            row=2,
            col=1,
        )

        fig.update_layout(
            xaxis_rangeslider_visible=False,
            xaxis2_rangeslider_visible=False,
        )
    else:
        fig = go.Figure()

        fig.add_trace(
            go.Candlestick(
                x=df["time"],
                open=df["open"],
                high=df["high"],
                low=df["low"],
                close=df["close"],
                name="OHLC",
                increasing_line_color=COLORS["success"],
                decreasing_line_color=COLORS["danger"],
            )
        )

        fig.update_layout(title=title, xaxis_rangeslider_visible=False)

    fig.update_layout(
        template=DEFAULT_TEMPLATE,
        height=height,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    return fig


def create_line_chart(
    df: pd.DataFrame,
    x_col: str,
    y_cols: list[str],
    title: str = "Line Chart",
    normalize: bool = False,
    height: int = 400,
    show_legend: bool = True,
) -> go.Figure:
    """
    Create a multi-line chart for comparing multiple series.

    Args:
        df: DataFrame with data
        x_col: Column name for x-axis
        y_cols: List of column names for y-axis
        title: Chart title
        normalize: Whether to normalize values to 100
        height: Chart height in pixels
        show_legend: Whether to show legend

    Returns:
        Plotly Figure object
    """
    fig = go.Figure()

    color_cycle = list(COLORS.values())

    for idx, col in enumerate(y_cols):
        if col not in df.columns:
            continue

        y_data = df[col].copy()

        if normalize and len(y_data) > 0:
            first_valid = y_data.dropna().iloc[0] if not y_data.dropna().empty else 1
            y_data = (y_data / first_valid) * 100

        fig.add_trace(
            go.Scatter(
                x=df[x_col],
                y=y_data,
                name=col,
                mode="lines",
                line=dict(color=color_cycle[idx % len(color_cycle)]),
            )
        )

    y_title = "Normalized (Base=100)" if normalize else "Value"

    fig.update_layout(
        title=title,
        xaxis_title=x_col.title(),
        yaxis_title=y_title,
        template=DEFAULT_TEMPLATE,
        height=height,
        hovermode="x unified",
        showlegend=show_legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    return fig


def create_ohlc_with_indicators(
    df: pd.DataFrame,
    indicators: dict[str, pd.Series] | None = None,
    title: str = "Price with Indicators",
    height: int = 600,
) -> go.Figure:
    """
    Create OHLC chart with technical indicators overlay.

    Args:
        df: DataFrame with OHLC data
        indicators: Dict of indicator name -> Series
        title: Chart title
        height: Chart height

    Returns:
        Plotly Figure object
    """
    fig = create_candlestick_chart(df, title=title, show_volume=True, height=height)

    if indicators:
        color_cycle = [COLORS["secondary"], COLORS["purple"], COLORS["cyan"], COLORS["pink"]]

        for idx, (name, series) in enumerate(indicators.items()):
            fig.add_trace(
                go.Scatter(
                    x=df["time"],
                    y=series,
                    name=name,
                    mode="lines",
                    line=dict(color=color_cycle[idx % len(color_cycle)], width=1),
                ),
                row=1,
                col=1,
            )

    return fig


# =============================================================================
# RISK CHARTS
# =============================================================================


def create_drawdown_chart(
    drawdown_series: pd.Series,
    dates: pd.Series | None = None,
    title: str = "Drawdown Analysis",
    height: int = 400,
) -> go.Figure:
    """
    Create a drawdown visualization chart.

    Args:
        drawdown_series: Series with drawdown values (negative percentages)
        dates: Optional date index
        title: Chart title
        height: Chart height

    Returns:
        Plotly Figure object
    """
    x_data = dates if dates is not None else drawdown_series.index

    fig = go.Figure()

    # Fill area for drawdown
    fig.add_trace(
        go.Scatter(
            x=x_data,
            y=drawdown_series * 100,
            fill="tozeroy",
            name="Drawdown",
            line=dict(color=COLORS["danger"]),
            fillcolor="rgba(214, 39, 40, 0.3)",
        )
    )

    # Add horizontal line at max drawdown
    max_dd = drawdown_series.min() * 100

    fig.add_hline(
        y=max_dd,
        line_dash="dash",
        line_color=COLORS["warning"],
        annotation_text=f"Max DD: {max_dd:.2f}%",
        annotation_position="right",
    )

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Drawdown (%)",
        template=DEFAULT_TEMPLATE,
        height=height,
        hovermode="x unified",
    )

    return fig


def create_var_chart(
    returns: pd.Series,
    var_95: float,
    var_99: float,
    title: str = "Value at Risk Distribution",
    height: int = 400,
) -> go.Figure:
    """
    Create a VaR visualization with return distribution.

    Args:
        returns: Series of returns
        var_95: VaR at 95% confidence
        var_99: VaR at 99% confidence
        title: Chart title
        height: Chart height

    Returns:
        Plotly Figure object
    """
    fig = go.Figure()

    # Histogram of returns
    fig.add_trace(
        go.Histogram(
            x=returns * 100,
            nbinsx=50,
            name="Returns Distribution",
            marker_color=COLORS["primary"],
            opacity=0.7,
        )
    )

    # VaR lines
    fig.add_vline(
        x=-var_95 * 100,
        line_dash="dash",
        line_color=COLORS["warning"],
        annotation_text=f"VaR 95%: {var_95 * 100:.2f}%",
        annotation_position="top",
    )

    fig.add_vline(
        x=-var_99 * 100,
        line_dash="dash",
        line_color=COLORS["danger"],
        annotation_text=f"VaR 99%: {var_99 * 100:.2f}%",
        annotation_position="top",
    )

    fig.update_layout(
        title=title,
        xaxis_title="Daily Returns (%)",
        yaxis_title="Frequency",
        template=DEFAULT_TEMPLATE,
        height=height,
        showlegend=True,
    )

    return fig


def create_correlation_heatmap(
    correlation_matrix: pd.DataFrame,
    title: str = "Correlation Matrix",
    height: int = 500,
) -> go.Figure:
    """
    Create a correlation heatmap.

    Args:
        correlation_matrix: DataFrame with correlation values
        title: Chart title
        height: Chart height

    Returns:
        Plotly Figure object
    """
    fig = go.Figure(
        data=go.Heatmap(
            z=correlation_matrix.values,
            x=correlation_matrix.columns.tolist(),
            y=correlation_matrix.index.tolist(),
            colorscale=HEATMAP_COLORS,
            zmin=-1,
            zmax=1,
            text=np.round(correlation_matrix.values, 2),
            texttemplate="%{text}",
            textfont={"size": 10},
            hoverongaps=False,
            colorbar=dict(title="Correlation"),
        )
    )

    fig.update_layout(
        title=title,
        template=DEFAULT_TEMPLATE,
        height=height,
        xaxis=dict(side="bottom"),
    )

    return fig


def create_risk_contribution_chart(
    contributions: dict[str, float],
    title: str = "Risk Contribution by Asset",
    height: int = 400,
) -> go.Figure:
    """
    Create a pie/bar chart showing risk contribution by asset.

    Args:
        contributions: Dict of asset -> contribution percentage
        title: Chart title
        height: Chart height

    Returns:
        Plotly Figure object
    """
    assets = list(contributions.keys())
    values = list(contributions.values())

    fig = make_subplots(
        rows=1,
        cols=2,
        specs=[[{"type": "pie"}, {"type": "bar"}]],
        subplot_titles=("Percentage Contribution", "Absolute Contribution"),
    )

    # Pie chart
    fig.add_trace(
        go.Pie(
            labels=assets,
            values=values,
            name="Risk Contribution",
            hole=0.4,
            marker_colors=list(COLORS.values())[: len(assets)],
        ),
        row=1,
        col=1,
    )

    # Bar chart
    fig.add_trace(
        go.Bar(
            x=assets,
            y=values,
            name="Contribution",
            marker_color=[COLORS["primary"]] * len(assets),
        ),
        row=1,
        col=2,
    )

    fig.update_layout(
        title=title,
        template=DEFAULT_TEMPLATE,
        height=height,
        showlegend=False,
    )

    return fig


# =============================================================================
# SENTIMENT CHARTS
# =============================================================================


def create_sentiment_gauge(
    score: float,
    title: str = "Sentiment Score",
    height: int = 300,
) -> go.Figure:
    """
    Create a sentiment gauge visualization.

    Args:
        score: Sentiment score between -1 and 1
        title: Chart title
        height: Chart height

    Returns:
        Plotly Figure object
    """
    # Normalize score to 0-100 scale for gauge
    normalized_score = (score + 1) * 50

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number+delta",
            value=normalized_score,
            title={"text": title},
            delta={"reference": 50, "relative": False},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1},
                "bar": {"color": COLORS["primary"]},
                "bgcolor": "white",
                "borderwidth": 2,
                "bordercolor": "gray",
                "steps": [
                    {"range": [0, 30], "color": COLORS["danger"]},
                    {"range": [30, 45], "color": COLORS["warning"]},
                    {"range": [45, 55], "color": COLORS["gray"]},
                    {"range": [55, 70], "color": "#90EE90"},
                    {"range": [70, 100], "color": COLORS["success"]},
                ],
                "threshold": {
                    "line": {"color": "black", "width": 4},
                    "thickness": 0.75,
                    "value": normalized_score,
                },
            },
        )
    )

    fig.update_layout(
        template=DEFAULT_TEMPLATE,
        height=height,
    )

    return fig


def create_sentiment_timeline(
    df: pd.DataFrame,
    date_col: str = "time",
    sentiment_col: str = "sentiment_score",
    source_col: str | None = None,
    title: str = "Sentiment Over Time",
    height: int = 400,
) -> go.Figure:
    """
    Create a sentiment timeline visualization.

    Args:
        df: DataFrame with sentiment data
        date_col: Column name for dates
        sentiment_col: Column name for sentiment scores
        source_col: Optional column for data source
        title: Chart title
        height: Chart height

    Returns:
        Plotly Figure object
    """
    fig = go.Figure()

    if source_col and source_col in df.columns:
        # Plot by source
        for source in df[source_col].unique():
            source_df = df[df[source_col] == source]
            fig.add_trace(
                go.Scatter(
                    x=source_df[date_col],
                    y=source_df[sentiment_col],
                    name=source.title(),
                    mode="lines+markers",
                    marker=dict(size=4),
                )
            )
    else:
        fig.add_trace(
            go.Scatter(
                x=df[date_col],
                y=df[sentiment_col],
                name="Sentiment",
                mode="lines+markers",
                line=dict(color=COLORS["primary"]),
                marker=dict(size=4),
            )
        )

    # Add neutral line
    fig.add_hline(y=0, line_dash="dash", line_color=COLORS["gray"], opacity=0.5)

    # Add colored regions
    fig.add_hrect(
        y0=0,
        y1=1,
        fillcolor=COLORS["success"],
        opacity=0.1,
        line_width=0,
    )

    fig.add_hrect(
        y0=-1,
        y1=0,
        fillcolor=COLORS["danger"],
        opacity=0.1,
        line_width=0,
    )

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Sentiment Score",
        yaxis=dict(range=[-1.1, 1.1]),
        template=DEFAULT_TEMPLATE,
        height=height,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    return fig


def create_sentiment_by_source_chart(
    sources: list[str],
    scores: list[float],
    title: str = "Sentiment by Source",
    height: int = 350,
) -> go.Figure:
    """
    Create a bar chart comparing sentiment across sources.

    Args:
        sources: List of source names
        scores: List of sentiment scores
        title: Chart title
        height: Chart height

    Returns:
        Plotly Figure object
    """
    colors = [
        COLORS["success"] if score > 0.1 else COLORS["danger"] if score < -0.1 else COLORS["gray"]
        for score in scores
    ]

    fig = go.Figure(
        go.Bar(
            x=sources,
            y=scores,
            marker_color=colors,
            text=[f"{s:.2f}" for s in scores],
            textposition="outside",
        )
    )

    fig.add_hline(y=0, line_dash="solid", line_color=COLORS["gray"])

    fig.update_layout(
        title=title,
        xaxis_title="Source",
        yaxis_title="Sentiment Score",
        yaxis=dict(range=[-1.1, 1.1]),
        template=DEFAULT_TEMPLATE,
        height=height,
    )

    return fig


# =============================================================================
# PERFORMANCE CHARTS
# =============================================================================


def create_equity_curve(
    equity: pd.Series,
    benchmark: pd.Series | None = None,
    title: str = "Equity Curve",
    height: int = 400,
) -> go.Figure:
    """
    Create an equity curve visualization.

    Args:
        equity: Series with portfolio equity values
        benchmark: Optional benchmark series
        title: Chart title
        height: Chart height

    Returns:
        Plotly Figure object
    """
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=equity.index,
            y=equity.values,
            name="Portfolio",
            mode="lines",
            line=dict(color=COLORS["primary"], width=2),
            fill="tozeroy",
            fillcolor="rgba(31, 119, 180, 0.2)",
        )
    )

    if benchmark is not None:
        fig.add_trace(
            go.Scatter(
                x=benchmark.index,
                y=benchmark.values,
                name="Benchmark",
                mode="lines",
                line=dict(color=COLORS["secondary"], width=2, dash="dash"),
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Portfolio Value",
        template=DEFAULT_TEMPLATE,
        height=height,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    return fig


def create_returns_distribution(
    returns: pd.Series,
    title: str = "Returns Distribution",
    height: int = 400,
) -> go.Figure:
    """
    Create a returns distribution histogram with statistics.

    Args:
        returns: Series of returns
        title: Chart title
        height: Chart height

    Returns:
        Plotly Figure object
    """
    fig = go.Figure()

    fig.add_trace(
        go.Histogram(
            x=returns * 100,
            nbinsx=50,
            name="Returns",
            marker_color=COLORS["primary"],
            opacity=0.7,
        )
    )

    # Add mean line
    mean_return = returns.mean() * 100
    fig.add_vline(
        x=mean_return,
        line_dash="dash",
        line_color=COLORS["success"],
        annotation_text=f"Mean: {mean_return:.2f}%",
        annotation_position="top",
    )

    # Add zero line
    fig.add_vline(x=0, line_color=COLORS["gray"], opacity=0.5)

    fig.update_layout(
        title=title,
        xaxis_title="Daily Returns (%)",
        yaxis_title="Frequency",
        template=DEFAULT_TEMPLATE,
        height=height,
    )

    return fig


def create_rolling_metrics_chart(
    df: pd.DataFrame,
    date_col: str = "date",
    metrics: list[str] | None = None,
    title: str = "Rolling Performance Metrics",
    height: int = 500,
) -> go.Figure:
    """
    Create a chart showing rolling performance metrics.

    Args:
        df: DataFrame with rolling metrics
        date_col: Column name for dates
        metrics: List of metric columns to plot
        title: Chart title
        height: Chart height

    Returns:
        Plotly Figure object
    """
    if metrics is None:
        metrics = [col for col in df.columns if col != date_col]

    num_metrics = len(metrics)
    fig = make_subplots(
        rows=num_metrics,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=metrics,
    )

    color_cycle = list(COLORS.values())

    for idx, metric in enumerate(metrics, 1):
        if metric not in df.columns:
            continue

        fig.add_trace(
            go.Scatter(
                x=df[date_col],
                y=df[metric],
                name=metric,
                mode="lines",
                line=dict(color=color_cycle[(idx - 1) % len(color_cycle)]),
            ),
            row=idx,
            col=1,
        )

    fig.update_layout(
        title=title,
        template=DEFAULT_TEMPLATE,
        height=height,
        showlegend=False,
        hovermode="x unified",
    )

    return fig


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================


def add_annotations(
    fig: go.Figure,
    annotations: list[dict[str, Any]],
) -> go.Figure:
    """
    Add annotations to an existing figure.

    Args:
        fig: Plotly Figure object
        annotations: List of annotation dicts with keys [x, y, text, ...]

    Returns:
        Updated Plotly Figure object
    """
    for ann in annotations:
        fig.add_annotation(
            x=ann.get("x"),
            y=ann.get("y"),
            text=ann.get("text", ""),
            showarrow=ann.get("showarrow", True),
            arrowhead=ann.get("arrowhead", 2),
            arrowsize=ann.get("arrowsize", 1),
            arrowwidth=ann.get("arrowwidth", 1),
            ax=ann.get("ax", 0),
            ay=ann.get("ay", -40),
            font=ann.get("font", {"size": 10}),
        )

    return fig


def update_chart_theme(fig: go.Figure, theme: str = "dark") -> go.Figure:
    """
    Update chart theme.

    Args:
        fig: Plotly Figure object
        theme: Theme name ('dark' or 'light')

    Returns:
        Updated Plotly Figure object
    """
    template = "plotly_dark" if theme == "dark" else "plotly_white"
    fig.update_layout(template=template)
    return fig
