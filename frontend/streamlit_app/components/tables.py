# frontend/streamlit-app/components/tables.py
"""
Reusable table components for Lumina Quant Lab dashboard.
Provides standardized data table displays with sorting, filtering, and styling.
"""

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

# =============================================================================
# STYLING CONFIGURATION
# =============================================================================

# Default table styling
DEFAULT_TABLE_STYLES = {
    "background_gradient": True,
    "precision": 2,
    "hide_index": True,
}

# Color maps for different data types
COLOR_MAPS = {
    "positive_negative": {
        "positive": "#2ca02c",  # Green
        "negative": "#d62728",  # Red
        "neutral": "#808080",  # Gray
    },
    "heat": {
        "low": "#d62728",
        "mid": "#ffffff",
        "high": "#2ca02c",
    },
    "sentiment": {
        "bullish": "#2ca02c",
        "neutral": "#808080",
        "bearish": "#d62728",
    },
}


# =============================================================================
# FORMATTING UTILITIES
# =============================================================================


def format_currency_col(value: float) -> str:
    """Format a value as currency for table display."""
    if pd.isna(value):
        return "-"
    if abs(value) >= 1e9:
        return f"${value / 1e9:.2f}B"
    elif abs(value) >= 1e6:
        return f"${value / 1e6:.2f}M"
    elif abs(value) >= 1e3:
        return f"${value / 1e3:.2f}K"
    else:
        return f"${value:,.2f}"


def format_percentage_col(value: float) -> str:
    """Format a value as percentage for table display."""
    if pd.isna(value):
        return "-"
    return f"{value * 100:.2f}%"


def format_number_col(value: float, decimals: int = 2) -> str:
    """Format a number with thousands separator."""
    if pd.isna(value):
        return "-"
    if abs(value) >= 1e9:
        return f"{value / 1e9:.{decimals}f}B"
    elif abs(value) >= 1e6:
        return f"{value / 1e6:.{decimals}f}M"
    elif abs(value) >= 1e3:
        return f"{value / 1e3:.{decimals}f}K"
    else:
        return f"{value:,.{decimals}f}"


def format_datetime_col(value: Any) -> str:
    """Format a datetime value for table display."""
    if pd.isna(value):
        return "-"
    try:
        return pd.to_datetime(value).strftime("%Y-%m-%d %H:%M")
    except (ValueError, TypeError):
        return str(value)


def format_date_col(value: Any) -> str:
    """Format a date value for table display."""
    if pd.isna(value):
        return "-"
    try:
        return pd.to_datetime(value).strftime("%Y-%m-%d")
    except (ValueError, TypeError):
        return str(value)


# =============================================================================
# STYLING FUNCTIONS
# =============================================================================


def color_positive_negative(val: float) -> str:
    """
    Color cells based on positive/negative values.

    Args:
        val: Numeric value

    Returns:
        CSS color string
    """
    if pd.isna(val):
        return ""
    if val > 0:
        return f"color: {COLOR_MAPS['positive_negative']['positive']}"
    elif val < 0:
        return f"color: {COLOR_MAPS['positive_negative']['negative']}"
    else:
        return f"color: {COLOR_MAPS['positive_negative']['neutral']}"


def background_positive_negative(val: float, alpha: float = 0.3) -> str:
    """
    Background color cells based on positive/negative values.

    Args:
        val: Numeric value
        alpha: Opacity of the background

    Returns:
        CSS background-color string
    """
    if pd.isna(val):
        return ""
    if val > 0:
        return f"background-color: rgba(44, 160, 44, {alpha})"
    elif val < 0:
        return f"background-color: rgba(214, 39, 40, {alpha})"
    else:
        return ""


def color_sentiment(val: float) -> str:
    """
    Color cells based on sentiment score.

    Args:
        val: Sentiment score (-1 to 1)

    Returns:
        CSS color string
    """
    if pd.isna(val):
        return ""
    if val > 0.1:
        return f"color: {COLOR_MAPS['sentiment']['bullish']}"
    elif val < -0.1:
        return f"color: {COLOR_MAPS['sentiment']['bearish']}"
    else:
        return f"color: {COLOR_MAPS['sentiment']['neutral']}"


# =============================================================================
# TABLE RENDERING FUNCTIONS
# =============================================================================


def render_dataframe(
    df: pd.DataFrame,
    title: str | None = None,
    hide_index: bool = True,
    use_container_width: bool = True,
    height: int | None = None,
    column_config: dict[str, Any] | None = None,
) -> None:
    """
    Render a basic DataFrame with Streamlit's native dataframe component.

    Args:
        df: DataFrame to display
        title: Optional title above the table
        hide_index: Whether to hide the index column
        use_container_width: Whether to use full container width
        height: Optional fixed height in pixels
        column_config: Optional column configuration dict
    """
    if title:
        st.markdown(f"### {title}")

    st.dataframe(
        df,
        hide_index=hide_index,
        use_container_width=use_container_width,
        height=height,
        column_config=column_config,
    )


def render_styled_table(
    df: pd.DataFrame,
    title: str | None = None,
    format_columns: dict[str, Callable] | None = None,
    style_columns: dict[str, Callable] | None = None,
    gradient_columns: list[str] | None = None,
    hide_index: bool = True,
    precision: int = 2,
) -> None:
    """
    Render a styled DataFrame with custom formatting and colors.

    Args:
        df: DataFrame to display
        title: Optional title above the table
        format_columns: Dict of column -> formatter function
        style_columns: Dict of column -> style function
        gradient_columns: List of columns to apply gradient background
        hide_index: Whether to hide the index
        precision: Decimal precision for floats
    """
    if title:
        st.markdown(f"### {title}")

    # Create a copy for styling
    styled_df = df.copy()

    # Apply formatters
    if format_columns:
        for col, formatter in format_columns.items():
            if col in styled_df.columns:
                styled_df[col] = styled_df[col].apply(formatter)

    # Build styler
    styler = df.style

    # Apply column-specific styles
    if style_columns:
        for col, style_func in style_columns.items():
            if col in df.columns:
                styler = styler.map(style_func, subset=[col])

    # Apply gradient backgrounds
    if gradient_columns:
        for col in gradient_columns:
            if col in df.columns:
                styler = styler.background_gradient(subset=[col], cmap="RdYlGn")

    # Set precision
    styler = styler.format(precision=precision)

    # Hide index if requested
    if hide_index:
        styler = styler.hide(axis="index")

    st.dataframe(styler, use_container_width=True)


def render_price_table(
    df: pd.DataFrame,
    title: str = "Price Data",
    show_change: bool = True,
) -> None:
    """
    Render a price data table with OHLCV columns.

    Args:
        df: DataFrame with price data (time, open, high, low, close, volume)
        title: Table title
        show_change: Whether to show price change column
    """
    st.markdown(f"### {title}")

    display_df = df.copy()

    # Add change column if requested
    if show_change and "close" in display_df.columns:
        display_df["change"] = display_df["close"].pct_change()
        display_df["change_pct"] = display_df["change"] * 100

    # Configure columns
    column_config = {
        "time": st.column_config.DatetimeColumn("Date", format="YYYY-MM-DD HH:mm"),
        "open": st.column_config.NumberColumn("Open", format="$%.2f"),
        "high": st.column_config.NumberColumn("High", format="$%.2f"),
        "low": st.column_config.NumberColumn("Low", format="$%.2f"),
        "close": st.column_config.NumberColumn("Close", format="$%.2f"),
        "volume": st.column_config.NumberColumn("Volume", format="%d"),
    }

    if show_change:
        column_config["change_pct"] = st.column_config.NumberColumn(
            "Change %",
            format="%.2f%%",
        )

    # Select columns to display
    display_cols = ["time", "open", "high", "low", "close", "volume"]
    if show_change:
        display_cols.append("change_pct")

    available_cols = [c for c in display_cols if c in display_df.columns]

    st.dataframe(
        display_df[available_cols],
        column_config={k: v for k, v in column_config.items() if k in available_cols},
        hide_index=True,
        use_container_width=True,
    )


def render_trades_table(
    df: pd.DataFrame,
    title: str = "Trade History",
) -> None:
    """
    Render a trades table with P&L styling.

    Args:
        df: DataFrame with trade data
        title: Table title
    """
    st.markdown(f"### {title}")

    if df.empty:
        st.info("No trades to display")
        return

    display_df = df.copy()

    # Configure columns
    column_config = {
        "entry_time": st.column_config.DatetimeColumn("Entry", format="YYYY-MM-DD HH:mm"),
        "exit_time": st.column_config.DatetimeColumn("Exit", format="YYYY-MM-DD HH:mm"),
        "ticker": st.column_config.TextColumn("Ticker"),
        "side": st.column_config.TextColumn("Side"),
        "entry_price": st.column_config.NumberColumn("Entry Price", format="$%.2f"),
        "exit_price": st.column_config.NumberColumn("Exit Price", format="$%.2f"),
        "quantity": st.column_config.NumberColumn("Qty", format="%d"),
        "pnl": st.column_config.NumberColumn("P&L", format="$%.2f"),
        "pnl_pct": st.column_config.NumberColumn("P&L %", format="%.2f%%"),
        "duration": st.column_config.TextColumn("Duration"),
    }

    # Calculate P&L percentage if not present
    if "pnl_pct" not in display_df.columns and "pnl" in display_df.columns:
        if "entry_price" in display_df.columns and "quantity" in display_df.columns:
            display_df["pnl_pct"] = (
                display_df["pnl"] / (display_df["entry_price"] * display_df["quantity"]) * 100
            )

    # Select available columns
    available_cols = [c for c in column_config.keys() if c in display_df.columns]

    st.dataframe(
        display_df[available_cols],
        column_config={k: v for k, v in column_config.items() if k in available_cols},
        hide_index=True,
        use_container_width=True,
    )


def render_risk_table(
    risk_data: dict[str, Any],
    title: str = "Risk Metrics",
) -> None:
    """
    Render a risk metrics table from a dictionary.

    Args:
        risk_data: Dictionary of risk metrics
        title: Table title
    """
    st.markdown(f"### {title}")

    # Convert dict to DataFrame
    df = pd.DataFrame([{"Metric": k, "Value": v} for k, v in risk_data.items()])

    # Format values based on content
    def format_value(row: pd.Series) -> str:
        metric = row["Metric"].lower()
        value = row["Value"]

        if pd.isna(value):
            return "-"

        if "%" in metric or "return" in metric or "var" in metric or "cvar" in metric:
            return format_percentage_col(value)
        elif "ratio" in metric or "beta" in metric or "sharpe" in metric:
            return f"{value:.2f}"
        elif "dollar" in metric or "amount" in metric or "value" in metric:
            return format_currency_col(value)
        else:
            return format_number_col(value)

    df["Formatted Value"] = df.apply(format_value, axis=1)

    st.dataframe(
        df[["Metric", "Formatted Value"]].rename(columns={"Formatted Value": "Value"}),
        hide_index=True,
        use_container_width=True,
    )


def render_correlation_table(
    correlation_matrix: pd.DataFrame,
    title: str = "Correlation Matrix",
) -> None:
    """
    Render a correlation matrix as a styled table.

    Args:
        correlation_matrix: DataFrame with correlation values
        title: Table title
    """
    st.markdown(f"### {title}")

    # Apply gradient styling
    styler = correlation_matrix.style.background_gradient(
        cmap="RdYlGn",
        vmin=-1,
        vmax=1,
    ).format("{:.2f}")

    st.dataframe(styler, use_container_width=True)


def render_portfolio_table(
    holdings: pd.DataFrame,
    title: str = "Portfolio Holdings",
) -> None:
    """
    Render a portfolio holdings table.

    Args:
        holdings: DataFrame with portfolio holdings
        title: Table title
    """
    st.markdown(f"### {title}")

    if holdings.empty:
        st.info("No holdings to display")
        return

    display_df = holdings.copy()

    # Configure columns
    column_config = {
        "ticker": st.column_config.TextColumn("Ticker"),
        "name": st.column_config.TextColumn("Name"),
        "quantity": st.column_config.NumberColumn("Shares", format="%d"),
        "avg_cost": st.column_config.NumberColumn("Avg Cost", format="$%.2f"),
        "current_price": st.column_config.NumberColumn("Current", format="$%.2f"),
        "market_value": st.column_config.NumberColumn("Market Value", format="$%.2f"),
        "cost_basis": st.column_config.NumberColumn("Cost Basis", format="$%.2f"),
        "unrealized_pnl": st.column_config.NumberColumn("Unrealized P&L", format="$%.2f"),
        "unrealized_pnl_pct": st.column_config.NumberColumn("P&L %", format="%.2f%%"),
        "weight": st.column_config.ProgressColumn(
            "Weight",
            format="%.1f%%",
            min_value=0,
            max_value=100,
        ),
        "day_change": st.column_config.NumberColumn("Day Change", format="$%.2f"),
        "day_change_pct": st.column_config.NumberColumn("Day %", format="%.2f%%"),
    }

    # Select available columns
    available_cols = [c for c in column_config.keys() if c in display_df.columns]

    st.dataframe(
        display_df[available_cols],
        column_config={k: v for k, v in column_config.items() if k in available_cols},
        hide_index=True,
        use_container_width=True,
    )


def render_model_comparison_table(
    models: pd.DataFrame,
    title: str = "Model Comparison",
) -> None:
    """
    Render a model comparison table.

    Args:
        models: DataFrame with model performance metrics
        title: Table title
    """
    st.markdown(f"### {title}")

    if models.empty:
        st.info("No models to compare")
        return

    display_df = models.copy()

    # Configure columns
    column_config = {
        "model_name": st.column_config.TextColumn("Model"),
        "model_type": st.column_config.TextColumn("Type"),
        "ticker": st.column_config.TextColumn("Ticker"),
        "mae": st.column_config.NumberColumn("MAE", format="%.4f"),
        "rmse": st.column_config.NumberColumn("RMSE", format="%.4f"),
        "mape": st.column_config.NumberColumn("MAPE %", format="%.2f%%"),
        "r2": st.column_config.NumberColumn("R²", format="%.4f"),
        "directional_accuracy": st.column_config.ProgressColumn(
            "Dir. Accuracy",
            format="%.1f%%",
            min_value=0,
            max_value=100,
        ),
        "trained_on": st.column_config.DatetimeColumn("Trained", format="YYYY-MM-DD"),
        "is_active": st.column_config.CheckboxColumn("Active"),
    }

    # Select available columns
    available_cols = [c for c in column_config.keys() if c in display_df.columns]

    st.dataframe(
        display_df[available_cols],
        column_config={k: v for k, v in column_config.items() if k in available_cols},
        hide_index=True,
        use_container_width=True,
    )


def render_sentiment_table(
    sentiment_data: pd.DataFrame,
    title: str = "Sentiment Data",
) -> None:
    """
    Render a sentiment data table.

    Args:
        sentiment_data: DataFrame with sentiment data
        title: Table title
    """
    st.markdown(f"### {title}")

    if sentiment_data.empty:
        st.info("No sentiment data to display")
        return

    display_df = sentiment_data.copy()

    # Configure columns
    column_config = {
        "time": st.column_config.DatetimeColumn("Date", format="YYYY-MM-DD HH:mm"),
        "ticker": st.column_config.TextColumn("Ticker"),
        "source": st.column_config.TextColumn("Source"),
        "sentiment_score": st.column_config.NumberColumn("Score", format="%.3f"),
        "confidence": st.column_config.ProgressColumn(
            "Confidence",
            format="%.0f%%",
            min_value=0,
            max_value=100,
        ),
        "volume": st.column_config.NumberColumn("Mentions", format="%d"),
        "text_snippet": st.column_config.TextColumn("Snippet", width="large"),
    }

    # Select available columns
    available_cols = [c for c in column_config.keys() if c in display_df.columns]

    st.dataframe(
        display_df[available_cols],
        column_config={k: v for k, v in column_config.items() if k in available_cols},
        hide_index=True,
        use_container_width=True,
    )


def render_stress_test_table(
    scenarios: dict[str, dict[str, Any]],
    title: str = "Stress Test Scenarios",
) -> None:
    """
    Render a stress test scenarios table.

    Args:
        scenarios: Dictionary of scenario results
        title: Table title
    """
    st.markdown(f"### {title}")

    # Convert to DataFrame
    rows = []
    for scenario_name, results in scenarios.items():
        row = {"Scenario": scenario_name}
        row.update(results)
        rows.append(row)

    df = pd.DataFrame(rows)

    if df.empty:
        st.info("No stress test scenarios available")
        return

    # Style based on impact
    def style_impact(val: float) -> str:
        if pd.isna(val):
            return ""
        if isinstance(val, (int, float)):
            if val < -0.1:
                return "background-color: rgba(214, 39, 40, 0.3)"
            elif val < -0.05:
                return "background-color: rgba(255, 127, 14, 0.3)"
            elif val > 0.05:
                return "background-color: rgba(44, 160, 44, 0.3)"
        return ""

    # Apply styling to numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    styler = df.style
    for col in numeric_cols:
        styler = styler.map(style_impact, subset=[col])

    styler = styler.format("{:.2%}", subset=numeric_cols)

    st.dataframe(styler, hide_index=True, use_container_width=True)


# =============================================================================
# INTERACTIVE TABLE COMPONENTS
# =============================================================================


def render_filterable_table(
    df: pd.DataFrame,
    title: str | None = None,
    filter_columns: list[str] | None = None,
    search_columns: list[str] | None = None,
) -> pd.DataFrame:
    """
    Render a table with filtering and search capabilities.

    Args:
        df: DataFrame to display
        title: Optional title
        filter_columns: Columns to add filters for
        search_columns: Columns to search in

    Returns:
        Filtered DataFrame
    """
    if title:
        st.markdown(f"### {title}")

    filtered_df = df.copy()

    # Create filter columns
    if filter_columns:
        filter_cols = st.columns(len(filter_columns))

        for idx, col in enumerate(filter_columns):
            if col not in df.columns:
                continue

            with filter_cols[idx]:
                unique_values = df[col].dropna().unique().tolist()

                if len(unique_values) <= 20:
                    selected = st.multiselect(
                        f"Filter by {col}",
                        options=unique_values,
                        default=[],
                        key=f"filter_{col}",
                    )

                    if selected:
                        filtered_df = filtered_df[filtered_df[col].isin(selected)]

    # Add search functionality
    if search_columns:
        search_term = st.text_input("🔍 Search", "", key="table_search")

        if search_term:
            mask = pd.Series([False] * len(filtered_df))
            for col in search_columns:
                if col in filtered_df.columns:
                    mask |= (
                        filtered_df[col].astype(str).str.contains(search_term, case=False, na=False)
                    )
            filtered_df = filtered_df[mask]

    # Display filtered table
    st.dataframe(filtered_df, hide_index=True, use_container_width=True)

    # Show filter info
    st.caption(f"Showing {len(filtered_df)} of {len(df)} rows")

    return filtered_df


def render_sortable_metrics_table(
    df: pd.DataFrame,
    metric_columns: list[str],
    title: str = "Metrics Comparison",
    ascending_better: list[str] | None = None,
) -> None:
    """
    Render a sortable metrics comparison table with best value highlighting.

    Args:
        df: DataFrame with metrics
        metric_columns: Columns containing metrics to compare
        title: Table title
        ascending_better: List of columns where lower is better
    """
    st.markdown(f"### {title}")

    if df.empty:
        st.info("No data to display")
        return

    ascending_better = ascending_better or []

    # Sort control
    col1, col2 = st.columns([3, 1])
    with col1:
        sort_by = st.selectbox(
            "Sort by",
            options=metric_columns,
            index=0,
            key="sort_by_metric",
        )
    with col2:
        ascending = st.checkbox(
            "Ascending",
            value=sort_by in ascending_better,
            key="sort_ascending",
        )

    # Sort DataFrame
    sorted_df = df.sort_values(by=sort_by, ascending=ascending)

    # Highlight best values
    def highlight_best(s: pd.Series) -> list[str]:
        is_ascending = s.name in ascending_better
        if is_ascending:
            best_val = s.min()
        else:
            best_val = s.max()

        return ["background-color: rgba(44, 160, 44, 0.3)" if v == best_val else "" for v in s]

    styler = sorted_df.style

    for col in metric_columns:
        if col in sorted_df.columns:
            styler = styler.apply(highlight_best, subset=[col])

    styler = styler.format("{:.4f}", subset=metric_columns)

    st.dataframe(styler, hide_index=True, use_container_width=True)


# =============================================================================
# EXPORT UTILITIES
# =============================================================================


def add_download_button(
    df: pd.DataFrame,
    filename: str = "data.csv",
    label: str = "📥 Download CSV",
) -> None:
    """
    Add a download button for a DataFrame.

    Args:
        df: DataFrame to download
        filename: Download filename
        label: Button label
    """
    csv = df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label=label,
        data=csv,
        file_name=filename,
        mime="text/csv",
    )


def render_table_with_download(
    df: pd.DataFrame,
    title: str,
    filename: str = "data.csv",
    **kwargs: Any,
) -> None:
    """
    Render a table with a download button.

    Args:
        df: DataFrame to display
        title: Table title
        filename: Download filename
        **kwargs: Additional arguments for render_dataframe
    """
    col1, col2 = st.columns([4, 1])

    with col1:
        st.markdown(f"### {title}")

    with col2:
        add_download_button(df, filename)

    render_dataframe(df, title=None, **kwargs)
