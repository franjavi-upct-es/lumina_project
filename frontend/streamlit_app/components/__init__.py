# frontend/streamlit-app/components/__init__.py
"""
Lumina Quant Lab - Reusable Streamlit Components

This module provides reusable UI components for the Lumina dashboard:
- charts: Plotly chart components for financial data visualization
- metrics: Metric display components for KPIs and statistics
- tables: Data table components with styling and interactivity
"""

from frontend.streamlit_app.components.charts import (
    # Constants
    COLORS,
    DEFAULT_TEMPLATE,
    # Utilities
    add_annotations,
    # Price charts
    create_candlestick_chart,
    create_correlation_heatmap,
    # Risk charts
    create_drawdown_chart,
    # Performance charts
    create_equity_curve,
    create_line_chart,
    create_ohlc_with_indicators,
    create_returns_distribution,
    create_risk_contribution_chart,
    create_rolling_metrics_chart,
    create_sentiment_by_source_chart,
    # Sentiment charts
    create_sentiment_gauge,
    create_sentiment_timeline,
    create_var_chart,
    update_chart_theme,
)
from frontend.streamlit_app.components.metrics import (
    # Data classes
    MetricConfig,
    PerformanceMetrics,
    RiskMetrics,
    SentimentMetrics,
    calculate_performance_metrics,
    # Calculators
    calculate_risk_metrics,
    # Formatters
    format_currency,
    format_number,
    format_percentage,
    format_ratio,
    render_fear_greed_gauge,
    # Metric cards
    render_metric_card,
    render_metric_row,
    # Performance metrics
    render_performance_metrics,
    render_performance_summary_card,
    # Risk metrics
    render_risk_metrics,
    # Sentiment metrics
    render_sentiment_metrics,
    render_styled_metric_card,
    render_var_summary,
)
from frontend.streamlit_app.components.tables import (
    # Export utilities
    add_download_button,
    background_positive_negative,
    # Styling functions
    color_positive_negative,
    color_sentiment,
    # Formatters
    format_currency_col,
    format_date_col,
    format_datetime_col,
    format_number_col,
    format_percentage_col,
    render_correlation_table,
    # Basic tables
    render_dataframe,
    # Interactive tables
    render_filterable_table,
    render_model_comparison_table,
    render_portfolio_table,
    # Specialized tables
    render_price_table,
    render_risk_table,
    render_sentiment_table,
    render_sortable_metrics_table,
    render_stress_test_table,
    render_styled_table,
    render_table_with_download,
    render_trades_table,
)

__all__ = [
    # Charts
    "create_candlestick_chart",
    "create_line_chart",
    "create_ohlc_with_indicators",
    "create_drawdown_chart",
    "create_var_chart",
    "create_correlation_heatmap",
    "create_risk_contribution_chart",
    "create_sentiment_gauge",
    "create_sentiment_timeline",
    "create_sentiment_by_source_chart",
    "create_equity_curve",
    "create_returns_distribution",
    "create_rolling_metrics_chart",
    "add_annotations",
    "update_chart_theme",
    "COLORS",
    "DEFAULT_TEMPLATE",
    # Metrics
    "MetricConfig",
    "RiskMetrics",
    "PerformanceMetrics",
    "SentimentMetrics",
    "format_currency",
    "format_percentage",
    "format_number",
    "format_ratio",
    "render_metric_card",
    "render_metric_row",
    "render_styled_metric_card",
    "render_risk_metrics",
    "render_var_summary",
    "render_performance_metrics",
    "render_performance_summary_card",
    "render_sentiment_metrics",
    "render_fear_greed_gauge",
    "calculate_risk_metrics",
    "calculate_performance_metrics",
    # Tables
    "format_currency_col",
    "format_percentage_col",
    "format_number_col",
    "format_datetime_col",
    "format_date_col",
    "color_positive_negative",
    "background_positive_negative",
    "color_sentiment",
    "render_dataframe",
    "render_styled_table",
    "render_price_table",
    "render_trades_table",
    "render_risk_table",
    "render_correlation_table",
    "render_portfolio_table",
    "render_model_comparison_table",
    "render_sentiment_table",
    "render_stress_test_table",
    "render_filterable_table",
    "render_sortable_metrics_table",
    "add_download_button",
    "render_table_with_download",
]
