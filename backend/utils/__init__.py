# backend/utils/__init__.py
"""
Utilities Module for Lumina Quant Lab

Provides common utility functions used across the platform:

Validation:
- validate_ticker: Validate ticker format
- validate_date_range: Validate date range logic
- validate_weights: Validate portfolio weights

Formatting:
- format_currency: Format number as currency
- format_percentage: Format number as percentage
- format_date: Consistent date formatting

Calculations:
- annualize_return: Annualize return based on frequency
- compound_return: Calculate compound return
- rolling_apply: Efficient rolling window operations

Caching:
- cache_result: Cache function results
- invalidate_cache: Clear cached results

Usage:
    from backend.utils import (
        validate_ticker,
        validate_date_range,
        format_currency,
        annualize_return,
    )

    # Validate inputs
    validate_ticker("AAPL")  # OK
    validate_date_range(start_date, end_date)  # OK

    # Format outputs
    print(format_currency(10000.50))  # "$10,000.50"
    print(format_percentage(0.1523))  # "15.23%"
"""

from backend.utils.calculations import (
    annualize_return,
    annualize_volatility,
    compound_return,
    max_drawdown,
    sharpe_ratio,
    sortino_ratio,
)
from backend.utils.formatting import (
    format_currency,
    format_date,
    format_number,
    format_percentage,
)
from backend.utils.validation import (
    validate_date_range,
    validate_ticker,
    validate_tickers,
    validate_weights,
)

__all__ = [
    # Validation
    "validate_ticker",
    "validate_tickers",
    "validate_date_range",
    "validate_weights",
    # Formatting
    "format_currency",
    "format_percentage",
    "format_number",
    "format_date",
    # Calculations
    "annualize_return",
    "annualize_volatility",
    "compound_return",
    "sharpe_ratio",
    "sortino_ratio",
    "max_drawdown",
]
