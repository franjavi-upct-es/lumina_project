# backend/utils/validation.py
"""
Input validation utilities for Lumina Quant Lab

Provides consistent validation across all modules.
"""

import re
from datetime import datetime, timedelta
from typing import Any


class ValidationError(Exception):
    """Custom exception for validation errors"""

    pass


def validate_ticker(ticker: str) -> str:
    """
    Validate and normalize a stock ticker symbol.

    Args:
        ticker: Ticker symbol to validate

    Returns:
        Normalized ticker (uppercase)

    Raises:
        ValidationError: If ticker is invalid
    """
    if not ticker:
        raise ValidationError("Ticker cannot be empty")

    # Normalize to uppercase
    ticker = ticker.upper().strip()

    # Basic validation: 1-10 alphanumeric characters
    if not re.match(r"^[A-Z0-9.]{1,10}$", ticker):
        raise ValidationError(
            f"Invalid ticker format: {ticker}. Must be 1-10 alphanumeric characters."
        )

    return ticker


def validate_tickers(tickers: list[str]) -> list[str]:
    """
    Validate a list of ticker symbols.

    Args:
        tickers: List of ticker symbols

    Returns:
        List of normalized tickers

    Raises:
        ValidationError: If any ticker is invalid
    """
    if not tickers:
        raise ValidationError("Tickers list cannot be empty")

    if not isinstance(tickers, list):
        raise ValidationError("Tickers must be a list")

    validated = []
    for ticker in tickers:
        validated.append(validate_ticker(ticker))

    # Check for duplicates
    if len(validated) != len(set(validated)):
        raise ValidationError("Duplicate tickers found")

    return validated


def validate_date_range(
    start_date: datetime | str,
    end_date: datetime | str,
    max_days: int | None = None,
    min_days: int | None = None,
) -> tuple[datetime, datetime]:
    """
    Validate a date range.

    Args:
        start_date: Start date (datetime or ISO string)
        end_date: End date (datetime or ISO string)
        max_days: Maximum allowed days in range
        min_days: Minimum required days in range

    Returns:
        Tuple of (start_date, end_date) as datetime objects

    Raises:
        ValidationError: If date range is invalid
    """
    # Parse strings to datetime
    if isinstance(start_date, str):
        try:
            start_date = datetime.fromisoformat(start_date.replace("Z", "+00:00"))
        except ValueError as e:
            raise ValidationError(f"Invalid start_date format: {e}")

    if isinstance(end_date, str):
        try:
            end_date = datetime.fromisoformat(end_date.replace("Z", "+00:00"))
        except ValueError as e:
            raise ValidationError(f"Invalid end_date format: {e}")

    # Validate order
    if start_date >= end_date:
        raise ValidationError("start_date must be before end_date")

    # Validate range
    days_diff = (end_date - start_date).days

    if max_days and days_diff > max_days:
        raise ValidationError(f"Date range ({days_diff} days) exceeds maximum ({max_days} days)")

    if min_days and days_diff < min_days:
        raise ValidationError(
            f"Date range ({days_diff} days) is less than minimum ({min_days} days)"
        )

    # Check not in future
    if end_date > datetime.now() + timedelta(days=1):
        raise ValidationError("end_date cannot be in the future")

    return start_date, end_date


def validate_weights(
    weights: dict[str, float] | list[float],
    tickers: list[str] | None = None,
    must_sum_to_one: bool = True,
    allow_negative: bool = False,
    allow_leverage: bool = False,
) -> dict[str, float]:
    """
    Validate portfolio weights.

    Args:
        weights: Weights as dict {ticker: weight} or list
        tickers: Required if weights is a list
        must_sum_to_one: Whether weights must sum to 1.0
        allow_negative: Allow negative weights (short positions)
        allow_leverage: Allow sum > 1.0

    Returns:
        Normalized weights dictionary

    Raises:
        ValidationError: If weights are invalid
    """
    # Convert list to dict if needed
    if isinstance(weights, list):
        if not tickers or len(tickers) != len(weights):
            raise ValidationError(
                "When weights is a list, tickers must be provided with matching length"
            )
        weights = dict(zip(tickers, weights))

    if not weights:
        raise ValidationError("Weights cannot be empty")

    # Validate individual weights
    for ticker, weight in weights.items():
        if not isinstance(weight, (int, float)):
            raise ValidationError(f"Weight for {ticker} must be numeric")

        if not allow_negative and weight < 0:
            raise ValidationError(f"Negative weight not allowed for {ticker}")

        if weight > 1.0 and not allow_leverage:
            raise ValidationError(f"Weight for {ticker} ({weight}) exceeds 1.0")

    # Validate sum
    total = sum(weights.values())

    if must_sum_to_one:
        if abs(total - 1.0) > 0.001:
            raise ValidationError(f"Weights must sum to 1.0, got {total:.4f}")

    if not allow_leverage and total > 1.001:
        raise ValidationError(f"Total weights ({total:.4f}) imply leverage")

    return weights


def validate_model_type(model_type: str) -> str:
    """
    Validate ML model type.

    Args:
        model_type: Type of model

    Returns:
        Normalized model type

    Raises:
        ValidationError: If model type is invalid
    """
    valid_types = ["lstm", "transformer", "xgboost", "ensemble"]

    model_type = model_type.lower().strip()

    if model_type not in valid_types:
        raise ValidationError(f"Invalid model type: {model_type}. Must be one of: {valid_types}")

    return model_type


def validate_strategy(strategy_name: str) -> str:
    """
    Validate backtesting strategy name.

    Args:
        strategy_name: Name of strategy

    Returns:
        Normalized strategy name

    Raises:
        ValidationError: If strategy is invalid
    """
    valid_strategies = [
        "rsi",
        "macd",
        "ma_crossover",
        "bollinger_bands",
        "mean_reversion",
        "momentum",
        "combo",
        "custom",
    ]

    strategy_name = strategy_name.lower().strip()

    if strategy_name not in valid_strategies:
        raise ValidationError(
            f"Invalid strategy: {strategy_name}. Must be one of: {valid_strategies}"
        )

    return strategy_name


def validate_hyperparameters(
    hyperparams: dict[str, Any],
    required: list[str] | None = None,
    constraints: dict[str, tuple] | None = None,
) -> dict[str, Any]:
    """
    Validate model hyperparameters.

    Args:
        hyperparams: Hyperparameters dictionary
        required: List of required parameter names
        constraints: Dict of {param: (min, max)} constraints

    Returns:
        Validated hyperparameters

    Raises:
        ValidationError: If hyperparameters are invalid
    """
    if not isinstance(hyperparams, dict):
        raise ValidationError("Hyperparameters must be a dictionary")

    # Check required parameters
    if required:
        missing = [p for p in required if p not in hyperparams]
        if missing:
            raise ValidationError(f"Missing required parameters: {missing}")

    # Check constraints
    if constraints:
        for param, (min_val, max_val) in constraints.items():
            if param in hyperparams:
                value = hyperparams[param]
                if value < min_val or value > max_val:
                    raise ValidationError(
                        f"Parameter {param}={value} out of range [{min_val}, {max_val}]"
                    )

    return hyperparams
