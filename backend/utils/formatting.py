# backend/utils/formatting.py
"""
Output formatting utilities for Lumina Quant Lab

Provides consistent formatting across all outputs.
"""

from datetime import datetime
from typing import Any


def format_currency(
    value: float | int,
    currency: str = "$",
    decimal_places: int = 2,
    include_sign: bool = False,
) -> str:
    """
    Format a number as currency.

    Args:
        value: Numeric value to format
        currency: Currency symbol
        decimal_places: Number of decimal places
        include_sign: Include + for positive values

    Returns:
        Formatted currency string

    Examples:
        format_currency(10000.50) -> "$10,000.50"
        format_currency(-500, include_sign=True) -> "-$500.00"
    """
    if value is None:
        return f"{currency}"

    sign = ""
    if value < 0:
        sign = "-"
        value = abs(value)
    elif include_sign and value > 0:
        sign = "+"

    formatted = f"{value:,.{decimal_places}f}"
    return f"{sign}{currency}{formatted}"


def format_percentage(
    value: float | int,
    decimal_places: int = 2,
    include_sign: bool = True,
    multiply_by_100: bool = True,
) -> str:
    """
    Format a number as percentage.

    Args:
        value: Numeric value (0.15 = 15% if multiply_by_100=True)
        decimal_places: Number of decimal places
        include_sign: Include + for positive values
        multiply_by_100: If True, multiply value by 100

    Returns:
        Formatted percentage string

    Examples:
        format_percentage(0.1523) -> "+15.23%"
        format_percentage(-0.05) -> "-5.00%"
    """
    if value is None:
        return "--%"

    if multiply_by_100:
        value = value * 100

    sign = ""
    if value > 0 and include_sign:
        sign = "+"
    elif value < 0:
        sign = ""  # Negative sign included in value

    return f"{sign}{value:.{decimal_places}f}%"


def format_number(
    value: float | int,
    decimal_places: int = 2,
    use_thousands_separator: bool = True,
    abbreviate: bool = False,
) -> str:
    """
    Format a number with various options.

    Args:
        value: Numeric value
        decimal_places: Number of decimal places
        use_thousands_separator: Use comma separators
        abbreviate: Abbreviate large numbers (K, M, B)

    Returns:
        Formatted number string

    Examples:
        format_number(1234567.89) -> "1,234,567.89"
        format_number(1234567.89, abbreviate=True) -> "1.23M"
    """
    if value is None:
        return "--"

    if abbreviate:
        abs_value = abs(value)
        sign = "-" if value < 0 else ""

        if abs_value >= 1_000_000_000:
            return f"{sign}{abs_value / 1_000_000_000:.{decimal_places}f}B"
        elif abs_value >= 1_000_000:
            return f"{sign}{abs_value / 1_000_000:.{decimal_places}f}M"
        elif abs_value >= 1_000:
            return f"{sign}{abs_value / 1_000:.{decimal_places}f}K"

    if use_thousands_separator:
        return f"{value:,.{decimal_places}f}"

    return f"{value:.{decimal_places}f}"


def format_date(
    date: datetime | str,
    format_string: str = "%Y-%m-%d",
    include_time: bool = False,
) -> str:
    """
    Format a date consistently.

    Args:
        date: Date to format (datetime or ISO string)
        format_string: strftime format string
        include_time: Include time in output

    Returns:
        Formatted date string
    """
    if date is None:
        return "--"

    if isinstance(date, str):
        date = datetime.fromisoformat(date.replace("Z", "+00:00"))

    if include_time:
        return date.strftime(f"{format_string} %H:%M:%S")

    return date.strftime(format_string)


def format_duration(seconds: float) -> str:
    """
    Format a duration in seconds to human-readable string.

    Args:
        seconds: Duration in seconds

    Returns:
        Formatted duration string

    Examples:
        format_duration(125.5) -> "2m 5.5s"
        format_duration(3661) -> "1h 1m 1s"
    """
    if seconds is None:
        return "--"

    if seconds < 60:
        return f"{seconds:.1f}s"

    minutes, secs = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)

    if hours > 0:
        return f"{int(hours)}h {int(minutes)}m {int(secs)}s"

    return f"{int(minutes)}m {secs:.1f}s"


def format_ratio(
    value: float,
    decimal_places: int = 2,
    include_sign: bool = True,
) -> str:
    """
    Format a ratio (like Sharpe ratio).

    Args:
        value: Ratio value
        decimal_places: Number of decimal places
        include_sign: Include + for positive values

    Returns:
        Formatted ratio string
    """
    if value is None:
        return "--"

    sign = ""
    if value > 0 and include_sign:
        sign = "+"

    return f"{sign}{value:.{decimal_places}f}"


def format_table_row(
    values: list[Any],
    widths: list[int] | None = None,
    alignment: str = "left",
) -> str:
    """
    Format a row of values for table display.

    Args:
        values: List of values
        widths: Column widths (auto if None)
        alignment: 'left', 'right', or 'center'

    Returns:
        Formatted row string
    """
    if widths is None:
        widths = [max(len(str(v)), 10) for v in values]

    formatted = []
    for value, width in zip(values, widths):
        str_value = str(value)
        if alignment == "right":
            formatted.append(str_value.rjust(width))
        elif alignment == "center":
            formatted.append(str_value.center(width))
        else:
            formatted.append(str_value.ljust(width))

    return " | ".join(formatted)


def format_metric_summary(metrics: dict[str, float]) -> str:
    """
    Format a dictionary of metrics as a summary string.

    Args:
        metrics: Dictionary of metric name -> value

    Returns:
        Formatted summary string
    """
    lines = []

    for name, value in metrics.items():
        # Determine formatting based on name
        if "return" in name.lower() or "rate" in name.lower():
            formatted_value = format_percentage(value)
        elif "sharpe" in name.lower() or "ratio" in name.lower():
            formatted_value = format_ratio(value)
        elif "drawdown" in name.lower():
            formatted_value = format_percentage(value)
        elif any(x in name.lower() for x in ["capital", "pnl", "profit", "loss"]):
            formatted_value = format_currency(value)
        else:
            formatted_value = format_number(value)

        # Format name (snake_case to Title Case)
        formatted_name = name.replace("_", " ").title()

        lines.append(f"  {formatted_name}: {formatted_value}")

    return "\n".join(lines)
