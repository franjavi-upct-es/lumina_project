# backend/utils/calculations.py
"""
Financial calculation utilities for Lumina Quant Lab

Provides common financial calculations used across modules.
"""

import numpy as np
import pandas as pd

ArrayLike = np.ndarray | pd.Series | list | None


def annualize_return(
    returns: ArrayLike,
    periods_per_year: int = 252,
    compounding: bool = True,
) -> float:
    """
    Annualize return from periodic returns.

    Args:
        returns: Array of periodic returns
        periods_per_year: Trading periods per year (252 for daily)
        compounding: Use compound (geometric) return

    Returns:
        Annualized return
    """
    returns = np.array(returns)
    returns = returns[~np.isnan(returns)]

    if len(returns) == 0:
        return 0.0

    if compounding:
        # Geometric mean
        total_return = np.prod(1 + returns) - 1
        n_periods = len(returns)
        annualized = (1 + total_return) ** (periods_per_year / n_periods) - 1
    else:
        # Simple average
        annualized = np.mean(returns) * periods_per_year

    return float(annualized)


def annualize_volatility(
    returns: ArrayLike,
    periods_per_year: int = 252,
) -> float:
    """
    Annualize volatility from periodic returns.

    Args:
        returns: Array of periodic returns
        periods_per_year: Trading periods per year (252 for daily)

    Returns:
        Annualized volatility (standard deviation)
    """
    returns = np.array(returns)
    returns = returns[~np.isnan(returns)]

    if len(returns) < 2:
        return 0.0

    return float(np.std(returns, ddof=1) * np.sqrt(periods_per_year))


def compound_return(returns: ArrayLike) -> float:
    """
    Calculate compound (total) return from periodic returns.

    Args:
        returns: Array of periodic returns

    Returns:
        Total compound return
    """
    returns = np.array(returns)
    returns = returns[~np.isnan(returns)]

    if len(returns) == 0:
        return 0.0

    return float(np.prod(1 + returns) - 1)


def sharpe_ratio(
    returns: ArrayLike,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """
    Calculate Sharpe ratio.

    Args:
        returns: Array of periodic returns
        risk_free_rate: Annual risk-free rate
        periods_per_year: Trading periods per year

    Returns:
        Sharpe ratio
    """
    returns = np.array(returns)
    returns = returns[~np.isnan(returns)]

    if len(returns) < 2:
        return 0.0

    # Annualize return and volatility
    ann_return = annualize_return(returns, periods_per_year)
    ann_vol = annualize_volatility(returns, periods_per_year)

    if ann_vol == 0:
        return 0.0

    return float((ann_return - risk_free_rate) / ann_vol)


def sortino_ratio(
    returns: ArrayLike,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
    target_return: float = 0.0,
) -> float:
    """
    Calculate Sortino ratio (penalizes only downside volatility).

    Args:
        returns: Array of periodic returns
        risk_free_rate: Annual risk-free rate
        periods_per_year: Trading periods per year
        target_return: Minimum acceptable return

    Returns:
        Sortino ratio
    """
    returns = np.array(returns)
    returns = returns[~np.isnan(returns)]

    if len(returns) < 2:
        return 0.0

    # Annualize return
    ann_return = annualize_return(returns, periods_per_year)

    # Downside deviation (only negative returns)
    downside_returns = returns[returns < target_return]
    if len(downside_returns) == 0:
        return float("inf") if ann_return > risk_free_rate else 0.0

    downside_dev = np.std(downside_returns, ddof=1) * np.sqrt(periods_per_year)

    if downside_dev == 0:
        return 0.0

    return float((ann_return - risk_free_rate) / downside_dev)


def max_drawdown(
    returns: ArrayLike = None,
    prices: ArrayLike = None,
) -> float:
    """
    Calculate maximum drawdown.

    Args:
        returns: Array of periodic returns (optional)
        prices: Array of prices (optional, preferred if available)

    Returns:
        Maximum drawdown (negative value, e.g., -0.25 = 25% drawdown)
    """
    # Convert to cumulative values
    if prices is not None:
        prices = np.array(prices)
        prices = prices[~np.isnan(prices)]
    elif returns is not None:
        returns = np.array(returns)
        returns = returns[~np.isnan(returns)]
        prices = np.cumprod(1 + returns)
    else:
        raise ValueError("Either returns or prices must be provided")

    if len(prices) < 2:
        return 0.0

    # Calculate running maximum
    running_max = np.maximum.accumulate(prices)

    # Calculate drawdown at each point
    drawdowns = prices / running_max - 1

    return float(np.min(drawdowns))


def calmar_ratio(
    returns: ArrayLike,
    periods_per_year: int = 252,
) -> float:
    """
    Calculate Calmar ratio (annualized return / max drawdown).

    Args:
        returns: Array of periodic returns
        periods_per_year: Trading periods per year

    Returns:
        Calmar ratio
    """
    returns = np.array(returns)
    returns = returns[~np.isnan(returns)]

    if len(returns) < 2:
        return 0.0

    ann_return = annualize_return(returns, periods_per_year)
    mdd = max_drawdown(returns=returns)

    if mdd == 0:
        return float("inf") if ann_return > 0 else 0.0

    return float(ann_return / abs(mdd))


def information_ratio(
    returns: ArrayLike,
    benchmark_returns: ArrayLike,
    periods_per_year: int = 252,
) -> float:
    """
    Calculate Information ratio (excess return / tracking error).

    Args:
        returns: Portfolio returns
        benchmark_returns: Benchmark returns
        periods_per_year: Trading periods per year

    Returns:
        Information ratio
    """
    returns = np.array(returns)
    benchmark_returns = np.array(benchmark_returns)

    # Align arrays
    min_len = min(len(returns), len(benchmark_returns))
    returns = returns[:min_len]
    benchmark_returns = benchmark_returns[:min_len]

    # Excess returns
    excess_returns = returns - benchmark_returns

    # Annualize
    ann_excess_return = annualize_return(excess_returns, periods_per_year)
    tracking_error = annualize_volatility(excess_returns, periods_per_year)

    if tracking_error == 0:
        return 0.0

    return float(ann_excess_return / tracking_error)


def value_at_risk(
    returns: ArrayLike,
    confidence: float = 0.95,
    method: str = "historical",
) -> float:
    """
    Calculate Value at Risk.

    Args:
        returns: Array of returns
        confidence: Confidence level (e.g., 0.95 for 95%)
        method: "historical" or "parametric"

    Returns:
        VaR (negative value representing potential loss)
    """
    returns = np.array(returns)
    returns = returns[~np.isnan(returns)]

    if len(returns) < 2:
        return 0.0

    if method == "historical":
        # Historical VaR (percentile method)
        return float(np.percentile(returns, (1 - confidence) * 100))

    elif method == "parametric":
        # Parametric VaR (assuming normal distribution)
        from scipy import stats

        mean = np.mean(returns)
        std = np.std(returns, ddof=1)
        return float(stats.norm.ppf(1 - confidence, mean, std))

    else:
        raise ValueError(f"Unknown VaR method: {method}")


def expected_shortfall(
    returns: ArrayLike,
    confidence: float = 0.95,
) -> float:
    """
    Calculate Expected Shortfall (Conditional VaR).

    Average loss beyond the VaR threshold.

    Args:
        returns: Array of returns
        confidence: Confidence level

    Returns:
        Expected shortfall (negative value)
    """
    returns = np.array(returns)
    returns = returns[~np.isnan(returns)]

    if len(returns) < 2:
        return 0.0

    var = value_at_risk(returns, confidence, method="historical")

    # Average of returns worse than VaR
    tail_returns = returns[returns <= var]

    if len(tail_returns) == 0:
        return var

    return float(np.mean(tail_returns))


def beta(
    returns: ArrayLike,
    benchmark_returns: ArrayLike,
) -> float:
    """
    Calculate portfolio beta vs benchmark.

    Args:
        returns: Portfolio returns
        benchmark_returns: Benchmark returns

    Returns:
        Beta coefficient
    """
    returns = np.array(returns)
    benchmark_returns = np.array(benchmark_returns)

    # Align arrays
    min_len = min(len(returns), len(benchmark_returns))
    returns = returns[:min_len]
    benchmark_returns = benchmark_returns[:min_len]

    # Remove NaNs
    mask = ~(np.isnan(returns) | np.isnan(benchmark_returns))
    returns = returns[mask]
    benchmark_returns = benchmark_returns[mask]

    if len(returns) < 2:
        return 1.0

    covariance = np.cov(returns, benchmark_returns)[0, 1]
    benchmark_variance = np.var(benchmark_returns, ddof=1)

    if benchmark_variance == 0:
        return 1.0

    return float(covariance / benchmark_variance)


def rolling_sharpe(
    returns: ArrayLike,
    window: int = 252,
    risk_free_rate: float = 0.0,
) -> np.ndarray:
    """
    Calculate rolling Sharpe ratio.

    Args:
        returns: Array of returns
        window: Rolling window size
        risk_free_rate: Annual risk-free rate

    Returns:
        Array of rolling Sharpe ratios
    """
    returns = pd.Series(returns)

    rolling_mean = returns.rolling(window).mean() * 252
    rolling_std = returns.rolling(window).std() * np.sqrt(252)

    sharpe = (rolling_mean - risk_free_rate) / rolling_std

    return sharpe.values


def win_rate(returns: ArrayLike) -> float:
    """
    Calculate win rate (percentage of positive returns).

    Args:
        returns: Array of returns

    Returns:
        Win rate as decimal (0.0 to 1.0)
    """
    returns = np.array(returns)
    returns = returns[~np.isnan(returns)]

    if len(returns) == 0:
        return 0.0

    return float(np.sum(returns > 0) / len(returns))


def profit_factor(returns: ArrayLike) -> float:
    """
    Calculate profit factor (gross profits / gross losses).

    Args:
        returns: Array of returns

    Returns:
        Profit factor
    """
    returns = np.array(returns)
    returns = returns[~np.isnan(returns)]

    gains = returns[returns > 0]
    losses = returns[returns < 0]

    gross_profit = np.sum(gains) if len(gains) > 0 else 0
    gross_loss = abs(np.sum(losses)) if len(losses) > 0 else 0

    if gross_loss == 0:
        return float("inf") if gross_profit > 0 else 0.0

    return float(gross_profit / gross_loss)
