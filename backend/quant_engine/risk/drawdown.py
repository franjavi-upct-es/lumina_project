# backend/quant_engine/risk/drawdown.py
"""
Drawdown Analysis Module

Analyzes portfolio drawdowns - the peak-to-trough decline in portfolio value.
Drawdown analysis is crucial for understanding:
- Maximum loss from peak
- Recovery time
- Risk-adjusted performance
- Psychological impact on investors

Metrics calculated:
- Maximum Drawdown (MDD)
- Average Drawdown
- Drawdown Duration
- Calmar Ratio (Return/MDD)
- Ulcer Index
- Recovery periods

References:
- Martin, P., & McCann, B. (1989). The investor's guide to fidelity funds.
- Young, T. W. (1991). Calmar ratio: A smoother tool.
"""

from typing import Any

import numpy as np
import pandas as pd
import polars as pl
from loguru import logger


class DrawdownAnalyzer:
    """
    Comprehensive drawdown analysis for portfolios and strategies

    Provides detailed analysis of drawdown patterns including:
    - Maximum drawdown identification
    - Drawdown duration analysis
    - Recovery time calculation
    - Underwater period analysis
    """

    def __init__(self):
        """Initialize drawdown analyzer"""
        self.equity_curve: np.ndarray | None = None
        self.drawdowns: np.ndarray | None = None
        self.drawdown_series: pd.Series | None = None

        logger.info("Drawdown analyzer initialized")

    def calculate_drawdowns(
        self,
        prices: np.ndarray | pd.Series | pl.Series,
    ) -> np.ndarray:
        """
        Calculate drawdown series from price/equity curve

        Drawdown = (Current Value - Peak Value) / Peak Value

        Args:
            prices: Price or equity curve series

        Returns:
            Array of drawdown values (negative values)
        """
        # Convert to numpy array
        if isinstance(prices, (pd.Series, pl.Series)):
            prices = prices.to_numpy()

        # Calculate running maximum (peak)
        running_max = np.maximum.accumulate(prices)

        # Calculate drawdown
        drawdowns = (prices - running_max) / running_max

        self.equity_curve = prices
        self.drawdowns = drawdowns

        return drawdowns

    def get_max_drawdown(
        self,
        prices: np.ndarray | pd.Series | pl.Series | None = None,
    ) -> dict[str, Any]:
        """
        Calculate maximum drawdown and related information

        Args:
            prices: Price/equity curve (if None, uses previously calculated)

        Returns:
            Dictionary with max drawdown information
        """
        if prices is not None:
            self.calculate_drawdowns(prices)

        if self.drawdowns is None:
            raise ValueError(
                "Drawdowns not calculated. Provide prices or call calculate_drawdowns first."
            )

        # Find maximum drawdown
        max_dd_idx = np.argmin(self.drawdowns)
        max_dd = self.drawdowns[max_dd_idx]

        # Find the peak before max drawdown
        peak_idx = np.argmax(self.equity_curve[: max_dd_idx + 1])
        peak_value = self.equity_curve[peak_idx]
        trough_value = self.equity_curve[max_dd_idx]

        # Calculate recovery information
        recovery_idx = None
        recovery_days = None

        if max_dd_idx < len(self.equity_curve) - 1:
            # Look for recovery (price exceeding peak)
            future_prices = self.equity_curve[max_dd_idx + 1 :]
            recovery_mask = future_prices >= peak_value

            if np.any(recovery_mask):
                recovery_idx = max_dd_idx + 1 + np.argmax(recovery_mask)
                recovery_days = recovery_idx - max_dd_idx

        # Duration of drawdown (peak to trough)
        drawdown_duration = max_dd_idx - peak_idx

        logger.info(f"Max Drawdown: {max_dd:.2%} (duration: {drawdown_duration} periods)")

        return {
            "max_drawdown": max_dd,
            "max_drawdown_pct": max_dd * 100,
            "peak_idx": peak_idx,
            "trough_idx": max_dd_idx,
            "peak_value": peak_value,
            "trough_value": trough_value,
            "drawdown_duration": drawdown_duration,
            "recovery_idx": recovery_idx,
            "recovery_duration": recovery_days,
            "recovered": recovery_idx is not None,
        }

    def get_drawdown_periods(
        self,
        prices: np.ndarray | pd.Series | pl.Series | None = None,
        threshold: float = -0.01,
    ) -> list[dict[str, Any]]:
        """
        Identify all drawdown periods

        Args:
            prices: Price/equity curve
            threshold: Minimum drawdown to consider (e.g., -0.01 for 1%)

        Returns:
            List of drawdown period dictionaries
        """
        if prices is not None:
            self.calculate_drawdowns(prices)

        if self.drawdowns is None:
            raise ValueError("Drawdowns not calculated")

        drawdown_periods = []
        in_drawdown = False
        current_period = None

        for i, dd in enumerate(self.drawdowns):
            if dd <= threshold and not in_drawdown:
                # Start of drawdown period
                in_drawdown = True
                peak_idx = np.argmax(self.equity_curve[: i + 1])
                current_period = {
                    "peak_idx": peak_idx,
                    "peak_value": self.equity_curve[peak_idx],
                    "start_idx": i,
                    "trough_idx": i,
                    "trough_value": self.equity_curve[i],
                    "min_drawdown": dd,
                }

            elif in_drawdown:
                # Update trough if deeper drawdown
                if dd < current_period["min_drawdown"]:
                    current_period["trough_idx"] = i
                    current_period["trough_value"] = self.equity_curve[i]
                    current_period["min_drawdown"] = dd

                # Check for recovery
                if dd >= 0:
                    # End of drawdown period
                    current_period["end_idx"] = i
                    current_period["recovery_value"] = self.equity_curve[i]
                    current_period["duration"] = i - current_period["start_idx"]
                    current_period["recovery_duration"] = i - current_period["trough_idx"]

                    drawdown_periods.append(current_period)
                    in_drawdown = False
                    current_period = None

        # Handle ongoing drawdown
        if in_drawdown and current_period is not None:
            current_period["end_idx"] = len(self.equity_curve) - 1
            current_period["recovery_value"] = None
            current_period["duration"] = len(self.equity_curve) - current_period["start_idx"]
            current_period["recovery_duration"] = None
            current_period["ongoing"] = True
            drawdown_periods.append(current_period)

        logger.info(f"Identified {len(drawdown_periods)} drawdown periods")

        return drawdown_periods

    def calculate_average_drawdown(
        self,
        prices: np.ndarray | pd.Series | pl.Series | None = None,
    ) -> float:
        """
        Calculate average drawdown

        Args:
            prices: Price/equity curve

        Returns:
            Average drawdown value
        """
        if prices is not None:
            self.calculate_drawdowns(prices)

        if self.drawdowns is None:
            raise ValueError("Drawdowns not calculated")

        # Average of all drawdown values
        avg_dd = np.mean(self.drawdowns)

        return avg_dd

    def calculate_ulcer_index(
        self,
        prices: np.ndarray | pd.Series | pl.Series | None = None,
        period: int | None = None,
    ) -> float:
        """
        Calculate Ulcer Index - measures depth and duration of drawdowns

        UI = sqrt(sum((100 * drawdown)^2) / N)

        Args:
            prices: Price/equity curve
            period: Number of periods to use (None = all data)

        Returns:
            Ulcer Index value
        """
        if prices is not None:
            self.calculate_drawdowns(prices)

        if self.drawdowns is None:
            raise ValueError("Drawdowns not calculated")

        # Select period
        if period is not None:
            drawdowns_subset = self.drawdowns[-period:]
        else:
            drawdowns_subset = self.drawdowns

        # Ulcer Index calculation
        squared_drawdowns = (drawdowns_subset * 100) ** 2
        ulcer_index = np.sqrt(np.mean(squared_drawdowns))

        logger.debug(f"Ulcer Index: {ulcer_index:.2f}")

        return ulcer_index

    def calculate_calmar_ratio(
        self,
        returns: np.ndarray | pd.Series | pl.Series,
        prices: np.ndarray | pd.Series | pl.Series | None = None,
        period_years: float = 3.0,
    ) -> float:
        """
        Calculate Calmar Ratio = Annualized Return / Maximum Drawdown

        Args:
            returns: Return series
            prices: Price/equity curve (for max drawdown calculation)
            period_years: Number of years for annualized return

        Returns:
            Calmar ratio
        """
        # Convert to numpy
        if isinstance(returns, (pd.Series, pl.Series)):
            returns = returns.to_numpy()

        # Calculate annualized return
        total_return = np.prod(1 + returns) - 1
        annualized_return = (1 + total_return) ** (1 / period_years) - 1

        # Get max drawdown
        if prices is not None:
            max_dd_info = self.get_max_drawdown(prices)
        else:
            max_dd_info = self.get_max_drawdown()

        max_dd = abs(max_dd_info["max_drawdown"])

        # Calmar ratio
        if max_dd == 0:
            return np.inf

        calmar = annualized_return / max_dd

        logger.info(f"Calmar Ratio: {calmar:.2f}")

        return calmar

    def calculate_underwater_periods(
        self,
        prices: np.ndarray | pd.Series | pl.Series | None = None,
    ) -> dict[str, Any]:
        """
        Calculate statistics about underwater periods (when below peak)

        Args:
            prices: Price/equity curve

        Returns:
            Dictionary with underwater period statistics
        """
        if prices is not None:
            self.calculate_drawdowns(prices)

        if self.drawdowns is None:
            raise ValueError("Drawdowns not calculated")

        # Identify underwater periods (drawdown < 0)
        underwater_mask = self.drawdowns < 0

        # Calculate run lengths
        underwater_runs = []
        current_run = 0

        for is_underwater in underwater_mask:
            if is_underwater:
                current_run += 1
            else:
                if current_run > 0:
                    underwater_runs.append(current_run)
                current_run = 0

        # Add final run if still underwater
        if current_run > 0:
            underwater_runs.append(current_run)

        # Statistics
        if underwater_runs:
            stats = {
                "pct_time_underwater": np.sum(underwater_mask) / len(underwater_mask) * 100,
                "n_underwater_periods": len(underwater_runs),
                "avg_underwater_duration": np.mean(underwater_runs),
                "max_underwater_duration": np.max(underwater_runs),
                "min_underwater_duration": np.min(underwater_runs),
                "median_underwater_duration": np.median(underwater_runs),
            }
        else:
            stats = {
                "pct_time_underwater": 0,
                "n_underwater_periods": 0,
                "avg_underwater_duration": 0,
                "max_underwater_duration": 0,
                "min_underwater_duration": 0,
                "median_underwater_duration": 0,
            }

        logger.info(f"Underwater {stats['pct_time_underwater']:.1f}% of time")

        return stats

    def get_full_analysis(
        self,
        prices: np.ndarray | pd.Series | pl.Series,
        returns: np.ndarray | pd.Series | pl.Series | None = None,
    ) -> dict[str, Any]:
        """
        Perform complete drawdown analysis

        Args:
            prices: Price/equity curve
            returns: Return series (for Calmar ratio)

        Returns:
            Comprehensive drawdown analysis dictionary
        """
        # Calculate drawdowns
        self.calculate_drawdowns(prices)

        # Get all metrics
        max_dd_info = self.get_max_drawdown()
        avg_dd = self.calculate_average_drawdown()
        ulcer_index = self.calculate_ulcer_index()
        underwater_stats = self.calculate_underwater_periods()
        drawdown_periods = self.get_drawdown_periods()

        analysis = {
            "max_drawdown": max_dd_info,
            "average_drawdown": avg_dd,
            "average_drawdown_pct": avg_dd * 100,
            "ulcer_index": ulcer_index,
            "underwater_stats": underwater_stats,
            "n_drawdown_periods": len(drawdown_periods),
            "drawdown_periods": drawdown_periods,
        }

        # Add Calmar ratio if returns provided
        if returns is not None:
            # Estimate period in years
            period_years = len(prices) / 252  # Assuming daily data
            calmar = self.calculate_calmar_ratio(returns, period_years=period_years)
            analysis["calmar_ratio"] = calmar

        logger.success("Complete drawdown analysis finished")

        return analysis

    def plot_drawdown(
        self,
        prices: np.ndarray | pd.Series | pl.Series | None = None,
    ) -> dict[str, np.ndarray]:
        """
        Prepare data for drawdown visualization

        Args:
            prices: Price/equity curve

        Returns:
            Dictionary with arrays for plotting
        """
        if prices is not None:
            self.calculate_drawdowns(prices)

        if self.drawdowns is None or self.equity_curve is None:
            raise ValueError("Drawdowns not calculated")

        # Running maximum for reference
        running_max = np.maximum.accumulate(self.equity_curve)

        return {
            "equity_curve": self.equity_curve,
            "running_max": running_max,
            "drawdown": self.drawdowns * 100,  # Convert to percentage
            "time_index": np.arange(len(self.equity_curve)),
        }


def calculate_max_drawdown(
    prices: np.ndarray | pd.Series | pl.Series,
) -> float:
    """
    Quick calculation of maximum drawdown

    Args:
        prices: Price or equity curve

    Returns:
        Maximum drawdown (negative value)

    Example:
        >>> prices = np.array([100, 110, 105, 95, 100, 120])
        >>> max_dd = calculate_max_drawdown(prices)
        >>> print(f"Max Drawdown: {max_dd:.2%}")
    """
    analyzer = DrawdownAnalyzer()
    result = analyzer.get_max_drawdown(prices)
    return result["max_drawdown"]


def analyze_drawdowns(
    prices: np.ndarray | pd.Series | pl.Series,
    returns: np.ndarray | pd.Series | pl.Series | None = None,
) -> dict[str, Any]:
    """
    Convenience function for full drawdown analysis

    Args:
        prices: Price or equity curve
        returns: Return series (optional, for additional metrics)

    Returns:
        Complete drawdown analysis

    Example:
        >>> import numpy as np
        >>> prices = np.cumprod(1 + np.random.normal(0.001, 0.02, 1000))
        >>> analysis = analyze_drawdowns(prices)
        >>> print(f"Max DD: {analysis['max_drawdown']['max_drawdown_pct']:.2f}%")
        >>> print(f"Avg DD: {analysis['average_drawdown_pct']:.2f}%")
        >>> print(f"Time underwater: {analysis['underwater_stats']['pct_time_underwater']:.1f}%")
    """
    analyzer = DrawdownAnalyzer()
    return analyzer.get_full_analysis(prices, returns)
