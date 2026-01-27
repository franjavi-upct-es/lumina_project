# backend/backtesting/vectorized.py
"""
Vectorized backtesting engine for fast strategy evaluation
Uses vectorized operations with numpy/pandas for maximum speed
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger


@dataclass
class VectorizedBacktestConfig:
    """Configuration for vectorized backtest"""

    initial_capital: float = 100000.0
    commission: float = 0.001  # 0.1%
    slippage: float = 0.0005  # 0.05%

    # Position sizing
    position_size: float = 1.0  # Fraction of capital per position
    max_positions: int = 1  # For multi-asset strategies

    # Risk management
    stop_loss: float | None = None  # Fraction (e.g., 0.05 = 5%)
    take_profit: float | None = None
    max_holding_period: int | None = None  # In bars

    # Rebalancing
    rebalance_frequency: str = "daily"  # 'daily', 'weekly', 'monthly'

    # Leverage
    max_leverage: float = 1.0


class VectorizedBacktest:
    """
    Fast vectorized backtesting engine

    Processes entire price history at once using numpy/pandas operations
    Much faster than event-driven but less flexible

    Best for:
    - Rapid strategy testing
    - Parameter optimization
    - Simple strategies without complex logic
    """

    def __init__(self, config: VectorizedBacktestConfig | None = None):
        self.config = config or VectorizedBacktestConfig()

        # Results storage
        self.results: dict[str, Any] | None = None
        self.equity_curve: pd.Series | None = None
        self.positions: pd.DataFrame | None = None
        self.trades: pd.DataFrame | None = None

    def run(self, prices: pd.DataFrame, signals: pd.DataFrame, **kwargs) -> dict[str, Any]:
        """
        Run vectorized backtest

        Args:
            prices: DataFrame with OHLCV data (index: datetime, columns: tickers)
            signals: DataFrame with trading signals (1=long, 0=neutral, -1=short)
            **kwargs: Additional parameters

        Returns:
            Dictionary with backtest results
        """
        logger.info(f"Running vectorized backtest: {len(prices)} bars")

        # Align signals with prices
        signals = signals.reindex(prices.index, method="ffill").fillna(0)

        # Calculate returns
        returns = prices.pct_change()

        # Apply signals to get strategy returns
        strategy_returns = self._calculate_strategy_returns(returns, signals)

        # Apply transaction costs
        strategy_returns = self._apply_transaction_costs(strategy_returns, signals)

        # Calculate equity curve
        self.equity_curve = (1 + strategy_returns).cumprod() * self.config.initial_capital

        # Extract trades
        self.trades = self._extract_trades(prices, signals)

        # Calculate metrics
        metrics = self._calculate_metrics(strategy_returns)

        self.results = {
            "metrics": metrics,
            "equity_curve": self.equity_curve,
            "trades": self.trades,
            "final_equity": float(self.equity_curve.iloc[-1]),
        }

    def _calculate_strategy_returns(
        self, returns: pd.DataFrame, signals: pd.DataFrame
    ) -> pd.Series:
        """
        Calculate strategy returns from market retrns and signals

        For multiple assets, applies position sizing
        """
        if returns.shape[1] == 1:
            # Single asset
            strategy_returns = returns.iloc[:, 0] * signals.iloc[:, 0].shift(1)
        else:
            # Multiple assets - equal weight or custom weighting
            weights = self._calculate_weights(signals)
            strategy_returns = (returns * weights.shift(1)).sum(axis=1)

        return strategy_returns.fillna(0)

    def _calculate_weights(self, signals: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate position weights from signals

        Applies positions sizing and max positions constraint
        """
        # Count number of positions each period
        num_positions = (signals != 0).sum(axis=1)

        # Cap at max positions
        num_positions = num_positions.clip(upper=self.config.max_positions)

        # Calculate weight per position
        weight_per_position = self.config.position_size / num_positions
        weight_per_position = weight_per_position.replace([np.inf, -np.inf], 0)

        # Apply to signals
        weights = signals.multiply(weight_per_position, axis=0)

        # Normalize to ensure total weight doesn't exceed position_size
        total_weight = weights.abs().sum(axis=1)
        mask = total_weight > self.config.position_size
        weights[mask] = weights[mask].div(total_weight[mask], axis=0) * self.config.position_size

        return weights

    def _apply_transaction_costs(self, returns: pd.Series, signals: pd.DataFrame) -> pd.Series:
        """
        Apply commission and slippage to returns

        Costs are incurred when positions change
        """
        # Detect position changes
        position_changes = signals.diff().abs()

        if isinstance(position_changes, pd.DataFrame):
            position_changes = position_changes.sum(axis=1)

        # Calculate total cost rate (commission + slippage on both sides)
        total_cost_rate = self.config.commission + self.config.slippage

        # Apply costs
        costs = position_changes * total_cost_rate
        adjusted_returns = returns - costs

        return adjusted_returns

    def _extract_trades(self, prices: pd.DataFrame, signals: pd.DataFrame) -> pd.DataFrame:
        """
        Extract individual trades from signals

        Returns:
            DataFrame with trade details
        """
        trades_list = []

        # Process each column (ticker)
        for col in signals.columns:
            col_signals = signals[col]
            col_prices = prices[col]

            # Find position changes
            position_changes = col_signals.diff()

            # Track open position
            entry_time = None
            entry_price = None
            entry_signal = None

            for i, (time, signal) in enumerate(col_signals.items()):
                if i == 0:
                    continue

                # Entry
                if entry_price is None and signal != 0:
                    entry_time = time
                    entry_price = col_prices.loc[time]
                    entry_signal = signal

                # Exit
                elif entry_time is not None and signal == 0:
                    exit_time = time
                    exit_price = col_prices.loc[time]

                    # Calculate P&L
                    if entry_signal == 1:  # Long
                        pnl = (exit_price - entry_price) / entry_price
                    else:  # Short
                        pnl = (entry_price - exit_price) / entry_price

                    # Apply costs
                    pnl -= (self.config.commission + self.config.slippage) * 2

                    trades_list.append(
                        {
                            "ticker": col,
                            "entry_time": entry_price,
                            "exit_time": exit_time,
                            "entry_price": entry_price,
                            "exit_price": exit_price,
                            "direction": "long" if entry_signal == 1 else "short",
                            "pnl_pct": pnl,
                            "pnl": pnl * self.config.initial_capital * self.config.position_size,
                        }
                    )

                    # Reset
                    entry_time = None
                    entry_price = None
                    entry_signal = None

        return pd.DataFrame(trades_list)

    def _calculate_metrics(self, returns: pd.Series) -> dict[str, float]:
        """Calculate performance metrics"""
        # Basic returns
        total_return = (1 + returns).prod() - 1

        # Annualized metrics
        num_days = len(returns)
        years = num_days / 252
        annualized_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0

        # Volatility
        volatility = returns.std() * np.sqrt(252)

        # Sharpe ratio
        risk_free_rate = 0.05
        sharpe_ratio = (annualized_return - risk_free_rate) / volatility if volatility > 0 else 0

        # Sortino ratio (downside deviation)
        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std() * np.sqrt(252)
        sortino_ratio = (
            (annualized_return - risk_free_rate) / downside_std if downside_std > 0 else 0
        )

        # Maximum drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()

        # Calmar ratio
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0

        # Win rate and trade stats
        if self.trades is not None and len(self.trades) > 0:
            winning_trades = self.trades[self.trades["pnl"] > 0]
            losing_trades = self.trades[self.trades["pnl"] < 0]

            win_rate = len(winning_trades) / len(self.trades)
            avg_win = winning_trades["pnl"].mean() if len(winning_trades) > 0 else 0
            avg_loss = losing_trades["pnl"].mean() if len(losing_trades) > 0 else 0

            total_wins = winning_trades["pnl"].sum() if len(winning_trades) > 0 else 0
            total_losses = abs(losing_trades["pnl"].sum()) if len(losing_trades) > 0 else 0
            profit_factor = total_wins / total_losses if total_losses > 0 else 0
        else:
            win_rate = 0
            avg_win = 0
            avg_loss = 0
            profit_factor = 0

        return {
            "total_return": float(total_return),
            "annualized_return": float(annualized_return),
            "volatility": float(volatility),
            "sharpe_ratio": float(sharpe_ratio),
            "sortino_ratio": float(sortino_ratio),
            "max_drawdown": float(max_drawdown),
            "calmar_ratio": float(calmar_ratio),
            "win_rate": float(win_rate),
            "avg_win": float(avg_win),
            "avg_loss": float(avg_loss),
            "profit_factor": float(profit_factor),
            "num_trades": len(self.trades) if self.trades is not None else 0,
        }

    def optimize_parameters(
        self,
        prices: pd.DataFrame,
        signal_generator: Callable,
        param_grid: dict[str, list[Any]],
        metric: str = "sharpe_ratio",
    ) -> dict[str, Any]:
        """
        Optimize strategy parameters using grid search

        Args:
            prices: Price data
            signal_generator: Function that generates signals given parameters
            param_grid: Dictionary of parameter names and values to test
            metric: Metric to optimize

        Returns:
            Dictionary with best parameters and results
        """
        from itertools import product

        logger.info(f"Starting parameter optimization: {len(param_grid)} parameters")

        # Generate all parameter combinations
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        combinations = list(product(*param_values))

        logger.info(f"Testing {len(combinations)} parameter combinations")

        results = []
        best_metric = -np.inf
        best_params = None

        for i, combo in enumerate(combinations):
            params = dict(zip(param_names, combo))

            try:
                # Generate signals with these parameters
                signals = signal_generator(prices, **params)

                # Run backtest
                result = self.run(prices, signals)

                # Store result
                metric_value = result["metrics"][metric]
                results.append(
                    {"params": params, "metrics": result["metrics"], metric: metric_value}
                )

                # Track best
                if metric_value > best_metric:
                    best_metric = metric_value
                    best_params = params

                if (i + 1) % 10 == 0:
                    logger.debug(f"Tested {i + 1}/{len(combinations)} combinations")

            except Exception as e:
                logger.warning(f"Failed for params {params}: {e}")
                continue

        logger.success(f"Optimization complete: Best {metric} = {best_metric:.4f}")

        return {
            "best_params": best_params,
            "best_metric_value": best_metric,
            "all_results": results,
        }


class MultiAssetVectorizedBacktester(VectorizedBacktest):
    """
    Extended vectorized backtester for multi-asset portfolios
    Handles portfolio rebalancing and capital allocation
    """

    def __init__(self, config: VectorizedBacktestConfig | None = None):
        super().__init__(config)

    def run_with_weights(
        self,
        prices: pd.DataFrame,
        weights: pd.DataFrame,
        rebalance_dates: pd.DatetimeIndex | None = None,
    ) -> dict[str, Any]:
        """
        Run backtest with explicit portfolio weights

        Args:
            prices: Asset prices
            weights: Portfolio weights over time
            rebalance_dates: Dates to rebalance (if None, continuous rebalancing)

        Returns:
            Backtest results
        """
        # Calculate returns
        returns = prices.pct_change()

        # Handle rebalancing
        if rebalance_dates is not None:
            weights = self._apply_rebalancing(weights, rebalance_dates)

        # Calculate portfolio returns
        portfolio_returns = (returns * weights.shift(1)).sum(axis=1)

        # Apply transaction costs at rebalancing
        if rebalance_dates is not None:
            weight_changes = weights.diff().abs().sum(axis=1)
            costs = weight_changes * (self.config.commission + self.config.slippage)
            portfolio_returns -= costs

        # Calculate equity curve
        self.equity_curve = (1 + portfolio_returns).cumprod() * self.config.initial_capital

        # Calculate metrics
        metrics = self._calculate_metrics(portfolio_returns)

        return {
            "metrics": metrics,
            "equity_curve": self.equity_curve,
            "weights": weights,
            "final_equity": float(self.equity_curve.iloc[-1]),
        }

    def _apply_rebalancing(
        self, weights: pd.DataFrame, rebalance_dates: pd.DatetimeIndex
    ) -> pd.DataFrame:
        """
        Apply periodic rebalancing to weights

        Between rebalancing dates, weights drift with returns
        """
        # Forward fill weights to rebalancing dates only
        rebalance_weights = weights.reindex(weights.index.union(rebalance_dates)).fillna(
            method="ffill"
        )

        # Keep only original index
        rebalance_weights = rebalance_weights.reindex(weights.index)

        return rebalance_weights


def simple_ma_crossover_strategy(
    prices: pd.Series,
    fast_period: int = 50,
    slow_period: int = 200,
) -> pd.Series:
    """
    Example: Simple moving average crossover strategy

    Args:
        prices: Price series
        fast_period: Fast MA period
        slow_period: Slow MA period

    Returns:
        Series of signals (1, 0, -1)
    """
    fast_ma = prices.rolling(fast_period).mean()
    slow_ma = prices.rolling(slow_period).mean()

    signals = pd.Series(0, index=prices.index)
    signals[fast_ma > slow_ma] = 1  # Long
    signals[fast_ma < slow_ma] = -1  # Short

    return signals


def momentum_strategy(prices: pd.Series, lookback: int = 20, holding_period: int = 5) -> pd.Series:
    """
    Example: Momentum strategy

    Long when price if above X-day high, short when below X-day low
    """
    returns = prices.pct_change(lookback)

    signals = pd.Series(0, index=prices.index)
    signals[returns > 0.02] = 1  # Long on positive momentum
    signals[returns < -0.02] = -1  # Short on negative momentum

    return signals
