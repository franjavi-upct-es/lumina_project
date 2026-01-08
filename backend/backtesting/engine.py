# backend/backtesting/engine.py
"""
Main backtesting engine with support for multiple execution modes
Orchestrates vectorized, event-driven, and Monte Carlo backtesting
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from backend.backtesting.event_driven import EventDrivenBacktest
from backend.backtesting.monte_carlo import MonteCarloSimulation
from backend.backtesting.transaction_costs import TransactionCostModel
from backend.backtesting.vectorized import VectorizedBacktest


class BacktestMode(Enum):
    """Backtesting execution modes"""

    VECTORIZED = "vectorized"
    EVENT_DRIVEN = "event_driven"
    MONTE_CARLO = "monte_carlo"


@dataclass
class BacktestConfig:
    """
    Configuration for backtesting
    """

    # Capital
    initial_capital: float = 100000.0

    # Position sizing
    position_size: float = 0.1  # Fraction of capital per position
    max_positions: int = 10

    # Transaction costs
    commission: float = 0.001  # 0.1%
    slippage: float = 0.0005  # 0.05%

    # Risk management
    stop_loss: float | None = None  # Percentage
    take_profit: float | None = None  # Percentage
    max_leverage: float = 1.0

    # Execution
    execution_mode: BacktestMode = BacktestMode.VECTORIZED

    # Rebalancing
    rebalance_frequency: str = "daily"  # daily, weekly, monthly

    # Monte Carlo specific
    num_simulations: int = 1000
    confidence_level: float = 0.95

    # Additional settings
    allow_shorting: bool = False
    use_market_orders: bool = True
    benchmark: str = "SPY"


@dataclass
class Trade:
    """
    Represents a single trade
    """

    ticker: str
    entry_time: datetime
    exit_time: datetime | None
    direction: str  # 'long' or 'short'
    entry_price: float
    exit_price: float | None
    quantity: float
    commission: float
    slippage: float
    pnl: float | None = None
    pnl_percent: float | None = None
    exit_reason: str | None = None  # 'signal', 'stop_loss', 'take_profit'

    def close(self, exit_time: datetime, exit_price: float, reason: str = "signal"):
        """Close the trade"""
        self.exit_time = exit_time
        self.exit_price = exit_price
        self.exit_reason = reason

        # Calculate P&L
        if self.direction == "long":
            self.pnl = (exit_price - self.entry_price) * self.quantity
        else:  # short
            self.pnl = (self.entry_price - exit_price) * self.quantity

        # Subtract costs
        self.pnl -= self.commission + self.slippage

        # Calculate percentage
        if self.entry_price > 0:
            self.pnl_percent = (self.pnl / (self.entry_price * self.quantity)) * 100


@dataclass
class BacktestResult:
    """
    Complete backtest results
    """

    # Strategy info
    strategy_name: str
    tickers: list[str]
    start_date: datetime
    end_date: datetime

    # Capital
    initial_capital: float
    final_capital: float

    # Returns
    total_return: float
    annualized_return: float

    # Risk metrics
    volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown: float

    # Trade statistics
    trades: list[Trade] = field(default_factory=list)
    num_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = 0.0

    # Time series data
    equity_curve: pd.DataFrame = field(default_factory=pd.DataFrame)
    drawdown_series: pd.Series = field(default_factory=pd.Series)

    # Benchmark comparison
    benchmark_return: float | None = None
    alpha: float | None = None
    beta: float | None = None

    # Additional metrics
    recovery_factor: float | None = None
    payoff_ratio: float | None = None

    def calculate_metrics(self):
        """Calculate derived metrics from trades"""
        if not self.trades:
            return

        # Filter completed trades
        completed = [t for t in self.trades if t.exit_time is not None]

        if not completed:
            return

        self.num_trades = len(completed)

        # Winning/losing trades
        wins = [t for t in completed if t.pnl > 0]
        losses = [t for t in completed if t.pnl < 0]

        self.winning_trades = len(wins)
        self.losing_trades = len(losses)
        self.win_rate = self.winning_trades / self.num_trades if self.num_trades > 0 else 0

        # Average win/loss
        self.avg_win = np.mean([t.pnl for t in wins]) if wins else 0.0
        self.avg_loss = np.mean([abs(t.pnl) for t in losses]) if losses else 0.0

        # Profit factor
        total_wins = sum(t.pnl for t in wins)
        total_losses = abs(sum(t.pnl for t in losses))
        self.profit_factor = total_wins / total_losses if total_losses > 0 else 0.0

        # Payoff ratio
        self.payoff_ratio = abs(self.avg_win / self.avg_loss) if self.avg_loss != 0 else 0.0

        # Recovery factor
        if self.max_drawdown != 0:
            self.recovery_factor = self.total_return / abs(self.max_drawdown)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary"""
        return {
            "strategy_name": self.strategy_name,
            "tickers": self.tickers,
            "start_date": self.start_date.isoformat(),
            "end_date": self.end_date.isoformat(),
            "initial_capital": self.initial_capital,
            "final_capital": self.final_capital,
            "total_return": self.total_return,
            "annualized_return": self.annualized_return,
            "volatility": self.volatility,
            "sharpe_ratio": self.sharpe_ratio,
            "sortino_ratio": self.sortino_ratio,
            "calmar_ratio": self.calmar_ratio,
            "max_drawdown": self.max_drawdown,
            "num_trades": self.num_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "win_rate": self.win_rate,
            "avg_win": self.avg_win,
            "avg_loss": self.avg_loss,
            "profit_factor": self.profit_factor,
            "benchmark_return": self.benchmark_return,
            "alpha": self.alpha,
            "beta": self.beta,
        }


class BacktestEngine:
    """
    Main backtesting engine

    Supports:
    - Vectorized backtesting (fast, simple strategies)
    - Event-driven backtesting (complex logic, realistic execution)
    - Monte Carlo simulation (risk assessment)
    """

    def __init__(self, config: BacktestConfig | None = None):
        """
        Initialize backtest engine

        Args:
            config: Backtest configuration
        """
        self.config = config or BacktestConfig()

        # Transaction cost model
        self.cost_model = TransactionCostModel(
            commission=self.config.commission,
            slippage=self.config.slippage,
        )

        # Results
        self.results: BacktestResult | None = None

        logger.info(f"Initialized backtest engine in {self.config.execution_mode.value} mode")

    def run(
        self,
        strategy_func: Callable,
        data: dict[str, pd.DataFrame],
        strategy_name: str = "Strategy",
    ) -> BacktestResult:
        """
        Run backtest with specified strategy

        Args:
            strategy_func: Strategy function that generates signals
            data: Dictionary mapping tickers to OHLCV DataFrames
            strategy_name: Name of the strategy

        Returns:
            BacktestResult object
        """
        logger.info(f"Running backtest: {strategy_name}")
        logger.info(f"Mode: {self.config.execution_mode.value}")
        logger.info(f"Tickers: {list(data.keys())}")

        # Select execution mode
        if self.config.execution_mode == BacktestMode.VECTORIZED:
            backtest = VectorizedBacktest(
                initial_capital=self.config.initial_capital,
                commission=self.config.commission,
                slippage=self.config.slippage,
            )
            results = backtest.run(strategy_func, data, strategy_name)

        elif self.config.execution_mode == BacktestMode.EVENT_DRIVEN:
            backtest = EventDrivenBacktest(
                initial_capital=self.config.initial_capital,
                commission=self.config.commission,
                slippage=self.config.slippage,
                position_size=self.config.position_size,
                max_positions=self.config.max_positions,
                stop_loss=self.config.stop_loss,
                take_profit=self.config.take_profit,
            )
            results = backtest.run(strategy_func, data, strategy_name)

        elif self.config.execution_mode == BacktestMode.MONTE_CARLO:
            mc = MonteCarloSimulation(
                num_simulations=self.config.num_simulations,
                confidence_level=self.config.confidence_level,
            )
            results = mc.run(strategy_func, data, self.config)

        else:
            raise ValueError(f"Unknown execution mode: {self.config.execution_mode}")

        self.results = results

        # Log summary
        self._log_summary(results)

        return results

    def _log_summary(self, results: BacktestResult):
        """Log backtest summary"""
        logger.info("=" * 60)
        logger.info("BACKTEST RESULTS")
        logger.info("=" * 60)
        logger.info(f"Strategy: {results.strategy_name}")
        logger.info(f"Period: {results.start_date.date()} to {results.end_date.date()}")
        logger.info(f"Initial Capital: ${results.initial_capital:,.2f}")
        logger.info(f"Final Capital: ${results.final_capital:,.2f}")
        logger.info(f"Total Return: {results.total_return * 100:.2f}%")
        logger.info(f"Annualized Return: {results.annualized_return * 100:.2f}%")
        logger.info(f"Volatility: {results.volatility * 100:.2f}%")
        logger.info(f"Sharpe Ratio: {results.sharpe_ratio:.2f}")
        logger.info(f"Max Drawdown: {results.max_drawdown * 100:.2f}%")
        logger.info(f"Total Trades: {results.num_trades}")
        logger.info(f"Win Rate: {results.win_rate * 100:.2f}%")
        logger.info(f"Profit Factor: {results.profit_factor:.2f}")
        logger.info("=" * 60)

    def compare_strategies(
        self,
        strategies: list[tuple[Callable, str]],
        data: dict[str, pd.DataFrame],
    ) -> pd.DataFrame:
        """
        Compare multiple strategies

        Args:
            strategies: List of (strategy_func, strategy_name) tuples
            data: Market data

        Returns:
            DataFrame with comparison metrics
        """
        logger.info(f"Comparing {len(strategies)} strategies")

        results = []

        for strategy_func, strategy_name in strategies:
            try:
                result = self.run(strategy_func, data, strategy_name)
                results.append(result.to_dict())
            except Exception as e:
                logger.error(f"Error running {strategy_name}: {e}")
                continue

        if not results:
            logger.error("No strategies completed successfully")
            return pd.DataFrame()

        # Convert to DataFrame
        comparison = pd.DataFrame(results)

        # Sort by Sharpe ratio
        comparison = comparison.sort_values("sharpe_ratio", ascending=False)

        logger.success("Strategy comparison complete")
        return comparison

    def optimize_parameters(
        self,
        strategy_func: Callable,
        data: dict[str, pd.DataFrame],
        param_grid: dict[str, list[Any]],
        optimization_metric: str = "sharpe_ratio",
    ) -> dict[str, Any]:
        """
        Optimize strategy parameters using grid search

        Args:
            strategy_func: Strategy function
            data: Market data
            param_grid: Dictionary of parameter ranges
            optimization_metric: Metric to optimize

        Returns:
            Best parameters and results
        """
        logger.info("Starting parameter optimization")
        logger.info(f"Optimization metric: {optimization_metric}")

        from itertools import product

        # Generate all parameter combinations
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        combinations = list(product(*param_values))

        logger.info(f"Testing {len(combinations)} parameter combinations")

        best_result = None
        best_params = None
        best_metric_value = float("-inf")

        for i, combo in enumerate(combinations):
            params = dict(zip(param_names, combo, strict=True))

            if (i + 1) % 10 == 0:
                logger.info(f"Progress: {i + 1}/{len(combinations)}")

            try:
                # Create strategy with parameters
                def parameterized_strategy(data_dict, params=params):
                    return strategy_func(data_dict, **params)

                # Run backtest
                result = self.run(
                    parameterized_strategy,
                    data,
                    f"Optimization_{i}",
                )

                # Check if better
                metric_value = getattr(result, optimization_metric)

                if metric_value > best_metric_value:
                    best_metric_value = metric_value
                    best_params = params
                    best_result = result

            except Exception as e:
                logger.error(f"Error with params {params}: {e}")
                continue

        if best_result is None:
            logger.error("Optimization failed - no valid results")
            return {}

        logger.success("Optimization complete!")
        logger.info(f"Best {optimization_metric}: {best_metric_value:.4f}")
        logger.info(f"Best parameters: {best_params}")

        return {
            "best_params": best_params,
            "best_metric_value": best_metric_value,
            "best_result": best_result,
        }

    def walk_forward_analysis(
        self,
        strategy_func: Callable,
        data: dict[str, pd.DataFrame],
        train_size: int = 252,  # 1 year
        test_size: int = 63,  # 3 months
        step_size: int = 21,  # 1 month
    ) -> list[BacktestResult]:
        """
        Perform walk-forward analysis

        Args:
            strategy_func: Strategy function
            data: Market data
            train_size: Training window size (days)
            test_size: Testing window size (days)
            step_size: Step size for rolling window

        Returns:
            List of test results
        """
        logger.info("Starting walk-forward analysis")
        logger.info(f"Train: {train_size} days, Test: {test_size} days, Step: {step_size} days")

        # Get date range from first ticker
        first_ticker = list(data.keys())[0]
        dates = data[first_ticker].index

        results = []
        window_num = 0

        start_idx = 0
        while start_idx + train_size + test_size < len(dates):
            window_num += 1

            # Define windows
            train_start = start_idx
            train_end = start_idx + train_size
            test_start = train_end
            test_end = test_start + test_size

            logger.info(
                f"Window {window_num}: Train [{dates[train_start].date()} to {dates[train_end - 1].date()}], "
                f"Test [{dates[test_start].date()} to {dates[test_end - 1].date()}]"
            )

            # Split data
            test_data = {ticker: df.iloc[test_start:test_end] for ticker, df in data.items()}

            # Run backtest on test data
            try:
                result = self.run(
                    strategy_func,
                    test_data,
                    f"WalkForward_W{window_num}",
                )
                results.append(result)
            except Exception as e:
                logger.error(f"Window {window_num} failed: {e}")

            # Move window
            start_idx += step_size

        logger.success(f"Walk-forward analysis complete: {len(results)} windows tested")

        # Summary statistics
        if results:
            avg_return = np.mean([r.total_return for r in results])
            avg_sharpe = np.mean([r.sharpe_ratio for r in results])
            consistency = np.std([r.total_return for r in results])

            logger.info(f"Average Return: {avg_return * 100:.2f}%")
            logger.info(f"Average Sharpe: {avg_sharpe:.2f}")
            logger.info(f"Consistency (std): {consistency:.4f}")

        return results
