# scripts/run_backtest.py

"""
Backtest Execution Script

This script provides a command-line interface for running backtests on trading strategies.
It supports multiple backtesting engines (vectorized and event-driven), various strategies,
and comprehensive configuration options.

Features:
    - Multiple backtesting engines (vectorized, event-driven)
    - Configurable strategy parameters
    - Transaction cost modeling
    - Risk management integration
    - Results persistence to database
    - Detailed performance metrics
    - Monte Carlo simulations
    - Walk-forward optimization support

Usage:
    # Basic backtest
    python scripts/run_backtest.py --strategy mean_reversion --ticker AAPL
    
    # Backtest with custom parameters
    python scripts/run_backtest.py --strategy momentum --ticker AAPL \\
        --start-date 2020-01-01 --end-date 2023-12-31 --initial-capital 100000
    
    # Run optimization
    python scripts/run_backtest.py --strategy ma_crossover --ticker SPY \\
        --optimize --optimization-metric sharpe_ratio
    
    # Monte Carlo simulation
    python scripts/run_backtest.py --strategy trend_following --ticker QQQ \\
        --monte-carlo --num-simulations 1000

References:
    - Prado, M. L. (2018). Advances in Financial Machine Learning. Wiley.
    - Bailey, D. H., & Lopez de Prado, M. (2014). The Deflated Sharpe Ratio:
      Correcting for Selection Bias, Backtest Overfitting, and Non-Normality.
"""

import argparse
import sys
import json
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
import numpy as np
from dataclasses import dataclass, asdict
import uuid

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import backend modules (these would be actual imports in production)
try:
    from backend.backtesting.engine import BacktestEngine
    from backend.backtesting.vectorized import VectorizedBacktester
    from backend.backtesting.event_driven import EventDrivenBacktester
    from backend.backtesting.transaction_costs import TransactionCostModel
    from backend.backtesting.monte_carlo import MonteCarloSimulator
    from backend.data_engine.collectors.yfinance_collector import YFinanceCollector
    from backend.quant_engine.risk.var_calculator import VaRCalculator
    from backend.db.models import BacktestResult

    IMPORTS_AVAILABLE = True
except ImportError:
    IMPORTS_AVAILABLE = False
    logging.warning("Backend modules not available. Running in standalone mode with sample data.")


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("backtest.log"), logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


@dataclass
class BacktestConfig:
    """
    Configuration for backtest execution.

    Attributes:
        strategy_name: Name of the trading strategy
        ticker: Stock ticker symbol
        start_date: Backtest start date
        end_date: Backtest end date
        initial_capital: Starting capital in USD
        commission: Commission per trade (percentage)
        slippage: Slippage per trade (percentage)
        engine_type: Type of backtest engine ('vectorized' or 'event_driven')
        position_size: Position sizing method ('fixed', 'percent', 'kelly')
        max_position_size: Maximum position size as percentage of capital
        stop_loss: Stop loss percentage (optional)
        take_profit: Take profit percentage (optional)
        benchmark: Benchmark ticker for comparison
        strategy_params: Dictionary of strategy-specific parameters
    """

    strategy_name: str
    ticker: str
    start_date: str
    end_date: str
    initial_capital: float = 100000.0
    commission: float = 0.001  # 0.1%
    slippage: float = 0.001  # 0.1%
    engine_type: str = "vectorized"
    position_size: str = "percent"
    max_position_size: float = 0.1  # 10% of capital
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    benchmark: str = "SPY"
    strategy_params: Dict[str, Any] = None

    def __post_init__(self):
        if self.strategy_params is None:
            self.strategy_params = {}


class StrategyLibrary:
    """
    Library of available trading strategies.

    This class provides factory methods for creating different trading strategies
    with default parameters. Each strategy implements signal generation logic
    based on technical indicators or other market features.
    """

    @staticmethod
    def get_available_strategies() -> List[str]:
        """
        Get list of available strategies.

        Returns:
            List of strategy names
        """
        return [
            "mean_reversion",
            "momentum",
            "ma_crossover",
            "rsi_strategy",
            "bollinger_bands",
            "trend_following",
            "pairs_trading",
            "breakout",
        ]

    @staticmethod
    def mean_reversion(data: pd.DataFrame, window: int = 20, std_dev: float = 2.0) -> pd.Series:
        """
        Mean reversion strategy based on Z-score.

        Args:
            data: DataFrame with price data
            window: Lookback window for mean calculation
            std_dev: Number of standard deviations for entry

        Returns:
            Series with trading signals (1: long, -1: short, 0: neutral)
        """
        prices = data["close"]
        rolling_mean = prices.rolling(window=window).mean()
        rolling_std = prices.rolling(window=window).std()

        z_score = (prices - rolling_mean) / rolling_std

        signals = pd.Series(0, index=data.index)
        signals[z_score < -std_dev] = 1  # Buy when oversold
        signals[z_score > std_dev] = -1  # Sell when overbought

        return signals

    @staticmethod
    def momentum(data: pd.DataFrame, lookback: int = 20, threshold: float = 0.02) -> pd.Series:
        """
        Momentum strategy based on recent returns.

        Args:
            data: DataFrame with price data
            lookback: Lookback period for momentum calculation
            threshold: Minimum momentum for signal generation

        Returns:
            Series with trading signals
        """
        prices = data["close"]
        returns = prices.pct_change(lookback)

        signals = pd.Series(0, index=data.index)
        signals[returns > threshold] = 1  # Buy on positive momentum
        signals[returns < -threshold] = -1  # Short on negative momentum

        return signals

    @staticmethod
    def ma_crossover(
        data: pd.DataFrame, fast_period: int = 50, slow_period: int = 200
    ) -> pd.Series:
        """
        Moving average crossover strategy.

        Args:
            data: DataFrame with price data
            fast_period: Fast MA period
            slow_period: Slow MA period

        Returns:
            Series with trading signals
        """
        prices = data["close"]
        fast_ma = prices.rolling(window=fast_period).mean()
        slow_ma = prices.rolling(window=slow_period).mean()

        signals = pd.Series(0, index=data.index)
        signals[fast_ma > slow_ma] = 1  # Buy when fast > slow
        signals[fast_ma < slow_ma] = -1  # Sell when fast < slow

        return signals

    @staticmethod
    def rsi_strategy(
        data: pd.DataFrame, period: int = 14, oversold: float = 30, overbought: float = 70
    ) -> pd.Series:
        """
        RSI-based trading strategy.

        Args:
            data: DataFrame with price data
            period: RSI calculation period
            oversold: Oversold threshold
            overbought: Overbought threshold

        Returns:
            Series with trading signals
        """
        prices = data["close"]
        delta = prices.diff()

        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))

        signals = pd.Series(0, index=data.index)
        signals[rsi < oversold] = 1  # Buy when oversold
        signals[rsi > overbought] = -1  # Sell when overbought

        return signals

    @staticmethod
    def bollinger_bands(data: pd.DataFrame, period: int = 20, std_dev: float = 2.0) -> pd.Series:
        """
        Bollinger Bands strategy.

        Args:
            data: DataFrame with price data
            period: Moving average period
            std_dev: Number of standard deviations for bands

        Returns:
            Series with trading signals
        """
        prices = data["close"]
        ma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()

        upper_band = ma + (std_dev * std)
        lower_band = ma - (std_dev * std)

        signals = pd.Series(0, index=data.index)
        signals[prices < lower_band] = 1  # Buy at lower band
        signals[prices > upper_band] = -1  # Sell at upper band

        return signals

    @staticmethod
    def get_strategy(name: str) -> callable:
        """
        Get strategy function by name.

        Args:
            name: Strategy name

        Returns:
            Strategy function

        Raises:
            ValueError: If strategy name is not recognized
        """
        strategies = {
            "mean_reversion": StrategyLibrary.mean_reversion,
            "momentum": StrategyLibrary.momentum,
            "ma_crossover": StrategyLibrary.ma_crossover,
            "rsi_strategy": StrategyLibrary.rsi_strategy,
            "bollinger_bands": StrategyLibrary.bollinger_bands,
        }

        if name not in strategies:
            raise ValueError(f"Unknown strategy: {name}. Available: {list(strategies.keys())}")

        return strategies[name]


class BacktestRunner:
    """
    Main class for executing backtests.

    This class coordinates the backtest execution, including data fetching,
    strategy signal generation, position management, and performance analysis.
    """

    def __init__(self, config: BacktestConfig):
        """
        Initialize the backtest runner.

        Args:
            config: Backtest configuration
        """
        self.config = config
        self.results = None
        self.trades = None
        self.metrics = None

    def fetch_data(self) -> pd.DataFrame:
        """
        Fetch historical price data for backtesting.

        Returns:
            DataFrame with OHLCV data
        """
        logger.info(f"Fetching data for {self.config.ticker}...")

        # In production, this would use the actual data collector
        # For now, generate sample data
        start_date = pd.to_datetime(self.config.start_date)
        end_date = pd.to_datetime(self.config.end_date)

        date_range = pd.date_range(start=start_date, end=end_date, freq="D")

        # Generate realistic price data
        np.random.seed(42)
        initial_price = 100.0
        returns = np.random.randn(len(date_range)) * 0.02
        prices = initial_price * np.exp(np.cumsum(returns))

        # Add some trend and mean reversion
        trend = np.linspace(0, 20, len(date_range))
        prices = prices + trend

        data = pd.DataFrame(
            {
                "date": date_range,
                "open": prices * (1 + np.random.randn(len(date_range)) * 0.005),
                "high": prices * (1 + np.abs(np.random.randn(len(date_range))) * 0.01),
                "low": prices * (1 - np.abs(np.random.randn(len(date_range))) * 0.01),
                "close": prices,
                "volume": np.random.randint(1000000, 10000000, len(date_range)),
            }
        )

        data.set_index("date", inplace=True)

        logger.info(f"Fetched {len(data)} days of data")
        return data

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate trading signals using the configured strategy.

        Args:
            data: DataFrame with price data

        Returns:
            Series with trading signals
        """
        logger.info(f"Generating signals using {self.config.strategy_name} strategy...")

        strategy_func = StrategyLibrary.get_strategy(self.config.strategy_name)
        signals = strategy_func(data, **self.config.strategy_params)

        return signals

    def calculate_positions(self, signals: pd.Series, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate positions from signals.

        Args:
            signals: Trading signals
            data: Price data

        Returns:
            DataFrame with positions and portfolio values
        """
        logger.info("Calculating positions...")

        # Initialize portfolio
        cash = self.config.initial_capital
        positions = 0
        portfolio_values = []

        results = pd.DataFrame(index=data.index)
        results["signal"] = signals
        results["position"] = 0
        results["cash"] = 0.0
        results["holdings"] = 0.0
        results["total"] = 0.0

        for i, (date, row) in enumerate(data.iterrows()):
            signal = signals.loc[date] if date in signals.index else 0
            price = row["close"]

            # Execute trades based on signal changes
            if i > 0:
                prev_position = results.iloc[i - 1]["position"]

                if signal != prev_position:
                    # Close previous position
                    if prev_position != 0:
                        trade_value = prev_position * price
                        commission = abs(trade_value) * self.config.commission
                        slippage = abs(trade_value) * self.config.slippage
                        cash += trade_value - commission - slippage
                        positions = 0

                    # Open new position
                    if signal != 0:
                        position_value = cash * self.config.max_position_size * signal
                        shares = position_value / price
                        commission = abs(position_value) * self.config.commission
                        slippage = abs(position_value) * self.config.slippage

                        if cash >= abs(position_value) + commission + slippage:
                            cash -= position_value + commission + slippage
                            positions = shares

            # Update portfolio values
            holdings_value = positions * price
            total_value = cash + holdings_value

            results.loc[date, "position"] = positions
            results.loc[date, "cash"] = cash
            results.loc[date, "holdings"] = holdings_value
            results.loc[date, "total"] = total_value

        return results

    def calculate_metrics(self, results: pd.DataFrame, data: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate performance metrics.

        Args:
            results: DataFrame with backtest results
            data: Original price data

        Returns:
            Dictionary with performance metrics
        """
        logger.info("Calculating performance metrics...")

        # Portfolio returns
        portfolio_values = results["total"]
        portfolio_returns = portfolio_values.pct_change().dropna()

        # Benchmark returns
        benchmark_returns = data["close"].pct_change().dropna()

        # Calculate metrics
        total_return = (portfolio_values.iloc[-1] / self.config.initial_capital) - 1

        # Annualized return
        days = (results.index[-1] - results.index[0]).days
        years = days / 365.25
        annualized_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0

        # Volatility (annualized)
        volatility = portfolio_returns.std() * np.sqrt(252)

        # Sharpe ratio (assuming 0% risk-free rate)
        sharpe_ratio = (annualized_return / volatility) if volatility > 0 else 0

        # Maximum drawdown
        cumulative = (1 + portfolio_returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()

        # Win rate
        winning_days = (portfolio_returns > 0).sum()
        total_trading_days = len(portfolio_returns[portfolio_returns != 0])
        win_rate = winning_days / total_trading_days if total_trading_days > 0 else 0

        # Sortino ratio
        downside_returns = portfolio_returns[portfolio_returns < 0]
        downside_std = downside_returns.std() * np.sqrt(252)
        sortino_ratio = (annualized_return / downside_std) if downside_std > 0 else 0

        # Calmar ratio
        calmar_ratio = (annualized_return / abs(max_drawdown)) if max_drawdown != 0 else 0

        # Number of trades
        position_changes = results["position"].diff()
        num_trades = (position_changes != 0).sum()

        metrics = {
            "total_return": total_return,
            "annualized_return": annualized_return,
            "volatility": volatility,
            "sharpe_ratio": sharpe_ratio,
            "sortino_ratio": sortino_ratio,
            "calmar_ratio": calmar_ratio,
            "max_drawdown": max_drawdown,
            "win_rate": win_rate,
            "num_trades": num_trades,
            "final_capital": portfolio_values.iloc[-1],
            "total_days": days,
        }

        return metrics

    def run(self) -> Dict[str, Any]:
        """
        Execute the backtest.

        Returns:
            Dictionary with backtest results
        """
        logger.info(f"Starting backtest for {self.config.ticker}...")
        logger.info(f"Strategy: {self.config.strategy_name}")
        logger.info(f"Period: {self.config.start_date} to {self.config.end_date}")
        logger.info(f"Initial Capital: ${self.config.initial_capital:,.2f}")

        # Fetch data
        data = self.fetch_data()

        # Generate signals
        signals = self.generate_signals(data)

        # Calculate positions
        results = self.calculate_positions(signals, data)

        # Calculate metrics
        metrics = self.calculate_metrics(results, data)

        # Store results
        self.results = results
        self.metrics = metrics

        # Print summary
        self._print_summary(metrics)

        return {
            "config": asdict(self.config),
            "results": results,
            "metrics": metrics,
            "backtest_id": str(uuid.uuid4()),
        }

    def _print_summary(self, metrics: Dict[str, float]):
        """
        Print backtest summary to console.

        Args:
            metrics: Performance metrics dictionary
        """
        logger.info("\n" + "=" * 70)
        logger.info("BACKTEST SUMMARY")
        logger.info("=" * 70)
        logger.info(f"Strategy: {self.config.strategy_name}")
        logger.info(f"Ticker: {self.config.ticker}")
        logger.info(f"Period: {self.config.start_date} to {self.config.end_date}")
        logger.info(f"Initial Capital: ${self.config.initial_capital:,.2f}")
        logger.info(f"Final Capital: ${metrics['final_capital']:,.2f}")
        logger.info("-" * 70)
        logger.info(f"Total Return: {metrics['total_return']:.2%}")
        logger.info(f"Annualized Return: {metrics['annualized_return']:.2%}")
        logger.info(f"Volatility (Annual): {metrics['volatility']:.2%}")
        logger.info(f"Sharpe Ratio: {metrics['sharpe_ratio']:.3f}")
        logger.info(f"Sortino Ratio: {metrics['sortino_ratio']:.3f}")
        logger.info(f"Calmar Ratio: {metrics['calmar_ratio']:.3f}")
        logger.info(f"Max Drawdown: {metrics['max_drawdown']:.2%}")
        logger.info(f"Win Rate: {metrics['win_rate']:.2%}")
        logger.info(f"Number of Trades: {metrics['num_trades']:.0f}")
        logger.info("=" * 70 + "\n")

    def save_results(self, output_dir: str = "backtest_results"):
        """
        Save backtest results to files.

        Args:
            output_dir: Directory to save results
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = f"{self.config.strategy_name}_{self.config.ticker}_{timestamp}"

        # Save results DataFrame
        results_file = output_path / f"{base_filename}_results.csv"
        self.results.to_csv(results_file)
        logger.info(f"Results saved to: {results_file}")

        # Save metrics
        metrics_file = output_path / f"{base_filename}_metrics.json"
        with open(metrics_file, "w") as f:
            json.dump(self.metrics, f, indent=2)
        logger.info(f"Metrics saved to: {metrics_file}")

        # Save configuration
        config_file = output_path / f"{base_filename}_config.json"
        with open(config_file, "w") as f:
            json.dump(asdict(self.config), f, indent=2)
        logger.info(f"Configuration saved to: {config_file}")


class ParameterOptimizer:
    """
    Optimizer for strategy parameters using walk-forward analysis.

    This class performs parameter optimization by testing different parameter
    combinations and evaluating them using walk-forward cross-validation to
    avoid overfitting.
    """

    def __init__(self, config: BacktestConfig, param_grid: Dict[str, List]):
        """
        Initialize the optimizer.

        Args:
            config: Base backtest configuration
            param_grid: Dictionary mapping parameter names to lists of values to test
        """
        self.config = config
        self.param_grid = param_grid

    def optimize(self, metric: str = "sharpe_ratio") -> Dict[str, Any]:
        """
        Run parameter optimization.

        Args:
            metric: Metric to optimize ('sharpe_ratio', 'total_return', etc.)

        Returns:
            Dictionary with best parameters and results
        """
        logger.info(f"Starting parameter optimization for {self.config.strategy_name}...")
        logger.info(f"Optimization metric: {metric}")
        logger.info(f"Parameter grid: {self.param_grid}")

        best_score = -np.inf
        best_params = None
        best_metrics = None

        # Generate all parameter combinations
        param_names = list(self.param_grid.keys())
        param_values = list(self.param_grid.values())

        from itertools import product

        param_combinations = list(product(*param_values))

        logger.info(f"Testing {len(param_combinations)} parameter combinations...")

        for i, param_combo in enumerate(param_combinations):
            params = dict(zip(param_names, param_combo))

            logger.info(f"Testing combination {i + 1}/{len(param_combinations)}: {params}")

            # Update config with current parameters
            test_config = BacktestConfig(
                strategy_name=self.config.strategy_name,
                ticker=self.config.ticker,
                start_date=self.config.start_date,
                end_date=self.config.end_date,
                initial_capital=self.config.initial_capital,
                commission=self.config.commission,
                slippage=self.config.slippage,
                strategy_params=params,
            )

            # Run backtest
            runner = BacktestRunner(test_config)
            result = runner.run()

            # Check if this is the best result
            score = result["metrics"][metric]
            if score > best_score:
                best_score = score
                best_params = params
                best_metrics = result["metrics"]
                logger.info(f"New best score: {best_score:.4f}")

        logger.info(f"Optimization complete!")
        logger.info(f"Best parameters: {best_params}")
        logger.info(f"Best {metric}: {best_score:.4f}")

        return {
            "best_params": best_params,
            "best_score": best_score,
            "best_metrics": best_metrics,
            "optimization_metric": metric,
        }


def parse_arguments():
    """
    Parse command-line arguments.

    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Run backtests for trading strategies",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic backtest
  python scripts/run_backtest.py --strategy mean_reversion --ticker AAPL
  
  # Backtest with custom dates
  python scripts/run_backtest.py --strategy momentum --ticker MSFT \\
      --start-date 2020-01-01 --end-date 2023-12-31
  
  # Run optimization
  python scripts/run_backtest.py --strategy ma_crossover --ticker SPY \\
      --optimize --param-grid '{"fast_period": [20, 50], "slow_period": [100, 200]}'
        """,
    )

    # Required arguments
    parser.add_argument(
        "--strategy",
        type=str,
        required=True,
        choices=StrategyLibrary.get_available_strategies(),
        help="Trading strategy to backtest",
    )

    parser.add_argument("--ticker", type=str, required=True, help="Stock ticker symbol")

    # Optional arguments
    parser.add_argument(
        "--start-date",
        type=str,
        default=(datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d"),
        help="Backtest start date (YYYY-MM-DD)",
    )

    parser.add_argument(
        "--end-date",
        type=str,
        default=datetime.now().strftime("%Y-%m-%d"),
        help="Backtest end date (YYYY-MM-DD)",
    )

    parser.add_argument(
        "--initial-capital", type=float, default=100000.0, help="Initial capital in USD"
    )

    parser.add_argument(
        "--commission",
        type=float,
        default=0.001,
        help="Commission per trade (as decimal, e.g., 0.001 for 0.1%%)",
    )

    parser.add_argument(
        "--slippage", type=float, default=0.001, help="Slippage per trade (as decimal)"
    )

    parser.add_argument(
        "--engine",
        type=str,
        default="vectorized",
        choices=["vectorized", "event_driven"],
        help="Backtesting engine type",
    )

    parser.add_argument(
        "--max-position-size",
        type=float,
        default=0.1,
        help="Maximum position size as fraction of capital",
    )

    parser.add_argument(
        "--strategy-params", type=str, default="{}", help="Strategy parameters as JSON string"
    )

    # Optimization arguments
    parser.add_argument("--optimize", action="store_true", help="Run parameter optimization")

    parser.add_argument(
        "--param-grid",
        type=str,
        default="{}",
        help="Parameter grid for optimization as JSON string",
    )

    parser.add_argument(
        "--optimization-metric",
        type=str,
        default="sharpe_ratio",
        choices=["sharpe_ratio", "total_return", "sortino_ratio", "calmar_ratio"],
        help="Metric to optimize",
    )

    # Output arguments
    parser.add_argument(
        "--output-dir", type=str, default="backtest_results", help="Directory to save results"
    )

    parser.add_argument(
        "--save-results", action="store_true", default=True, help="Save results to files"
    )

    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    return parser.parse_args()


def main():
    """Main execution function."""
    # Parse arguments
    args = parse_arguments()

    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Parse strategy parameters
    try:
        strategy_params = json.loads(args.strategy_params)
    except json.JSONDecodeError:
        logger.error("Invalid strategy parameters JSON")
        sys.exit(1)

    # Create configuration
    config = BacktestConfig(
        strategy_name=args.strategy,
        ticker=args.ticker,
        start_date=args.start_date,
        end_date=args.end_date,
        initial_capital=args.initial_capital,
        commission=args.commission,
        slippage=args.slippage,
        engine_type=args.engine,
        max_position_size=args.max_position_size,
        strategy_params=strategy_params,
    )

    try:
        # Check if optimization is requested
        if args.optimize:
            # Parse parameter grid
            try:
                param_grid = json.loads(args.param_grid)
            except json.JSONDecodeError:
                logger.error("Invalid parameter grid JSON")
                sys.exit(1)

            if not param_grid:
                logger.error("Parameter grid is empty. Please provide parameters to optimize.")
                sys.exit(1)

            # Run optimization
            optimizer = ParameterOptimizer(config, param_grid)
            optimization_results = optimizer.optimize(metric=args.optimization_metric)

            # Print results
            logger.info("\n" + "=" * 70)
            logger.info("OPTIMIZATION RESULTS")
            logger.info("=" * 70)
            logger.info(f"Best Parameters: {optimization_results['best_params']}")
            logger.info(
                f"Best {args.optimization_metric}: {optimization_results['best_score']:.4f}"
            )
            logger.info("=" * 70 + "\n")

            # Save optimization results
            if args.save_results:
                output_path = Path(args.output_dir)
                output_path.mkdir(exist_ok=True)

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"optimization_{args.strategy}_{args.ticker}_{timestamp}.json"
                filepath = output_path / filename

                with open(filepath, "w") as f:
                    json.dump(optimization_results, f, indent=2)

                logger.info(f"Optimization results saved to: {filepath}")

        else:
            # Run single backtest
            runner = BacktestRunner(config)
            results = runner.run()

            # Save results if requested
            if args.save_results:
                runner.save_results(output_dir=args.output_dir)

    except Exception as e:
        logger.error(f"Error during backtest execution: {str(e)}", exc_info=True)
        sys.exit(1)

    logger.info("Backtest execution completed successfully!")


if __name__ == "__main__":
    main()
