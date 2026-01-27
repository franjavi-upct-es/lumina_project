# backend/backtesting/__init__.py
"""
Backtesting Engine Module for Lumina Quant Lab

Provides comprehensive backtesting capabilities:

VectorizedBacktest:
- Fast vectorized backtesting
- Portfolio-level operations
- Efficient for simple strategies

EventDrivenBacktest:
- Event-by-event simulation
- Realistic order execution
- Support for complex strategies
- Market/Limit/Stop orders

TransactionCosts:
- Commission modeling
- Slippage estimation
- Market impact models
- Bid-ask spread simulation

MonteCarloSimulator:
- Geometric Brownian Motion
- Historical bootstrap
- Confidence intervals
- Path-dependent analysis

Strategies:
- Pre-built strategy templates
- RSI, MACD, Moving Average crossover
- Momentum, Mean Reversion
- Custom strategy framework

Usage:
    from backend.backtesting import EventDrivenBacktest, MonteCarloSimulator

    # Event-driven backtest
    backtest = EventDrivenBacktest(
        initial_capital=100000,
        commission=0.001,
        slippage=0.0005
    )
    backtest.set_strategy(my_strategy)
    backtest.load_market_data("AAPL", data)
    results = backtest.run()

    # Monte Carlo simulation
    mc = MonteCarloSimulator()
    paths = mc.simulate_gbm(mean_return=0.10, volatility=0.20)
"""

from backend.backtesting.event_driven import (
    EventDrivenBacktest,
    FillEvent,
    MarketEvent,
    OrderEvent,
    Portfolio,
    SignalEvent,
)
from backend.backtesting.monte_carlo import MonteCarloConfig, MonteCarloSimulator
from backend.backtesting.strategies import (
    BaseStrategy,
    MACDStrategy,
    MeanReversionStrategy,
    MomentumStrategy,
    MovingAverageCrossover,
    RSIStrategy,
)
from backend.backtesting.transaction_costs import TransactionCostModel
from backend.backtesting.vectorized import VectorizedBacktest

__all__ = [
    # Event-driven
    "EventDrivenBacktest",
    "Portfolio",
    "MarketEvent",
    "SignalEvent",
    "OrderEvent",
    "FillEvent",
    # Monte Carlo
    "MonteCarloSimulator",
    "MonteCarloConfig",
    # Strategies
    "BaseStrategy",
    "RSIStrategy",
    "MACDStrategy",
    "MomentumStrategy",
    "MeanReversionStrategy",
    "MovingAverageCrossover",
    "VectorizedBacktest",
    "TransactionCostModel",
]
