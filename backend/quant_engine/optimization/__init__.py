# backend/quant_engine/optimization/__init__.py
"""
Portfolio Optimization Module for Lumina Quant Lab

Provides various portfolio optimization methods:

PortfolioOptimizer:
- Mean-variance optimization (Markowitz)
- Minimum volatility portfolio
- Maximum Sharpe ratio portfolio
- Risk parity
- Hierarchical Risk Parity (HRP)

BlackLitterman:
- Views-based optimization
- Prior estimation
- Posterior computation
- Confidence weighting

GeneticAlgorithm:
- Evolutionary optimization
- Multi-objective optimization
- Constraint handling

Usage:
    from backend.quant_engine.optimization import PortfolioOptimizer

    optimizer = PortfolioOptimizer(returns)

    # Maximum Sharpe ratio
    weights_sharpe = optimizer.max_sharpe()

    # Minimum volatility
    weights_min_vol = optimizer.min_volatility()

    # Risk parity
    weights_rp = optimizer.risk_parity()
"""

from backend.quant_engine.optimization.black_litterman import BlackLitterman
from backend.quant_engine.optimization.genetic_algorithm import GeneticOptimizer
from backend.quant_engine.optimization.portfolio_optimizer import PortfolioOptimizer

__all__ = [
    "PortfolioOptimizer",
    "BlackLitterman",
    "GeneticOptimizer",
]
