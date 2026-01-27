# backend/quant_engine/__init__.py
"""
Quantitative Engine Module for Lumina Quant Lab

Provides advanced quantitative analysis tools:

Risk:
- VaR Calculator: Value at Risk computation
- CVaR Calculator: Conditional VaR (Expected Shortfall)
- Drawdown: Drawdown analysis and metrics
- Stress Testing: Historical and hypothetical scenarios

Regimes:
- HMM Detector: Hidden Markov Model regime detection
- Clustering: K-means based regime clustering
- Volatility Regimes: Volatility-based regime classification

Statistics:
- Cointegration: Engle-Granger, Johansen tests
- Causality: Granger causality testing
- Stationarity: ADF, KPSS tests

Factors:
- Fama-French: Factor loading computation
- PCA Analysis: Principal component analysis
- Risk Factors: Custom factor construction

Optimization:
- Portfolio Optimizer: Mean-variance optimization
- Black-Litterman: Views-based optimization
- Genetic Algorithm: Evolutionary optimization

Usage:
    from backend.quant_engine import VaRCalculator, PortfolioOptimizer

    # Calculate VaR
    var_calc = VaRCalculator()
    var_95 = var_calc.historical_var(returns, confidence=0.95)

    # Optimize portfolio
    optimizer = PortfolioOptimizer(returns)
    weights = optimizer.max_sharpe()
"""

from backend.quant_engine.factors import FamaFrench, PCAAnalysis
from backend.quant_engine.optimization import BlackLitterman, PortfolioOptimizer
from backend.quant_engine.regimes import HMMDetector, VolatilityRegimes
from backend.quant_engine.risk import CVaRCalculator, DrawdownAnalyzer, VaRCalculator
from backend.quant_engine.statistics import Causality, Cointegration, Stationarity

__all__ = [
    # Risk
    "VaRCalculator",
    "CVaRCalculator",
    "DrawdownAnalyzer",
    # Regimes
    "HMMDetector",
    "VolatilityRegimes",
    # Statistics
    "Cointegration",
    "Causality",
    "Stationarity",
    # Factors
    "FamaFrench",
    "PCAAnalysis",
    # Optimization
    "PortfolioOptimizer",
    "BlackLitterman",
]
