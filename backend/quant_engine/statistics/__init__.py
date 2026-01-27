# backend/quant_engine/statistics/__init__.py
"""
Statistical Analysis Module for Lumina Quant Lab

Provides statistical tests and analysis for financial data:

Cointegration:
- Engle-Granger two-step test
- Johansen test for multiple series
- Cointegration vector estimation
- Error correction models

Causality:
- Granger causality testing
- VAR-based causality
- Information flow analysis

Stationarity:
- Augmented Dickey-Fuller test
- KPSS test
- Phillips-Perron test
- Unit root testing

Usage:
    from backend.quant_engine.statistics import Cointegration, Stationarity

    # Test for cointegration
    coint = Cointegration()
    result = coint.engle_granger(series1, series2)

    # Test for stationarity
    stat = Stationarity()
    is_stationary = stat.adf_test(returns)
"""

from backend.quant_engine.statistics.causality import CausalityTester as Causality
from backend.quant_engine.statistics.cointegration import Cointegration
from backend.quant_engine.statistics.stationarity import Stationarity

__all__ = [
    "Cointegration",
    "Causality",
    "Stationarity",
]
