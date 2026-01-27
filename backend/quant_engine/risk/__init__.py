# backend/quant_engine/risk/__init__.py
"""
Risk Analysis Module for Lumina Quant Lab

Provides comprehensive risk metrics and analysis:

VaRCalculator:
- Historical VaR
- Parametric VaR (variance-covariance)
- Monte Carlo VaR
- Cornish-Fisher VaR (adjusted for skewness/kurtosis)

CVaRCalculator:
- Expected Shortfall computation
- Tail risk analysis
- Conditional expectations

DrawdownAnalyzer:
- Maximum drawdown
- Drawdown duration
- Recovery analysis
- Underwater periods

StressTesting:
- Historical scenarios (2008, COVID, etc.)
- Hypothetical scenarios
- Factor shocks
- Correlation breakdown scenarios

Usage:
    from backend.quant_engine.risk import VaRCalculator

    var_calc = VaRCalculator()
    var_95 = var_calc.historical_var(returns, confidence=0.95)
    cvar_95 = var_calc.expected_shortfall(returns, confidence=0.95)
"""

from backend.quant_engine.risk.cvar_calculator import CVaRCalculator
from backend.quant_engine.risk.drawdown import DrawdownAnalyzer
from backend.quant_engine.risk.stress_testing import StressTesting
from backend.quant_engine.risk.var_calculator import VaRCalculator

__all__ = [
    "VaRCalculator",
    "CVaRCalculator",
    "DrawdownAnalyzer",
    "StressTesting",
]
