# backend/quant_engine/factors/__init__.py
"""
Quantitative Factor Analysis Module for Lumina 2.0
==================================================
Risk factor modules for quantitative analysis.

Modules:
- fama_french: Fama-French factor model (3, 5, 6 factors)
- pca_analysis: Principal Component Analysis for factor discovery
- risk_factors: Integrated risk factor framework

Author: Lumina Quant Lab
Version: 2.0.0
"""

from backend.quant_engine.factors.fama_french import (
    FactorAttribution,
    FactorModel,
    FamaFrenchAnalyzer,
    FamaFrenchDataLoader,
    Frequency,
    get_factor_data,
    get_factor_exposure,
)
from backend.quant_engine.factors.fama_french import (
    FactorExposure as FFFactorExposure,
)
from backend.quant_engine.factors.pca_analysis import (
    FactorInterpretation,
    PCAAnalyzer,
    PCAResults,
    RollingPCAResults,
    ScalingMethod,
    SelectionMethod,
    fit_pca,
    get_factor_scores,
)
from backend.quant_engine.factors.risk_factors import (
    CovarianceMethod,
    FactorExposure,
    FactorMimickingPortfolio,
    RiskDecomposition,
    RiskFactor,
    RiskFactorAnalyzer,
    RiskFactorCalculator,
    RiskFactorType,
    RiskMeasure,
    analyze_factor_exposures_async,
    calculate_risk_factors_async,
    decompose_portfolio_risk,
    get_standard_risk_factors,
)

__all__ = [
    # Fama-French module
    "FamaFrenchDataLoader",
    "FamaFrenchAnalyzer",
    "FactorModel",
    "Frequency",
    "FFFactorExposure",
    "FactorAttribution",
    "get_factor_exposure",
    "get_factor_data",
    # PCA module
    "PCAAnalyzer",
    "ScalingMethod",
    "SelectionMethod",
    "PCAResults",
    "FactorInterpretation",
    "RollingPCAResults",
    "fit_pca",
    "get_factor_scores",
    # Risk Factors Framework
    "RiskFactorCalculator",
    "RiskFactorAnalyzer",
    "RiskFactorType",
    "CovarianceMethod",
    "RiskMeasure",
    "RiskFactor",
    "FactorExposure",
    "RiskDecomposition",
    "FactorMimickingPortfolio",
    "calculate_risk_factors_async",
    "analyze_factor_exposures_async",
    "get_standard_risk_factors",
    "decompose_portfolio_risk",
]
