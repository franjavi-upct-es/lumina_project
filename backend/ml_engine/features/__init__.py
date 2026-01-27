# backend/ml_engine/features/__init__.py
"""
ML Features Module for Lumina Quant Lab

Provides feature computation for machine learning models:

TechnicalFeatures:
- Price-based features (returns, gaps, ranges)
- Volume indicators (OBV, VWAP, volume ratios)
- Volatility measures (ATR, Bollinger Bands)
- Momentum oscillators (RSI, Stochastic, MACD)
- Trend indicators (ADX, Parabolic SAR)

SentimentFeatures:
- News sentiment scores
- Social media sentiment
- Analyst recommendations
- Options market sentiment

FundamentalFeatures:
- Valuation ratios (P/E, P/B, EV/EBITDA)
- Growth metrics (revenue, earnings)
- Quality factors (ROE, ROA, margins)

MacroFeatures:
- Interest rates
- Inflation indicators
- Economic growth
- Market regime indicators

Usage:
    from backend.ml_engine.features import TechnicalFeatures

    tech_features = TechnicalFeatures()
    features = tech_features.compute(data)
"""

from backend.ml_engine.features.fundamental import FundamentalFeatures
from backend.ml_engine.features.macro import MacroFeatures
from backend.ml_engine.features.sentiment import SentimentFeatures
from backend.ml_engine.features.technical import TechnicalFeatures

__all__ = [
    "TechnicalFeatures",
    "SentimentFeatures",
    "FundamentalFeatures",
    "MacroFeatures",
]
