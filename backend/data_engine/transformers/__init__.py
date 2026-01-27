# backend/data_engine/transformers/__init__.py
"""
Data Transformers Module for Lumina Quant Lab

Provides data transformation and feature engineering capabilities:

FeatureEngineer:
- Price features (returns, gaps, ranges)
- Volume features (OBV, VWAP, volume ratios)
- Volatility features (ATR, Bollinger Bands, historical vol)
- Momentum features (RSI, Stochastic, ROC, MFI)
- Trend features (Moving averages, MACD, ADX, SAR)
- Statistical features (Skewness, kurtosis, z-scores)
- Lag features for time-series modeling
- Rolling window statistics

RegimeDetector:
- Hidden Markov Model regime detection
- Volatility regime classification
- Trend/sideways/reversal detection

Normalizer:
- Z-score normalization
- Min-max scaling
- Robust scaling for outliers

Usage:
    from backend.data_engine.transformers import FeatureEngineer

    fe = FeatureEngineer()
    enriched_data = fe.create_all_features(
        data,
        add_lags=True,
        add_rolling=True
    )

    feature_names = fe.get_all_feature_names()
"""

from backend.data_engine.transformers.feature_engineering import FeatureEngineer
from backend.data_engine.transformers.normalization import DataNormalizer
from backend.data_engine.transformers.regime_detection import RegimeDetector

__all__ = [
    "FeatureEngineer",
    "RegimeDetector",
    "DataNormalizer",
]
