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

from importlib import import_module
from typing import Any

from backend.data_engine.transformers.feature_engineering import FeatureEngineer

_LAZY_EXPORTS = {
    "RegimeDetector": ("backend.data_engine.transformers.regime_detection", "RegimeDetector"),
}


def __getattr__(name: str) -> Any:
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attribute_name = _LAZY_EXPORTS[name]
    try:
        value = getattr(import_module(module_name), attribute_name)
    except ModuleNotFoundError as exc:
        if exc.name == "hmmlearn":
            raise ModuleNotFoundError(
                f"{name} requires hmmlearn. Install the ml dependency group with "
                "`uv sync --group ml`."
            ) from exc
        raise

    globals()[name] = value
    return value


__all__ = [
    "FeatureEngineer",
    "RegimeDetector",
]
