# backend/data_engine/transformers/__init__.py
"""
Transformers Module for Lumina V3
=================================

Feature transformation and engineering pipelines.

Modules:
- feature_engineering: Technical indicators and derived features
- normalization: Feature scaling and normalization
- regime_detection: Market regime classification

Version: 3.0.0
"""

from backend.data_engine.transformers.feature_engineering import FeatureEngineer
from backend.data_engine.transformers.normalization import FeatureNormalizer
from backend.data_engine.transformers.regime_detection import RegimeDetector

__all__ = [
    "FeatureEngineer",
    "FeatureNormalizer",
    "RegimeDetector",
]
