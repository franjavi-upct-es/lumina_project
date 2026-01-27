# backend/quant_engine/regimes/__init__.py
"""
Market Regime Detection Module for Lumina Quant Lab

Provides regime classification for adaptive strategies:

HMMDetector:
- Hidden Markov Model for regime detection
- Bull/Bear/Sideways classification
- Transition probability estimation
- Regime probability time series

Clustering:
- K-means regime clustering
- DBSCAN for outlier regimes
- Hierarchical clustering

VolatilityRegimes:
- Volatility-based regime classification
- Low/Normal/High volatility states
- GARCH regime switching

Usage:
    from backend.quant_engine.regimes import HMMDetector

    detector = HMMDetector(n_regimes=3)
    detector.fit(returns)

    current_regime = detector.predict_regime(recent_returns)
    regime_probs = detector.get_regime_probabilities()
"""

from backend.quant_engine.regimes.clustering import RegimeClustering
from backend.quant_engine.regimes.hmm_detector import HMMDetector
from backend.quant_engine.regimes.volatility_regimes import VolatilityRegimes

__all__ = [
    "HMMDetector",
    "RegimeClustering",
    "VolatilityRegimes",
]
