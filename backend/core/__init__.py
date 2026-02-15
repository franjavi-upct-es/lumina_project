# backend/core/__init__.py
"""
Core Integration Module for Lumina V3 Quant Lab
===============================================

This module provides unified access to V3 components following the Deep Fusion Architecture.
V3 represents a paradigm shift from V2's linear processing to cognitive autonomous trading.

Architecture Layers:
- Perception Layer: Multi-modal encoders (Temporal, Semantic, Structural)
- Fusion Layer: Deep sensor fusion with cross-modal attention
- Cognition Layer: Hierarchical RL with PPO/SAC agents
- Safety Layer: Uncertainty gates and hard risk protocols

Quick Start (V3):
    from backend.core import (
        # Feature Store (Hot/Cold Architecture)
        get_feature_store_client,
        OnlineFeatureStore,
        OfflineFeatureStore,

        # Data Collection
        YFinanceCollector,
        AlphaVantageCollector,
        FREDCollector,
        NewsCollector,

        # Feature Engineering
        FeatureEngineer,
        RegimeDetector,

        # Configuration
        settings,
    )

    # Access feature store
    feature_store = get_feature_store_client()

    # Collect and engineer features
    collector = YFinanceCollector()
    data = await collector.collect_with_retry("AAPL", start_date, end_date)

    # Engineer features for perception layer
    fe = FeatureEngineer()
    features = fe.create_all_features(data)

    # Store in feature store (offline)
    await feature_store.offline.store_features("AAPL", features)

Version: 3.0.0 (Deep Fusion Architecture)
Author: Lumina Quant Lab
"""

from typing import TYPE_CHECKING

# ============================================================================
# CONFIGURATION
# ============================================================================
from backend.config import get_settings, settings

# ============================================================================
# DATA ENGINE - Collectors
# ============================================================================
from backend.data_engine.collectors import (
    AlphaVantageCollector,
    BaseDataCollector,
    FREDCollector,
    NewsCollector,
    RedditCollector,
    YFinanceCollector,
)

# ============================================================================
# DATA ENGINE - Feature Store (V3 Hot/Cold Architecture)
# ============================================================================
from backend.data_engine.feature_store import (
    FeatureStoreClient,
    OfflineFeatureStore,
    OnlineFeatureStore,
    get_feature_store_client,
)
from backend.data_engine.feature_store.definitions import (
    FeatureCategory,
    FeatureDefinition,
    FeatureMetadata,
)

# ============================================================================
# DATA ENGINE - Pipelines
# ============================================================================
from backend.data_engine.pipelines import (
    CleaningPipeline,
    IngestionPipeline,
)

# ============================================================================
# DATA ENGINE - Storage
# ============================================================================
from backend.data_engine.storage import (
    ParquetWriter,
    RedisCache,
    TimescaleAdapter,
)

# ============================================================================
# DATA ENGINE - Transformers
# ============================================================================
from backend.data_engine.transformers import (
    FeatureEngineer,
    FeatureNormalizer,
    RegimeDetector,
)

# ============================================================================
# TYPE CHECKING ONLY IMPORTS
# ============================================================================
if TYPE_CHECKING:
    from backend.db import Feature, Model, PriceData


# ============================================================================
# MODULE EXPORTS
# ============================================================================

__all__ = [
    # Configuration
    "settings",
    "get_settings",
    # Feature Store (V3 Hot/Cold)
    "FeatureStoreClient",
    "OnlineFeatureStore",
    "OfflineFeatureStore",
    "get_feature_store_client",
    "FeatureCategory",
    "FeatureDefinition",
    "FeatureMetadata",
    # Data Collectors
    "BaseDataCollector",
    "YFinanceCollector",
    "AlphaVantageCollector",
    "FREDCollector",
    "NewsCollector",
    "RedditCollector",
    # Transformers
    "FeatureEngineer",
    "FeatureNormalizer",
    "RegimeDetector",
    # Pipelines
    "IngestionPipeline",
    "CleaningPipeline",
    # Storage
    "RedisCache",
    "TimescaleAdapter",
    "ParquetWriter",
]


# ============================================================================
# VERSION INFO
# ============================================================================

__version__ = "3.0.0"
__architecture__ = "Deep Fusion"
__status__ = "Alpha - Infrastructure Layer"
