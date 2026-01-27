# backend/data_engine/__init__.py
"""
Data Engine Module for Lumina Quant Lab

Provides comprehensive data collection, transformation, and storage capabilities:

Collectors:
- YFinanceCollector: Yahoo Finance historical data
- AlphaVantageCollector: Premium market data
- FREDCollector: Economic indicators
- NewsCollector: Financial news
- RedditCollector: Social sentiment

Transformers:
- FeatureEngineer: 100+ technical indicators
- RegimeDetector: Market regime classification
- Normalizer: Data normalization utilities

Storage:
- TimescaleAdapter: Time-series database operations
- ParquetWriter: Efficient file storage
- FeatureStore: Feature versioning and retrieval

Usage:
    from backend.data_engine import YFinanceCollector, FeatureEngineer

    # Collect data
    collector = YFinanceCollector()
    data = await collector.collect_with_retry("AAPL", start_date, end_date)

    # Engineer features
    fe = FeatureEngineer()
    enriched_data = fe.create_all_features(data)
"""

# Collectors
from backend.data_engine.collectors import (
    BaseDataCollector,
    YFinanceCollector,
)

# Transformers
from backend.data_engine.transformers import (
    FeatureEngineer,
)

# Storage (lazy imports to avoid DB connection at import time)
# from backend.data_engine.storage import TimescaleAdapter, ParquetWriter, FeatureStore

__all__ = [
    # Collectors
    "BaseDataCollector",
    "YFinanceCollector",
    # Transformers
    "FeatureEngineer",
    # Storage
    "TimescaleAdapter",
    "ParquetWriter",
    "FeatureStore",
]
