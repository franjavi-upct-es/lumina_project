# backend/data_engine/__init__.py
"""
Data Engine for Lumina V3
=========================

The Data Engine handles all data acquisition, transformation, and storage
for the V3 Deep Fusion Architecture.

Components:
- Collectors: Data acquisition from various sources
- Transformers: Feature engineering and normalization
- Pipelines: ETL orchestration
- Storage: TimescaleDB, Redis, and Parquet adapters
- Feature Store: Hot/Cold storage for features and embeddings

Usage:
    from backend.data_engine import (
        # Feature Store
        get_feature_store_client,

        # Collectors
        YFinanceCollector,
        AlphaVantageCollector,
        FREDCollector,

        # Transformers
        FeatureEngineer,
        FeatureNormalizer,
        RegimeDetector,

        # Pipelines
        IngestionPipeline,
        CleaningPipeline,
    )

    # Collect and process data
    collector = YFinanceCollector()
    data = await collector.collect_with_retry("AAPL", start, end)

    # Engineer features
    engineer = FeatureEngineer()
    features = engineer.create_all_features(data)

    # Store in feature store
    fs = get_feature_store_client()
    await fs.store_features("AAPL", features)

    # Retrieve for inference
    embeddings = await fs.get_embeddings("AAPL")

Version: 3.0.0
"""

# Feature Store
# Collectors
from backend.data_engine.collectors import (
    AlphaVantageCollector,
    BaseDataCollector,
    FREDCollector,
    NewsCollector,
    RedditCollector,
    YFinanceCollector,
)
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

# Pipelines
from backend.data_engine.pipelines import (
    CleaningPipeline,
    IngestionPipeline,
)

# Storage
from backend.data_engine.storage import (
    ParquetWriter,
    RedisCache,
    TimescaleAdapter,
)

# Transformers
from backend.data_engine.transformers import (
    FeatureEngineer,
    FeatureNormalizer,
    RegimeDetector,
)

__all__ = [
    # Feature Store
    "FeatureStoreClient",
    "OnlineFeatureStore",
    "OfflineFeatureStore",
    "get_feature_store_client",
    "FeatureCategory",
    "FeatureDefinition",
    "FeatureMetadata",
    # Collectors
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
    "TimescaleAdapter",
    "RedisCache",
    "ParquetWriter",
]

__version__ = "3.0.0"
__status__ = "Alpha - Infrastructure Layer"
