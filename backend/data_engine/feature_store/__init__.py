# backend/data_engine/feature_store/__init__.py
"""
Feature Store Module for Lumina V3 - Hot/Cold Architecture
===========================================================

The Feature Store is the "Short-Term Memory" of the Chimera Agent.
It implements a dual-storage strategy to support low-latency inference:

Hot Storage (Online - Redis):
    - Pre-computed embeddings from perception encoders
    - TTL-based caching for real-time access
    - Microsecond-level lookup for trading decisions
    - Stores: Price embeddings (TFT), Semantic vectors (LLM), Graph embeddings (GNN)

Cold Storage (Offline - TimescaleDB + Parquet):
    - Historical features for model training
    - Raw technical indicators and transformations
    - Compressed columnar storage for batch access
    - Supports time-travel queries and feature versioning

Architecture Pattern:
    ┌─────────────────────────────────────────────┐
    │         Perception Encoders                 │
    │  (TFT, NLP Engine, GNN - Background Jobs)   │
    └─────────────────┬───────────────────────────┘
                      │
                      ▼
    ┌─────────────────────────────────────────────┐
    │      Online Store (Redis - Hot)             │
    │  embeddings:price:AAPL → [128d vector]      │
    │  embeddings:news:AAPL  → [64d vector]       │
    │  embeddings:graph:AAPL → [32d vector]       │
    │  TTL: 1 hour (auto-expire stale data)       │
    └─────────────────┬───────────────────────────┘
                      │
                      │ (Async sync every 15min)
                      ▼
    ┌─────────────────────────────────────────────┐
    │   Offline Store (TimescaleDB + Parquet)     │
    │  features_table: Raw indicators             │
    │  /features/AAPL/2024/01/*.parquet           │
    │  Purpose: Training data, backtesting        │
    └─────────────────────────────────────────────┘

Usage:
    from backend.data_engine.feature_store import get_feature_store_client

    # Initialize unified client
    fs = get_feature_store_client()

    # Store features (automatically routes to hot + cold)
    await fs.store_features(
        ticker="AAPL",
        features=feature_df,
        embeddings={"price": price_vec, "news": news_vec}
    )

    # Retrieve for inference (hot path - microseconds)
    embeddings = await fs.online.get_embeddings("AAPL")

    # Retrieve for training (cold path - milliseconds)
    historical = await fs.offline.get_features(
        ticker="AAPL",
        start_date=datetime(2023, 1, 1),
        end_date=datetime(2024, 1, 1)
    )

Version: 3.0.0
"""

from backend.data_engine.feature_store.client import (
    FeatureStoreClient,
    get_feature_store_client,
)
from backend.data_engine.feature_store.definitions import (
    FeatureCategory,
    FeatureDefinition,
    FeatureMetadata,
)
from backend.data_engine.feature_store.offline import OfflineFeatureStore
from backend.data_engine.feature_store.online import OnlineFeatureStore

__all__ = [
    # Client
    "FeatureStoreClient",
    "get_feature_store_client",
    # Stores
    "OnlineFeatureStore",
    "OfflineFeatureStore",
    # Definitions
    "FeatureCategory",
    "FeatureDefinition",
    "FeatureMetadata",
]
