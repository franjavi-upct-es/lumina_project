# backend/data_engine/feature_store/client.py
"""
Unified Feature Store Client
============================

Single interface to both Hot and Cold storage layers.
Routes operations to appropriate backend based on use case.

Usage:
    client = get_feature_store_client()

    # Store features (routes to both hot and cold)
    await client.store_features(ticker, features, embeddings)

    # Get for inference (hot path)
    embeddings = await client.get_embeddings(ticker)

    # Get for training (cold path)
    features = await client.get_historical_features(ticker, start, end)

Version: 3.0.0
"""

from datetime import datetime
from typing import Any

import numpy as np
import polars as pl
from loguru import logger

from backend.data_engine.feature_store.definitions import FeatureMetadata
from backend.data_engine.feature_store.offline import OfflineFeatureStore, get_offline_store
from backend.data_engine.feature_store.online import OnlineFeatureStore, get_online_store


class FeatureStoreClient:
    """
    Unified client for Hot/Cold feature storage

    Automatically routes operations to appropriate backend.
    """

    def __init__(
        self,
        online_store: OnlineFeatureStore | None = None,
        offline_store: OfflineFeatureStore | None = None,
    ):
        """
        Initialize feature store client

        Args:
            online_store: Redis-based online store
            offline_store: TimescaleDB/Parquet offline store
        """
        self.online = online_store or get_online_store()
        self.offline = offline_store or get_offline_store()

        logger.info("FeatureStoreClient initialized")

    async def store_features(
        self,
        ticker: str,
        features: pl.DataFrame,
        embeddings: dict[str, np.ndarray] | None = None,
        feature_names: list[str] | None = None,
        categories: dict[str, str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> FeatureMetadata:
        """
        Store features in both hot and cold storage

        Args:
            ticker: Asset ticker
            features: DataFrame with raw features
            embeddings: Pre-computed embeddings {"tft": array, "llm": array, "gnn": array}
            feature_names: Feature column names
            categories: Feature categories
            metadata: Additional metadata

        Returns:
            FeatureMetadata object
        """
        try:
            # Store in cold storage (TimescaleDB + Parquet)
            feature_metadata = await self.offline.store_features(
                ticker=ticker,
                features=features,
                feature_names=feature_names,
                categories=categories,
                metadata=metadata,
            )

            # Store embeddings in hot storage (Redis)
            if embeddings:
                for encoder, vector in embeddings.items():
                    await self.online.store_embedding(
                        ticker=ticker,
                        encoder=encoder,
                        vector=vector,
                        metadata={
                            "model_version": metadata.get("model_version") if metadata else None
                        },
                    )

            # Also store latest raw features in hot storage (fallback)
            if len(features) > 0:
                latest_row = features.tail(1).to_dict(as_series=False)
                latest_features = {k: v[0] for k, v in latest_row.items() if k != "time"}
                await self.online.store_latest_features(ticker, latest_features)

            logger.success(f"Stored features for {ticker} (hot + cold)")
            return feature_metadata

        except Exception as e:
            logger.error(f"Error storing features: {e}")
            raise

    async def get_embeddings(
        self,
        ticker: str,
        encoders: list[str] | None = None,
    ) -> dict[str, np.ndarray]:
        """
        Get embeddings for inference (hot path)

        Args:
            ticker: Asset ticker
            encoders: List of encoders

        Returns:
            Dict of encoder -> embedding vector
        """
        return await self.online.get_embeddings(ticker, encoders)

    async def get_historical_features(
        self,
        ticker: str,
        start_date: datetime,
        end_date: datetime,
        feature_names: list[str] | None = None,
    ) -> pl.DataFrame | None:
        """
        Get historical features for training (cold path)

        Args:
            ticker: Asset ticker
            start_date: Start of time range
            end_date: End of time range
            feature_names: Specific features to retrieve

        Returns:
            Polars DataFrame with features
        """
        return await self.offline.get_features(
            ticker=ticker,
            feature_names=feature_names,
            start_date=start_date,
            end_date=end_date,
        )

    async def clear_ticker(self, ticker: str) -> tuple[int, int]:
        """
        Clear all data for a ticker (hot + cold)

        Returns:
            (online_keys_deleted, offline_records_deleted)
        """
        online_deleted = await self.online.clear_ticker(ticker)
        offline_deleted = await self.offline.delete_features(ticker)

        logger.info(f"Cleared {ticker}: {online_deleted} online + {offline_deleted} offline")
        return (online_deleted, offline_deleted)

    async def health_check(self) -> dict[str, Any]:
        """
        Check health of both storage backends

        Returns:
            Combined health status
        """
        online_health = await self.online.health_check()

        return {
            "online": online_health,
            "offline": {"status": "healthy"},  # Simplified
            "overall": "healthy" if online_health["status"] == "healthy" else "degraded",
        }


# Global instance
_client: FeatureStoreClient | None = None


def get_feature_store_client() -> FeatureStoreClient:
    """Get global feature store client"""
    global _client
    if _client is None:
        _client = FeatureStoreClient()
    return _client
