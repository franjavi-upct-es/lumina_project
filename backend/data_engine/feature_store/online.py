# backend/data_engine/feature_store/online.py
"""
Online Feature Store - Hot Storage (Redis)
==========================================

The "Short-Term Memory" of the Chimera Agent.
Stores pre-computed embeddings from perception encoders for real-time inference.

Architecture:
    - Redis as in-memory cache
    - TTL-based expiration (1 hour default)
    - Microsecond-level lookups
    - JSON-serialized vectors

Key Patterns:
    embeddings:price:{ticker}  → TFT embedding (128d vector)
    embeddings:news:{ticker}   → LLM embedding (64d vector)
    embeddings:graph:{ticker}  → GNN embedding (32d vector)
    features:latest:{ticker}   → Latest raw features (for fallback)

Usage Pattern (Inference Loop):
    1. Market tick arrives at execution engine
    2. Agent queries: embeddings = await online_store.get_embeddings("AAPL")
    3. Returns dict: {"price": [128d], "news": [64d], "graph": [32d]}
    4. Fusion layer processes in <10ms
    5. Action sent to broker

Author: Lumina Quant Lab
Version: 3.0.0
"""

import json
from datetime import datetime
from typing import Any

import numpy as np
from loguru import logger
from redis import Redis
from redis.asyncio import Redis as AsyncRedis

from backend.config.settings import get_settings
from backend.data_engine.feature_store.definitions import EmbeddingVector

settings = get_settings()


class OnlineFeatureStore:
    """
    Redis-based online feature store for real-time embeddings

    The "Hot Path" for inference - optimized for latency over completeness.
    """

    def __init__(
        self,
        redis_url: str | None = None,
        default_ttl: int = 3600,  # 1 hour
        embedding_ttl: int = 900,  # 15 minutes for embeddings
    ):
        """
        Initialize online feature store

        Args:
            redis_url: Redis connection URL
            default_ttl: Default TTL for features in seconds
            embedding_ttl: TTL for embeddings (shorter than features)
        """
        self.redis_url = redis_url or settings.REDIS_URL
        self.default_ttl = default_ttl
        self.embedding_ttl = embedding_ttl

        # Initialize Redis client
        self.redis: AsyncRedis | None = None
        self._sync_redis: Redis | None = None

        logger.info("OnlineFeatureStore initialized")

    async def connect(self):
        """Establish Redis connection"""
        if self.redis is None:
            self.redis = await AsyncRedis.from_url(
                self.redis_url,
                decode_responses=True,
                socket_connect_timeout=5,
            )
            logger.success("Connected to Redis (async)")

    async def disconnect(self):
        """Close Redis connection"""
        if self.redis:
            await self.redis.close()
            self.redis = None
            logger.info("Disconnected from Redis")

    async def store_embedding(
        self,
        ticker: str,
        encoder: str,
        vector: list[float] | np.ndarray,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """
        Store pre-computed embedding from perception encoder

        Args:
            ticker: Asset ticker
            encoder: Encoder name (tft, llm, gnn)
            vector: Embedding vector
            metadata: Additional metadata (confidence, model version, etc.)

        Returns:
            True if stored successfully
        """
        try:
            await self.connect()

            # Convert numpy array to list
            if isinstance(vector, np.ndarray):
                vector = vector.tolist()

            # Create embedding object
            embedding = EmbeddingVector(
                ticker=ticker,
                encoder=encoder,
                vector=vector,
                dimension=len(vector),
                timestamp=datetime.utcnow(),
                metadata=metadata or {},
            )

            # Store with key: embeddings:{encoder}:{ticker}
            key = f"embeddings:{encoder}:{ticker}"
            value = embedding.model_dump_json()

            await self.redis.set(key, value, ex=self.embedding_ttl)

            logger.debug(f"Stored {encoder} embedding for {ticker} (dim={len(vector)})")
            return True

        except Exception as e:
            logger.error(f"Error storing embedding: {e}")
            return False

    async def get_embedding(
        self,
        ticker: str,
        encoder: str,
    ) -> np.ndarray | None:
        """
        Retrieve embedding vector from hot storage

        Args:
            ticker: Asset ticker
            encoder: Encoder name (tft, llm, gnn)

        Returns:
            Numpy array of embedding or None if not found/expired
        """
        try:
            await self.connect()

            key = f"embeddings:{encoder}:{ticker}"
            value = await self.redis.get(key)

            if value is None:
                logger.warning(f"Embedding not found: {key}")
                return None

            # Deserialize
            embedding = EmbeddingVector.model_validate_json(value)

            # Check if stale (beyond TTL + grace period)
            age = (datetime.utcnow() - embedding.timestamp).total_seconds()
            if age > (self.embedding_ttl + 300):  # 5 min grace period
                logger.warning(f"Stale embedding: {key} (age={age}s)")
                return None

            return np.array(embedding.vector, dtype=np.float32)

        except Exception as e:
            logger.error(f"Error retrieving embedding: {e}")
            return None

    async def get_embeddings(
        self,
        ticker: str,
        encoders: list[str] | None = None,
    ) -> dict[str, np.ndarray]:
        """
        Retrieve all embeddings for a ticker (multi-modal)

        This is the PRIMARY method used by the agent during inference.
        Returns all available embeddings for fusion layer.

        Args:
            ticker: Asset ticker
            encoders: List of encoders to fetch (default: all)

        Returns:
            Dict mapping encoder name to embedding vector
            Example: {"tft": array(128d), "llm": array(64d), "gnn": array(32d)}
        """
        if encoders is None:
            encoders = ["tft", "llm", "gnn"]  # Default V3 encoders

        embeddings = {}

        for encoder in encoders:
            vector = await self.get_embedding(ticker, encoder)
            if vector is not None:
                embeddings[encoder] = vector

        if not embeddings:
            logger.warning(f"No embeddings available for {ticker}")
        else:
            logger.debug(f"Retrieved {len(embeddings)} embeddings for {ticker}")

        return embeddings

    async def store_latest_features(
        self,
        ticker: str,
        features: dict[str, float],
        timestamp: datetime | None = None,
    ) -> bool:
        """
        Store latest raw feature values (fallback for missing embeddings)

        Args:
            ticker: Asset ticker
            features: Dict of feature_name -> value
            timestamp: Feature timestamp

        Returns:
            True if stored successfully
        """
        try:
            await self.connect()

            data = {
                "ticker": ticker,
                "features": features,
                "timestamp": (timestamp or datetime.utcnow()).isoformat(),
            }

            key = f"features:latest:{ticker}"
            value = json.dumps(data)

            await self.redis.set(key, value, ex=self.default_ttl)

            logger.debug(f"Stored {len(features)} features for {ticker}")
            return True

        except Exception as e:
            logger.error(f"Error storing features: {e}")
            return False

    async def get_latest_features(
        self,
        ticker: str,
    ) -> dict[str, float] | None:
        """
        Retrieve latest raw features

        Args:
            ticker: Asset ticker

        Returns:
            Dict of feature_name -> value or None
        """
        try:
            await self.connect()

            key = f"features:latest:{ticker}"
            value = await self.redis.get(key)

            if value is None:
                return None

            data = json.loads(value)
            return data.get("features")

        except Exception as e:
            logger.error(f"Error retrieving features: {e}")
            return None

    async def clear_ticker(self, ticker: str) -> int:
        """
        Clear all cached data for a ticker

        Args:
            ticker: Asset ticker

        Returns:
            Number of keys deleted
        """
        try:
            await self.connect()

            # Find all keys for this ticker
            pattern = f"*:{ticker}"
            keys = []

            async for key in self.redis.scan_iter(match=pattern):
                keys.append(key)

            if keys:
                deleted = await self.redis.delete(*keys)
                logger.info(f"Cleared {deleted} keys for {ticker}")
                return deleted

            return 0

        except Exception as e:
            logger.error(f"Error clearing ticker: {e}")
            return 0

    async def health_check(self) -> dict[str, Any]:
        """
        Check Redis connection health

        Returns:
            Health status dict
        """
        try:
            await self.connect()

            # Ping Redis
            await self.redis.ping()

            # Get info
            info = await self.redis.info()

            # Count keys
            total_keys = await self.redis.dbsize()

            # Sample keys
            sample_embeddings = 0
            sample_features = 0

            async for key in self.redis.scan_iter(match="embeddings:*", count=100):
                sample_embeddings += 1

            async for key in self.redis.scan_iter(match="features:*", count=100):
                sample_features += 1

            return {
                "status": "healthy",
                "redis_version": info.get("redis_version"),
                "total_keys": total_keys,
                "embedding_keys": sample_embeddings,
                "feature_keys": sample_features,
                "memory_used_mb": info.get("used_memory") / 1024 / 1024,
            }

        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
            }


# Global instance
_online_store: OnlineFeatureStore | None = None


def get_online_store() -> OnlineFeatureStore:
    """Get global online feature store instance"""
    global _online_store
    if _online_store is None:
        _online_store = OnlineFeatureStore()
    return _online_store
