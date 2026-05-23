# backend/perception/common/embedding_writer.py
"""Unified writer from encoders into the online Feature Store."""

from __future__ import annotations

import numpy as np
from loguru import logger
from prometheus_client import Counter, Histogram

from backend.data_engine.storage.redis_cache import EmbeddingKind, RedisCache

EMBEDDINGS_WRITTEN = Counter("embeddings_written_total", "Embeddings written", labelnames=("kind",))
EMBEDDING_WRITE_LATENCY = Histogram(
    "embedding_write_latency_seconds",
    "Latency of embedding writes",
    labelnames=("kind",),
    buckets=(0.001, 0.002, 0.005, 0.01, 0.05, 0.1),
)


class EmbeddingWriter:
    def __init__(self, redis: RedisCache) -> None:
        self._redis = redis

    async def write(self, kind: EmbeddingKind, ticker: str, vec: np.ndarray) -> None:
        if not np.isfinite(vec).all():
            logger.warning(f"Skipped {kind}/{ticker}: non-finite values")
            return
        with EMBEDDING_WRITE_LATENCY.labels(kind=kind).time():
            await self._redis.set_embedding(kind, ticker, vec)
        EMBEDDINGS_WRITTEN.labels(kind=kind).inc()

    async def write_batch(self, kind: EmbeddingKind, vecs: dict[str, np.ndarray]) -> None:
        for ticker, vec in vecs.items():
            await self.write(kind, ticker, vec)
