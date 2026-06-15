# backend/feature_store/online.py
"""Online Feature Store: Redis-backed, sub-millisecond reads for inference."""

from __future__ import annotations

from typing import cast

import numpy as np
from prometheus_client import Histogram

from backend.data_engine.storage.redis_cache import EmbeddingKind, RedisCache
from backend.feature_store.definitions import FEATURE_REGISTRY, FeatureDef

FEATURE_STORE_ONLINE_LATENCY = Histogram(
    "feature_store_online_latency_seconds",
    "Latency of online feature retrievals",
    labelnames=("method",),
    buckets=(0.0005, 0.001, 0.002, 0.005, 0.01, 0.05, 0.1),
)

_FEATURE_TO_KIND: dict[str, EmbeddingKind] = {
    "price_emb": "price",
    "semantic_emb": "semantic",
    "graph_emb": "graph",
}


class OnlineFeatureStore:
    def __init__(self, redis: RedisCache) -> None:
        self._redis = redis

    async def get(self, feature_name: str, ticker: str) -> np.ndarray | None:
        fdef = self._resolve_hot(feature_name)
        kind = _FEATURE_TO_KIND[fdef.name]
        with FEATURE_STORE_ONLINE_LATENCY.labels(method="get").time():
            return await self._redis.get_embedding(kind, ticker)

    async def mget(self, feature_name: str, tickers: list[str]) -> dict[str, np.ndarray]:
        fdef = self._resolve_hot(feature_name)
        kind = _FEATURE_TO_KIND[fdef.name]
        with FEATURE_STORE_ONLINE_LATENCY.labels(method="mget").time():
            return await self._redis.mget_embeddings(kind, tickers)

    async def get_bundle(
        self,
        ticker: str,
        feature_names: list[str] | None = None,
    ) -> dict[str, np.ndarray]:
        names = feature_names or list(_FEATURE_TO_KIND.keys())
        for n in names:
            self._resolve_hot(n)
        keys = [f"emb:{_FEATURE_TO_KIND[n]}:{ticker}" for n in names]
        with FEATURE_STORE_ONLINE_LATENCY.labels(method="get_bundle").time():
            raws = await self._redis.client.mget(keys)
        out: dict[str, np.ndarray] = {}
        for name, raw in zip(names, raws, strict=True):
            if raw is None:
                continue
            out[name] = np.frombuffer(cast(bytes, raw), dtype=np.float32).copy()
        return out

    @staticmethod
    def _resolve_hot(feature_name: str) -> FeatureDef:
        if feature_name not in FEATURE_REGISTRY:
            raise KeyError(f"Unknown feature: {feature_name}")
        fdef = FEATURE_REGISTRY[feature_name]
        if not fdef.is_hot:
            raise ValueError(f"Feature '{feature_name}' is not hot ({fdef.source})")
        if feature_name not in _FEATURE_TO_KIND:
            raise ValueError(f"Hot feature '{feature_name}' has no kind mapping")
        return fdef
