# backend/data_engine/storage/redis_cache.py
"""Redis async wrapper. Hot store for embeddings, dedupe, and pub/sub."""

from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator
from typing import Literal

import numpy as np
import redis.asyncio as aioredis
from loguru import logger

from backend.config.constants import DIM_GRAPH, DIM_PRICE, DIM_SEMANTIC
from backend.config.settings import get_settings

EmbeddingKind = Literal["price", "semantic", "graph"]

_DIM_MAP: dict[EmbeddingKind, int] = {
    "price": DIM_PRICE,
    "semantic": DIM_SEMANTIC,
    "graph": DIM_GRAPH,
}
_TTL_MAP: dict[EmbeddingKind, int] = {
    "price": 5 * 60,
    "semantic": 24 * 3600,
    "graph": 3600,
}


def k_embedding(kind: EmbeddingKind, ticker: str) -> str:
    return f"emb:{kind}:{ticker}"


def k_tick_latest(ticker: str) -> str:
    return f"tick:latest:{ticker}"


def k_news_dedupe(content_hash: str) -> str:
    return f"news:dedupe:{content_hash}"


def ch_price(ticker: str) -> str:
    return f"channel:price.{ticker}"


CH_NEWS_GLOBAL = "channel:news.global"


class RedisCache:
    """Async wrapper around redis-py with binary numpy serialization."""

    def __init__(self) -> None:
        self._client: aioredis.Redis | None = None
        self._settings = get_settings()

    async def connect(self) -> None:
        if self._client is not None:
            return
        self._client = aioredis.from_url(
            self._settings.REDIS_URL,
            max_connections=self._settings.REDIS_MAX_CONNECTIONS,
            decode_responses=False,
        )
        await self._client.ping()
        logger.info("RedisCache connected")

    async def disconnect(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    @property
    def client(self) -> aioredis.Redis:
        if self._client is None:
            raise RuntimeError("RedisCache not connected. Call connect() first.")
        return self._client

    async def set_embedding(self, kind: EmbeddingKind, ticker: str, vec: np.ndarray) -> None:
        expected_dim = _DIM_MAP[kind]
        if vec.shape != (expected_dim,):
            raise ValueError(
                f"Embedding shape mismatch for {kind}: expected ({expected_dim},), got {vec.shape}"
            )
        if vec.dtype != np.float32:
            vec = vec.astype(np.float32)
        await self.client.set(k_embedding(kind, ticker), vec.tobytes(), ex=_TTL_MAP[kind])

    async def get_embedding(self, kind: EmbeddingKind, ticker: str) -> np.ndarray | None:
        raw = await self.client.get(k_embedding(kind, ticker))
        if raw is None:
            return None
        return np.frombuffer(raw, dtype=np.float32).copy()

    async def mget_embeddings(
        self, kind: EmbeddingKind, tickers: list[str]
    ) -> dict[str, np.ndarray]:
        if not tickers:
            return {}
        keys = [k_embedding(kind, t) for t in tickers]
        raws = await self.client.mget(keys)
        out: dict[str, np.ndarray] = {}
        for ticker, raw in zip(tickers, raws, strict=True):
            if raw is not None:
                out[ticker] = np.frombuffer(raw, dtype=np.float32).copy()
        return out

    async def publish_tick(self, ticker: str, tick: dict) -> int:
        payload = json.dumps(tick, default=str).encode("utf-8")
        await self.client.set(k_tick_latest(ticker), payload, ex=300)
        return await self.client.publish(ch_price(ticker), payload)

    async def subscribe_ticks(self, tickers: list[str]) -> AsyncIterator[tuple[str, dict]]:
        pubsub = self.client.pubsub()
        channels = [ch_price(t) for t in tickers]
        await pubsub.subscribe(*channels)
        try:
            async for msg in pubsub.listen():
                if msg["type"] != "message":
                    continue
                channel = msg["channel"].decode("utf-8")
                ticker = channel.split(".", 1)[1]
                data = json.loads(msg["data"])
                yield ticker, data
        finally:
            await pubsub.unsubscribe(*channels)
            await pubsub.aclose()

    async def is_duplicate_event(self, content_hash: str, ttl: int = 86400) -> bool:
        was_set = await self.client.set(k_news_dedupe(content_hash), b"1", ex=ttl, nx=True)
        return not bool(was_set)

    async def health_check(self) -> dict:
        t0 = time.perf_counter()
        try:
            await self.client.ping()
            return {"connected": True, "latency_ms": (time.perf_counter() - t0) * 1000}
        except Exception as exc:
            logger.error(f"RedisCache health check failed: {exc}")
            return {"connected": False, "latency_ms": -1}


_cache: RedisCache | None = None


def get_redis_cache() -> RedisCache:
    global _cache
    if _cache is None:
        _cache = RedisCache()
    return _cache
