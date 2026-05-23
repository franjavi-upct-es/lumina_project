# backend/data_engine/collectors/circuit_breaker.py
"""Global circuit breaker. Collectors pause when the DLQ or buffers overflow."""

from __future__ import annotations

import asyncio

from loguru import logger

from backend.data_engine.storage.redis_cache import RedisCache

_DLQ_CRITICAL_THRESHOLD = 5000


class CircuitBreaker:
    """Shared across collectors. When OPEN, collectors must pause."""

    def __init__(self, redis: RedisCache) -> None:
        self._redis = redis
        self._open = asyncio.Event()
        self._check_interval_s = 10.0
        self._task: asyncio.Task | None = None

    async def start(self) -> None:
        self._task = asyncio.create_task(self._monitor_loop(), name="circuit_breaker")

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)

    @property
    def is_open(self) -> bool:
        return self._open.is_set()

    async def wait_if_open(self) -> None:
        while self._open.is_set():
            await asyncio.sleep(1.0)

    def trip(self, reason: str) -> None:
        if not self._open.is_set():
            logger.critical(f"Circuit breaker TRIPPED: {reason}")
            self._open.set()

    def reset(self) -> None:
        if self._open.is_set():
            logger.info("Circuit breaker RESET")
            self._open.clear()

    async def _monitor_loop(self) -> None:
        try:
            while True:
                await asyncio.sleep(self._check_interval_s)
                await self._check_dlq()
        except asyncio.CancelledError:
            raise

    async def _check_dlq(self) -> None:
        try:
            total = 0
            for dtype in ("price", "news"):
                size = await self._redis.client.llen(f"dlq:ingestion:{dtype}")
                total += int(size or 0)
            if total >= _DLQ_CRITICAL_THRESHOLD:
                self.trip(f"DLQ size {total} exceeds {_DLQ_CRITICAL_THRESHOLD}")
            elif total < _DLQ_CRITICAL_THRESHOLD // 2:
                self.reset()
        except Exception as exc:
            logger.warning(f"Circuit breaker monitor error: {exc}")
