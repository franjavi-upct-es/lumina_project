# backend/data_engine/pipelines/ingestion.py
"""Async ingestion pipeline: Redis pub/sub -> cleaning -> TimescaleDB."""

from __future__ import annotations

import asyncio
import contextlib
import json
import signal
import time
from datetime import UTC, datetime
from typing import Any

from loguru import logger
from tenacity import AsyncRetrying, retry_if_exception_type, stop_after_attempt, wait_exponential

from backend.config.constants import TARGET_TICKERS
from backend.config.settings import get_settings
from backend.data_engine.pipelines.cleaning import (
    clean_news_text,
    compute_content_hash,
    dedupe_news_events,
)
from backend.data_engine.pipelines.metrics import (
    INGESTION_BACKPRESSURE,
    INGESTION_BUFFER_SIZE,
    INGESTION_ERRORS,
    INGESTION_LAG,
    INGESTION_THROUGHPUT,
)
from backend.data_engine.storage.redis_cache import RedisCache
from backend.data_engine.storage.timescale import NewsEvent, OHLCVRow, TimescaleStore

_DLQ_KEY_PREFIX = "dlq:ingestion"


class IngestionPipeline:
    """Orchestrates real-time ingestion from Redis into TimescaleDB."""

    def __init__(
        self, timescale: TimescaleStore, redis: RedisCache, tickers: list[str] | None = None
    ) -> None:
        self._ts = timescale
        self._redis = redis
        self._settings = get_settings()
        self._tickers = tickers or list(TARGET_TICKERS)
        self._batch_size = self._settings.INGESTION_BATCH_SIZE
        self._flush_interval = self._settings.INGESTION_FLUSH_INTERVAL_S
        self._backpressure_threshold = int(
            self._batch_size * self._settings.INGESTION_BACKPRESSURE_FACTOR
        )
        self._price_buffer: list[OHLCVRow] = []
        self._news_buffer: list[NewsEvent] = []
        self._buffer_lock = asyncio.Lock()
        self._running = False
        self._tasks: list[asyncio.Task] = []
        self._shutdown_event = asyncio.Event()
        self._price_paused = False
        self._news_paused = False

    async def start(self) -> None:
        if self._running:
            logger.warning("IngestionPipeline already running")
            return
        self._running = True
        self._shutdown_event.clear()
        await self._ts.connect()
        await self._redis.connect()
        self._tasks = [
            asyncio.create_task(self._consume_price_stream(), name="consume_price"),
            asyncio.create_task(self._consume_news_stream(), name="consume_news"),
            asyncio.create_task(self._flush_loop(), name="flush_loop"),
        ]
        self._install_signal_handlers()
        logger.info(f"IngestionPipeline started (batch_size={self._batch_size})")

    async def stop(self, drain_timeout_s: float = 10.0) -> None:
        if not self._running:
            return
        logger.info("IngestionPipeline stopping (graceful drain)...")
        self._running = False
        self._shutdown_event.set()
        for t in self._tasks:
            t.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        try:
            await asyncio.wait_for(self._drain_all_buffers(), timeout=drain_timeout_s)
        except TimeoutError:
            logger.error("Drain timed out; some data may be lost")
        logger.info("IngestionPipeline stopped")

    def _install_signal_handlers(self) -> None:
        try:
            loop = asyncio.get_running_loop()
            for sig in (signal.SIGTERM, signal.SIGINT):
                loop.add_signal_handler(sig, lambda: asyncio.create_task(self.stop()))
        except (NotImplementedError, RuntimeError):
            pass

    async def _consume_price_stream(self) -> None:
        try:
            async for ticker, tick in self._redis.subscribe_ticks(self._tickers):
                if not self._running:
                    break
                if len(self._price_buffer) >= self._backpressure_threshold:
                    if not self._price_paused:
                        INGESTION_BACKPRESSURE.labels(data_type="price").inc()
                        logger.warning(f"Price backpressure: buffer={len(self._price_buffer)}")
                        self._price_paused = True
                    await asyncio.sleep(0.1)
                    continue
                self._price_paused = False
                try:
                    row = self._tick_to_ohlcv(ticker, tick)
                    async with self._buffer_lock:
                        self._price_buffer.append(row)
                    INGESTION_BUFFER_SIZE.labels(data_type="price").set(len(self._price_buffer))
                except Exception as exc:
                    INGESTION_ERRORS.labels(data_type="price", stage="parse").inc()
                    logger.error(f"Failed to parse tick for {ticker}: {exc}")
        except asyncio.CancelledError:
            raise

    async def _consume_news_stream(self) -> None:
        pubsub = self._redis.client.pubsub()
        await pubsub.subscribe("channel:news.global")
        try:
            async for msg in pubsub.listen():
                if not self._running:
                    break
                if msg["type"] != "message":
                    continue
                if len(self._news_buffer) >= self._backpressure_threshold:
                    if not self._news_paused:
                        INGESTION_BACKPRESSURE.labels(data_type="news").inc()
                        self._news_paused = True
                    await asyncio.sleep(0.1)
                    continue
                self._news_paused = False
                try:
                    raw = json.loads(msg["data"])
                    event = self._raw_to_news_event(raw)
                    if event is None:
                        continue
                    async with self._buffer_lock:
                        self._news_buffer.append(event)
                    INGESTION_BUFFER_SIZE.labels(data_type="news").set(len(self._news_buffer))
                except Exception as exc:
                    INGESTION_ERRORS.labels(data_type="news", stage="parse").inc()
                    logger.error(f"Failed to parse news: {exc}")
        except asyncio.CancelledError:
            raise
        finally:
            await pubsub.unsubscribe()
            await pubsub.aclose()

    async def _flush_loop(self) -> None:
        try:
            while self._running:
                with contextlib.suppress(TimeoutError):
                    await asyncio.wait_for(
                        self._shutdown_event.wait(), timeout=self._flush_interval
                    )
                await self._flush_price_buffer()
                await self._flush_news_buffer()
        except asyncio.CancelledError:
            raise

    async def _drain_all_buffers(self) -> None:
        await self._flush_price_buffer(force=True)
        await self._flush_news_buffer(force=True)

    async def _flush_price_buffer(self, force: bool = False) -> None:
        async with self._buffer_lock:
            if not self._price_buffer:
                return
            batch = self._price_buffer
            self._price_buffer = []
        await self._write_with_retry(batch, self._ts.insert_ohlcv_batch, data_type="price")

    async def _flush_news_buffer(self, force: bool = False) -> None:
        async with self._buffer_lock:
            if not self._news_buffer:
                return
            batch = dedupe_news_events(self._news_buffer)
            self._news_buffer = []
        success = 0
        for event in batch:
            try:
                await self._insert_news_with_retry(event)
                success += 1
            except Exception:
                await self._push_to_dlq("news", event.model_dump(mode="json"))
                INGESTION_ERRORS.labels(data_type="news", stage="write").inc()
        INGESTION_THROUGHPUT.labels(data_type="news").inc(success)
        INGESTION_BUFFER_SIZE.labels(data_type="news").set(len(self._news_buffer))

    async def _write_with_retry(self, batch: list, writer, data_type: str) -> None:
        if not batch:
            return
        t0 = time.perf_counter()
        try:
            async for attempt in AsyncRetrying(
                stop=stop_after_attempt(self._settings.INGESTION_MAX_RETRIES),
                wait=wait_exponential(multiplier=0.5, min=0.5, max=10),
                retry=retry_if_exception_type(Exception),
                reraise=True,
            ):
                with attempt:
                    n = await writer(batch)
            INGESTION_THROUGHPUT.labels(data_type=data_type).inc(n)
            INGESTION_LAG.labels(data_type=data_type).observe(time.perf_counter() - t0)
            INGESTION_BUFFER_SIZE.labels(data_type=data_type).set(0)
        except Exception as exc:
            logger.error(f"[{data_type}] write failed, DLQ'ing {len(batch)}: {exc}")
            INGESTION_ERRORS.labels(data_type=data_type, stage="write").inc()
            for item in batch:
                await self._push_to_dlq(
                    data_type,
                    item.model_dump(mode="json") if hasattr(item, "model_dump") else item,
                )

    async def _insert_news_with_retry(self, event: NewsEvent) -> None:
        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(self._settings.INGESTION_MAX_RETRIES),
            wait=wait_exponential(multiplier=0.5, min=0.5, max=10),
            reraise=True,
        ):
            with attempt:
                await self._ts.insert_news_event(event)

    @staticmethod
    def _tick_to_ohlcv(ticker: str, tick: dict[str, Any]) -> OHLCVRow:
        raw_ts: Any = tick.get("t") or tick.get("time")
        if raw_ts is None:
            raise ValueError(f"Tick missing 't' / 'time' field: {tick!r}")
        if isinstance(raw_ts, int | float):
            ts = datetime.fromtimestamp(raw_ts / 1000.0, tz=UTC)
        elif isinstance(raw_ts, str):
            ts = datetime.fromisoformat(raw_ts.replace("Z", "+00:00"))
        elif isinstance(raw_ts, datetime):
            ts = raw_ts
        else:
            raise TypeError(f"Unsupported tick timestamp type: {type(raw_ts).__name__}")
        return OHLCVRow(
            time=ts,
            ticker=ticker,
            open=float(tick["o"]),
            high=float(tick["h"]),
            low=float(tick["l"]),
            close=float(tick["c"]),
            volume=int(tick["v"]),
            vwap=float(tick["vw"]) if tick.get("vw") else None,
            trade_count=int(tick["n"]) if tick.get("n") else None,
        )

    def _raw_to_news_event(self, raw: dict[str, Any]) -> NewsEvent | None:
        headline = clean_news_text(raw.get("headline") or raw.get("title") or "")
        body = clean_news_text(raw.get("body") or raw.get("description"))
        source = raw.get("source", "unknown")
        if not headline:
            return None
        content_hash = compute_content_hash(source, headline, body)
        tickers_raw = raw.get("tickers") or []
        tickers = [t for t in tickers_raw if t in TARGET_TICKERS]
        if not tickers:
            return None
        ts = raw.get("published_at") or raw.get("time")
        if isinstance(ts, str):
            ts = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        elif ts is None:
            ts = datetime.now(UTC)
        return NewsEvent(
            time=ts,
            tickers=tickers,
            source=source,
            headline=headline,
            body=body,
            url=raw.get("url"),
            content_hash=content_hash,
            raw=raw,
        )

    async def _push_to_dlq(self, data_type: str, payload: dict) -> None:
        key = f"{_DLQ_KEY_PREFIX}:{data_type}"
        try:
            await self._redis.client.lpush(key, json.dumps(payload, default=str))
            await self._redis.client.ltrim(key, 0, 9999)
        except Exception as exc:
            logger.critical(f"Cannot push to DLQ ({data_type}): {exc}")

    async def health(self) -> dict:
        return {
            "running": self._running,
            "price_buffer": len(self._price_buffer),
            "news_buffer": len(self._news_buffer),
            "price_paused": self._price_paused,
            "news_paused": self._news_paused,
            "tickers_subscribed": len(self._tickers),
        }


async def _amain() -> None:
    redis = RedisCache()
    timescale = TimescaleStore()
    pipeline = IngestionPipeline(timescale, redis)
    await pipeline.start()
    try:
        await pipeline._shutdown_event.wait()
    finally:
        await pipeline.stop()
        await redis.disconnect()
        await timescale.disconnect()


def main() -> int:
    from backend.config.logging import configure_logging

    configure_logging()
    try:
        asyncio.run(_amain())
    except Exception:
        logger.exception("Ingestion pipeline crashed")
        return 1
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
