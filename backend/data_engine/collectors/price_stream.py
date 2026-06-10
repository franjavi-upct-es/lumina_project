# backend/data_engine/collectors/price_stream.py
"""Polygon.io price stream collector (WebSocket + REST backfill)."""

from __future__ import annotations

import asyncio
import contextlib
import json
import random
import signal
from datetime import UTC, date, datetime, timedelta
from typing import Any

import httpx
import websockets
from loguru import logger
from prometheus_client import Counter
from tenacity import AsyncRetrying, stop_after_attempt, wait_exponential

from backend.config.settings import get_settings
from backend.data_engine.collectors.circuit_breaker import CircuitBreaker
from backend.data_engine.storage.redis_cache import RedisCache
from backend.data_engine.storage.timescale import OHLCVRow, TimescaleStore

PRICE_STREAM_MESSAGES = Counter(
    "price_stream_messages_total",
    "Total tick messages from Polygon",
    labelnames=("ticker",),
)
PRICE_STREAM_RECONNECTS = Counter("price_stream_reconnects_total", "Reconnection attempts")


class PolygonPriceStream:
    def __init__(self, redis: RedisCache, circuit_breaker: CircuitBreaker | None = None) -> None:
        self._redis = redis
        self._cb = circuit_breaker
        self._settings = get_settings()
        # The concrete websockets connection class has moved across versions
        # (WebSocketClientProtocol in <15, ClientConnection in >=15). We
        # only ever use it as a context-managed object so Any is safe.
        self._ws: Any | None = None
        self._subscribed: set[str] = set()
        self._running = False
        self._last_message_at: float = 0.0

    async def run(self, tickers: list[str]) -> None:
        self._running = True
        self._subscribed = set(tickers)
        backoff_s = 1.0
        while self._running:
            if self._cb and self._cb.is_open:
                logger.warning("Price stream paused by circuit breaker")
                await self._cb.wait_if_open()
            try:
                await self._connect_and_stream()
                backoff_s = 1.0
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                PRICE_STREAM_RECONNECTS.inc()
                jitter = random.uniform(0, 0.3 * backoff_s)
                wait = min(backoff_s + jitter, self._settings.COLLECTOR_MAX_BACKOFF_S)
                logger.error(f"Price WS error: {exc}. Reconnecting in {wait:.1f}s")
                await asyncio.sleep(wait)
                backoff_s = min(backoff_s * 2, self._settings.COLLECTOR_MAX_BACKOFF_S)
        await self._close()

    async def stop(self) -> None:
        self._running = False
        await self._close()

    async def _close(self) -> None:
        if self._ws:
            with contextlib.suppress(Exception):
                await self._ws.close()
            self._ws = None

    async def _connect_and_stream(self) -> None:
        url = self._settings.POLYGON_WS_URL
        async with websockets.connect(url, ping_interval=20, ping_timeout=20) as ws:
            self._ws = ws
            await ws.send(json.dumps({"action": "auth", "params": self._settings.POLYGON_API_KEY}))
            auth_resp = json.loads(await ws.recv())
            if not self._is_auth_success(auth_resp):
                raise RuntimeError(f"Polygon auth failed: {auth_resp}")
            params = ",".join(f"AM.{t}" for t in self._subscribed)
            await ws.send(json.dumps({"action": "subscribe", "params": params}))
            logger.info(f"Subscribed to {len(self._subscribed)} tickers")
            hb_task = asyncio.create_task(self._heartbeat_loop())
            try:
                async for raw in ws:
                    self._last_message_at = asyncio.get_running_loop().time()
                    await self._handle_message(raw)
            finally:
                hb_task.cancel()
                await asyncio.gather(hb_task, return_exceptions=True)

    @staticmethod
    def _is_auth_success(resp) -> bool:
        if isinstance(resp, list):
            return any(m.get("status") == "auth_success" for m in resp)
        return resp.get("status") == "auth_success"

    async def _heartbeat_loop(self) -> None:
        self._last_message_at = asyncio.get_running_loop().time()
        while True:
            await asyncio.sleep(self._settings.COLLECTOR_HEARTBEAT_S / 2)
            now = asyncio.get_running_loop().time()
            if now - self._last_message_at > self._settings.COLLECTOR_HEARTBEAT_S:
                logger.warning("Heartbeat timeout — forcing reconnect")
                if self._ws:
                    await self._ws.close()
                return

    async def _handle_message(self, raw) -> None:
        try:
            msgs = json.loads(raw)
            if not isinstance(msgs, list):
                msgs = [msgs]
            for m in msgs:
                if m.get("ev") == "AM":
                    await self._publish_aggregate(m)
        except Exception as exc:
            logger.error(f"Failed to handle message: {exc}")

    async def _publish_aggregate(self, m: dict) -> None:
        ticker = m.get("sym")
        if not ticker:
            return
        tick = {
            "t": m.get("s"),
            "o": m.get("o"),
            "h": m.get("h"),
            "l": m.get("l"),
            "c": m.get("c"),
            "v": m.get("v"),
            "vw": m.get("vw"),
            "n": m.get("n"),
        }
        if None in (tick["t"], tick["o"], tick["h"], tick["l"], tick["c"], tick["v"]):
            return
        await self._redis.publish_tick(ticker, tick)
        PRICE_STREAM_MESSAGES.labels(ticker=ticker).inc()

    async def backfill_historical(
        self,
        ticker: str,
        start: date,
        end: date,
        store: TimescaleStore,
        chunk_days: int = 30,
    ) -> int:
        """Backfill 1-minute bars from Polygon REST into TimescaleDB."""
        total = 0
        cursor = start
        async with httpx.AsyncClient(timeout=30.0) as client:
            while cursor < end:
                chunk_end = min(cursor + timedelta(days=chunk_days), end)
                rows = await self._fetch_chunk(client, ticker, cursor, chunk_end)
                if rows:
                    inserted = await store.insert_ohlcv_batch(rows)
                    total += inserted
                    logger.info(f"[{ticker}] backfilled {inserted} rows {cursor} → {chunk_end}")
                cursor = chunk_end
                await asyncio.sleep(12.5)
        return total

    async def _fetch_chunk(
        self,
        client: httpx.AsyncClient,
        ticker: str,
        start: date,
        end: date,
    ) -> list[OHLCVRow]:
        url = (
            f"{self._settings.POLYGON_REST_URL}/v2/aggs/ticker/{ticker}"
            f"/range/1/minute/{start.isoformat()}/{end.isoformat()}"
        )
        params = {
            "adjusted": "true",
            "sort": "asc",
            "limit": 50000,
            "apiKey": self._settings.POLYGON_API_KEY,
        }
        rows: list[OHLCVRow] = []
        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=2, max=30),
            reraise=True,
        ):
            with attempt:
                resp = await client.get(url, params=params)
                resp.raise_for_status()
                data = resp.json()
        for bar in data.get("results") or []:
            try:
                rows.append(
                    OHLCVRow(
                        time=datetime.fromtimestamp(bar["t"] / 1000, tz=UTC),
                        ticker=ticker,
                        open=bar["o"],
                        high=bar["h"],
                        low=bar["l"],
                        close=bar["c"],
                        volume=bar["v"],
                        vwap=bar.get("vw"),
                        trade_count=bar.get("n"),
                    )
                )
            except (KeyError, TypeError) as exc:
                logger.warning(f"Skipped malformed bar: {exc}")
        return rows


async def _amain() -> None:
    from backend.config.logging import configure_logging

    configure_logging()
    settings = get_settings()
    if not settings.POLYGON_API_KEY:
        raise RuntimeError("POLYGON_API_KEY is required for LUMINA_SERVICE_MODE=price_stream")

    redis = RedisCache()
    await redis.connect()
    circuit_breaker = CircuitBreaker(redis)
    await circuit_breaker.start()
    stream = PolygonPriceStream(redis, circuit_breaker=circuit_breaker)
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, lambda: asyncio.create_task(stream.stop()))
    try:
        await stream.run(settings.LIVE_TICKERS)
    finally:
        await circuit_breaker.stop()
        await redis.disconnect()


def main() -> int:
    try:
        asyncio.run(_amain())
    except Exception:
        logger.exception("PolygonPriceStream crashed")
        return 1
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
