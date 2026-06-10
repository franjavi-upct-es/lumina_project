"""Deterministic local market/news feed for full-stack smoke tests."""

from __future__ import annotations

import asyncio
import json
import signal
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

import numpy as np
from loguru import logger

from backend.config.settings import Settings, get_settings
from backend.data_engine.storage.redis_cache import RedisCache
from backend.data_engine.storage.timescale import OHLCVRow, TimescaleStore


@dataclass(slots=True)
class SyntheticFeedService:
    redis: RedisCache
    timescale: TimescaleStore
    settings: Settings
    _rng: np.random.Generator = field(init=False)
    _prices: dict[str, float] = field(init=False)
    _running: bool = field(init=False, default=False)
    _last_news_at: datetime = field(init=False)

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.settings.SYNTHETIC_FEED_SEED)
        self._prices = {
            ticker: 100.0 + 20.0 * i for i, ticker in enumerate(self.settings.LIVE_TICKERS)
        }
        self._running = False
        self._last_news_at = datetime.min.replace(tzinfo=UTC)

    async def bootstrap_history(self) -> int:
        """Insert daily synthetic OHLCV rows for graph/bootstrap readiness."""
        rows: list[OHLCVRow] = []
        end = datetime.now(UTC).replace(hour=21, minute=0, second=0, microsecond=0)
        start = end - timedelta(days=self.settings.SYNTHETIC_FEED_BOOTSTRAP_DAYS)
        for ticker_idx, ticker in enumerate(self.settings.LIVE_TICKERS):
            price = 100.0 + 20.0 * ticker_idx
            for day in range(self.settings.SYNTHETIC_FEED_BOOTSTRAP_DAYS):
                ts = start + timedelta(days=day)
                drift = 0.0003 + ticker_idx * 0.00002
                shock = float(self._rng.normal(0.0, 0.012))
                close = max(1.0, price * (1.0 + drift + shock))
                high = max(price, close) * (1.0 + abs(float(self._rng.normal(0.0, 0.003))))
                low = min(price, close) * (1.0 - abs(float(self._rng.normal(0.0, 0.003))))
                volume = int(500_000 + self._rng.integers(0, 250_000))
                rows.append(
                    OHLCVRow(
                        time=ts,
                        ticker=ticker,
                        open=price,
                        high=high,
                        low=low,
                        close=close,
                        volume=volume,
                        vwap=(price + close) * 0.5,
                        trade_count=int(volume / 100),
                    )
                )
                price = close
            self._prices[ticker] = price
        inserted = await self.timescale.insert_ohlcv_batch(rows)
        logger.info("Synthetic feed bootstrapped {} OHLCV rows", inserted)
        return inserted

    async def run(self) -> None:
        self._running = True
        await self.bootstrap_history()
        await self._publish_news()
        while self._running:
            now = datetime.now(UTC)
            await self._publish_ticks(now)
            if (now - self._last_news_at).total_seconds() >= (
                self.settings.SYNTHETIC_FEED_NEWS_INTERVAL_SECONDS
            ):
                await self._publish_news()
            await asyncio.sleep(self.settings.SYNTHETIC_FEED_TICK_INTERVAL_SECONDS)

    async def stop(self) -> None:
        self._running = False

    async def _publish_ticks(self, now: datetime) -> None:
        market_shock = float(self._rng.normal(0.0, 0.0015))
        timestamp_ms = int(now.timestamp() * 1000)
        for ticker_idx, ticker in enumerate(self.settings.LIVE_TICKERS):
            open_price = self._prices[ticker]
            idiosyncratic = float(self._rng.normal(0.0, 0.002))
            close = max(1.0, open_price * (1.0 + market_shock + idiosyncratic))
            high = max(open_price, close) * (1.0 + abs(float(self._rng.normal(0.0, 0.0008))))
            low = min(open_price, close) * (1.0 - abs(float(self._rng.normal(0.0, 0.0008))))
            volume = int(25_000 + self._rng.integers(0, 10_000) + ticker_idx * 100)
            tick = {
                "t": timestamp_ms,
                "o": open_price,
                "h": high,
                "l": low,
                "c": close,
                "v": volume,
                "vw": (open_price + close) * 0.5,
                "n": max(1, int(volume / 100)),
            }
            await self.redis.publish_tick(ticker, tick)
            self._prices[ticker] = close

    async def _publish_news(self) -> None:
        now = datetime.now(UTC)
        tickers = list(self.settings.LIVE_TICKERS[: min(3, len(self.settings.LIVE_TICKERS))])
        payload = {
            "headline": "Synthetic market update: broad risk appetite remains orderly",
            "body": "Deterministic local feed event for Lumina semantic readiness.",
            "source": "synthetic",
            "url": None,
            "published_at": now.isoformat(),
            "tickers": tickers,
        }
        await self.redis.client.publish("channel:news.global", json.dumps(payload, default=str))
        self._last_news_at = now


async def _amain() -> None:
    from backend.config.logging import configure_logging

    configure_logging()
    settings = get_settings()
    redis = RedisCache()
    timescale = TimescaleStore()
    await redis.connect()
    await timescale.connect()
    service = SyntheticFeedService(redis, timescale, settings)
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, lambda: asyncio.create_task(service.stop()))
    try:
        await service.run()
    finally:
        await redis.disconnect()
        await timescale.disconnect()


def main() -> int:
    try:
        asyncio.run(_amain())
    except Exception:
        logger.exception("SyntheticFeedService crashed")
        return 1
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
