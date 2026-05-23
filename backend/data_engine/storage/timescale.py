# backend/data_engine/storage/timescale.py
"""TimescaleDB async wrapper. Single gateway to the cold store."""

from __future__ import annotations

import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Literal
from uuid import UUID

import asyncpg
import polars as pl
from loguru import logger
from pydantic import BaseModel, Field

from backend.config.settings import get_settings

Frequency = Literal["1m", "5m", "1h", "1d"]
_BUCKET_MAP: dict[Frequency, str] = {
    "1m": "1 minute",
    "5m": "5 minutes",
    "1h": "1 hour",
    "1d": "1 day",
}


class OHLCVRow(BaseModel):
    time: datetime
    ticker: str
    open: float
    high: float
    low: float
    close: float
    volume: int
    vwap: float | None = None
    trade_count: int | None = None


class NewsEvent(BaseModel):
    time: datetime
    tickers: list[str]
    source: str
    headline: str
    body: str | None = None
    url: str | None = None
    content_hash: str
    raw: dict | None = None
    event_id: UUID | None = None


class SupplyChainEdge(BaseModel):
    source_ticker: str
    target_ticker: str
    relationship_type: str
    weight: float = Field(ge=0, le=1, default=1.0)
    valid_from: datetime
    valid_to: datetime | None = None


class TimescaleStore:
    """Async CRUD over TimescaleDB. Connection pool managed internally."""

    def __init__(self) -> None:
        self._pool: asyncpg.Pool | None = None
        self._settings = get_settings()

    async def connect(self) -> None:
        if self._pool is not None:
            return
        self._pool = await asyncpg.create_pool(
            dsn=self._settings.TIMESCALE_URL,
            min_size=self._settings.TIMESCALE_POOL_MIN,
            max_size=self._settings.TIMESCALE_POOL_MAX,
            command_timeout=30,
        )
        logger.info("TimescaleStore pool initialized")

    async def disconnect(self) -> None:
        if self._pool is not None:
            await self._pool.close()
            self._pool = None
            logger.info("TimescaleStore pool closed")

    @asynccontextmanager
    async def _conn(self) -> AsyncIterator[asyncpg.Connection]:
        if self._pool is None:
            await self.connect()
        assert self._pool is not None
        async with self._pool.acquire() as conn:
            yield conn

    async def insert_ohlcv_batch(self, rows: list[OHLCVRow]) -> int:
        if not rows:
            return 0
        records = [
            (r.time, r.ticker, r.open, r.high, r.low, r.close, r.volume, r.vwap, r.trade_count)
            for r in rows
        ]
        query = """
            INSERT INTO ohlcv_1m
                (time, ticker, open, high, low, close, volume, vwap, trade_count)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
            ON CONFLICT (ticker, time) DO NOTHING
        """
        async with self._conn() as conn:
            await conn.executemany(query, records)
        logger.debug(f"Inserted OHLCV batch: {len(records)} rows")
        return len(records)

    async def get_historical_window(
        self,
        ticker: str,
        start: datetime,
        end: datetime,
        freq: Frequency = "1m",
    ) -> pl.DataFrame:
        bucket = _BUCKET_MAP[freq]
        query = """
            SELECT
                time_bucket($1::interval, time) AS time,
                first(open, time) AS open,
                max(high) AS high,
                min(low) AS low,
                last(close, time) AS close,
                sum(volume) AS volume,
                avg(vwap) AS vwap,
                sum(trade_count) AS trade_count
            FROM ohlcv_1m
            WHERE ticker = $2 AND time >= $3 AND time < $4
            GROUP BY 1
            ORDER BY 1
        """
        async with self._conn() as conn:
            rows = await conn.fetch(query, bucket, ticker, start, end)
        if not rows:
            return pl.DataFrame(
                schema={
                    "time": pl.Datetime,
                    "open": pl.Float64,
                    "high": pl.Float64,
                    "low": pl.Float64,
                    "close": pl.Float64,
                    "volume": pl.Int64,
                    "vwap": pl.Float64,
                    "trade_count": pl.Int64,
                }
            )
        return pl.DataFrame([dict(r) for r in rows])

    async def insert_news_event(self, event: NewsEvent) -> UUID | None:
        query = """
            INSERT INTO news_events
                (time, tickers, source, headline, body, url, content_hash, raw)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            ON CONFLICT (content_hash, time) DO NOTHING
            RETURNING event_id
        """
        async with self._conn() as conn:
            row = await conn.fetchrow(
                query,
                event.time,
                event.tickers,
                event.source,
                event.headline,
                event.body,
                event.url,
                event.content_hash,
                event.raw,
            )
        return row["event_id"] if row else None

    async def query_news_by_ticker(
        self, ticker: str, since: datetime, limit: int = 100
    ) -> list[NewsEvent]:
        query = """
            SELECT time, event_id, tickers, source, headline, body, url, content_hash, raw
            FROM news_events
            WHERE $1 = ANY(tickers) AND time >= $2
            ORDER BY time DESC
            LIMIT $3
        """
        async with self._conn() as conn:
            rows = await conn.fetch(query, ticker, since, limit)
        return [NewsEvent(**dict(r)) for r in rows]

    async def insert_supply_chain_edges(self, edges: list[SupplyChainEdge]) -> int:
        if not edges:
            return 0
        records = [
            (
                e.source_ticker,
                e.target_ticker,
                e.relationship_type,
                e.weight,
                e.valid_from,
                e.valid_to,
            )
            for e in edges
        ]
        query = """
            INSERT INTO supply_chain_edges
                (source_ticker, target_ticker, relationship_type, weight, valid_from, valid_to)
            VALUES ($1, $2, $3, $4, $5, $6)
        """
        async with self._conn() as conn:
            await conn.executemany(query, records)
        return len(records)

    async def get_supply_chain_graph(
        self, as_of: datetime
    ) -> tuple[list[str], list[tuple[str, str, float]]]:
        query = """
            SELECT source_ticker, target_ticker, weight
            FROM supply_chain_edges
            WHERE valid_from <= $1 AND (valid_to IS NULL OR valid_to > $1)
        """
        async with self._conn() as conn:
            rows = await conn.fetch(query, as_of)
        edges = [(r["source_ticker"], r["target_ticker"], float(r["weight"])) for r in rows]
        nodes = sorted({n for e in edges for n in (e[0], e[1])})
        return nodes, edges

    async def health_check(self) -> dict:
        t0 = time.perf_counter()
        try:
            async with self._conn() as conn:
                count = await conn.fetchval(
                    "SELECT count(*) FROM timescaledb_information.hypertables"
                )
            return {
                "connected": True,
                "latency_ms": (time.perf_counter() - t0) * 1000,
                "hypertable_count": int(count or 0),
            }
        except Exception as exc:
            logger.error(f"TimescaleStore health check failed: {exc}")
            return {"connected": False, "latency_ms": -1, "hypertable_count": 0}


_store: TimescaleStore | None = None


def get_timescale_store() -> TimescaleStore:
    global _store
    if _store is None:
        _store = TimescaleStore()
    return _store
