# scripts/backfill_historical.py
"""CLI: backfill OHLCV history into TimescaleDB.

Two data sources are supported:

* ``yfinance`` — free, daily resolution, useful for prototyping and the
  Phase-0/1 milestone. 1-minute resolution is also available but only
  for the most recent 7 days (Yahoo limitation).
* ``polygon`` — production source, 1-minute resolution going back 10
  years on the Starter plan. Subject to the published rate limit
  (5 req/min on Starter); the Polygon collector already enforces the
  required spacing.

Usage
-----
    python -m scripts.backfill_historical --source yfinance \
        --start 2018-01-01 --end 2024-12-31

    python -m scripts.backfill_historical --source polygon \
        --tickers AAPL,MSFT --start 2023-01-01 --end 2024-01-01

Exit codes
----------
0 — backfill completed (some tickers may have returned no data, but
    the process finished without errors)
1 — environment error (missing API key, DB connection refused, etc.)
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from datetime import date, datetime
from urllib.parse import urlsplit, urlunsplit

from loguru import logger

from backend.config.constants import TARGET_TICKERS
from backend.config.logging import configure_logging
from backend.config.settings import get_settings
from backend.data_engine.collectors.yfinance_collector import YFinanceCollector
from backend.data_engine.storage.timescale import get_timescale_store


def _mask_url(url: str) -> str:
    parts = urlsplit(url)
    if "@" not in parts.netloc:
        return url
    userinfo, hostinfo = parts.netloc.rsplit("@", 1)
    username = userinfo.split(":", 1)[0]
    return urlunsplit((parts.scheme, f"{username}:***@{hostinfo}", parts.path, parts.query, ""))


def _connection_hint(exc: BaseException) -> str:
    settings = get_settings()
    timescale_url = _mask_url(settings.TIMESCALE_URL)
    host = urlsplit(settings.TIMESCALE_URL).hostname
    hint = f"Cannot connect to TimescaleDB at {timescale_url}: {exc}"
    if host == "timescale":
        hint += (
            "\nThe hostname 'timescale' only resolves inside Docker Compose. "
            "For host-side commands, use "
            "TIMESCALE_URL=postgresql://lumina:lumina@localhost:5432/lumina "
            "or run `make backfill-yfinance`, which sets that override."
        )
    else:
        hint += (
            "\nMake sure the database is running, for example: `docker compose up -d timescale`."
        )
    return hint


async def _backfill_yfinance(tickers: list[str], start: date, end: date) -> int:
    store = get_timescale_store()
    await store.connect()
    try:
        return await YFinanceCollector.backfill_to_timescale(
            store=store,
            tickers=tickers,
            start=start,
            end=end,
        )
    finally:
        await store.disconnect()


async def _backfill_polygon(tickers: list[str], start: date, end: date) -> int:
    """Polygon backfill via the price-stream collector.

    The :class:`PolygonPriceStream.backfill_historical` method already
    handles the API rate limit and chunked retrieval; we delegate to it
    here so this script remains a thin orchestration layer.
    """
    from backend.data_engine.collectors.price_stream import PolygonPriceStream
    from backend.data_engine.storage.redis_cache import get_redis_cache

    settings = get_settings()
    if not settings.POLYGON_API_KEY:
        raise RuntimeError("POLYGON_API_KEY is not set; cannot run Polygon backfill.")

    store = get_timescale_store()
    await store.connect()
    redis = get_redis_cache()
    await redis.connect()
    stream = PolygonPriceStream(redis=redis)
    total = 0
    try:
        for ticker in tickers:
            total += await stream.backfill_historical(
                ticker=ticker,
                start=start,
                end=end,
                store=store,
            )
    finally:
        await store.disconnect()
        await redis.disconnect()
    return total


def main(argv: list[str] | None = None) -> int:
    configure_logging()
    parser = argparse.ArgumentParser(description="Backfill OHLCV into TimescaleDB.")
    parser.add_argument("--source", choices=["yfinance", "polygon"], default="yfinance")
    parser.add_argument("--start", type=lambda s: datetime.fromisoformat(s).date(), required=True)
    parser.add_argument("--end", type=lambda s: datetime.fromisoformat(s).date(), required=True)
    parser.add_argument(
        "--tickers",
        default="",
        help="Comma-separated overrides; default = TARGET_TICKERS.",
    )
    args = parser.parse_args(argv)

    tickers = (
        [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
        if args.tickers
        else sorted(TARGET_TICKERS)
    )
    logger.info(
        f"Backfilling {len(tickers)} tickers from {args.source}: {args.start} → {args.end}",
    )

    runner = _backfill_yfinance if args.source == "yfinance" else _backfill_polygon
    try:
        inserted = asyncio.run(runner(tickers, args.start, args.end))
    except (RuntimeError, OSError) as exc:
        logger.error(_connection_hint(exc) if isinstance(exc, OSError) else str(exc))
        return 1
    logger.success(f"Backfill complete: {inserted} rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
