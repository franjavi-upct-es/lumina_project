# backend/data_engine/collectors/yfinance_collector.py
"""Free historical data collector using yfinance — for preliminary tests.

Why this module exists
----------------------
Polygon.io Starter ($29/mo) is the production data source per the
architecture spec. But subscribing before we know the model can learn
anything is wasteful. yfinance gives us:

  * 10+ years of daily and 1-minute OHLCV
  * Zero cost, zero API key
  * Officially undocumented ⇒ rate-limit lottery, but for offline
    backfills it is perfectly adequate.

Limitations vs Polygon (be aware!)
----------------------------------
  * 1-minute granularity only available for the LAST 7 DAYS via Yahoo;
    older intraday data is at best 5-minute resolution.
    → We therefore use yfinance for *daily* bars when reaching back >7d,
      and only as a "live-ish" 1-min source for the most recent week.
  * No Level-2 / order-book data.
  * No formal SLA. Treat the return value as best-effort.
  * Adjusted prices are forward-adjusted by default; we set
    ``auto_adjust=False`` and keep raw OHLC + a separate ``adj_close``
    column so we can compute splits/dividends ourselves if needed.

Public API
----------
``YFinanceCollector.fetch_daily(ticker, start, end)``
    Return a Polars DataFrame matching the schema of ohlcv_1m
    (column names compatible) for *daily* bars.

``YFinanceCollector.fetch_intraday(ticker, days_back=5, interval="1m")``
    Recent intraday bars only (yfinance limitation).

``YFinanceCollector.backfill_to_timescale(...)``
    Dump the result of ``fetch_daily`` into the existing TimescaleDB
    ``ohlcv_1m`` hypertable, filling each daily row by repeating it as a
    *single 1-minute bar* at 16:00 UTC. This lets the rest of the
    pipeline run end-to-end on free data, at the price of losing
    intraday resolution.

For the *Phase-1 milestone* the spec says we just need to prove the data
loop works. yfinance is enough; switch to Polygon later by setting
``BROKER_MODE=polygon`` in .env.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, date

import polars as pl
import yfinance as yf
from loguru import logger

from backend.data_engine.storage.timescale import OHLCVRow, TimescaleStore


class YFinanceCollector:
    """Synchronous-by-design wrapper around yfinance.

    yfinance is not async; we offload calls to a thread executor when we
    need them inside an event loop.
    """

    @staticmethod
    def _yf_to_polars(df) -> pl.DataFrame:
        """Convert a yfinance-pandas frame to our canonical Polars schema."""
        if df.empty:
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
        df = df.reset_index()
        time_col = "Date" if "Date" in df.columns else "Datetime"
        df = df.rename(
            columns={
                time_col: "time",
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume",
            }
        )
        # yfinance returns naive timestamps for daily, tz-aware for intraday;
        # normalise to UTC.
        df["time"] = (
            df["time"].dt.tz_localize("UTC")
            if df["time"].dt.tz is None
            else df["time"].dt.tz_convert("UTC")
        )
        # vwap and trade_count are absent in yfinance — fill with None.
        df["vwap"] = None
        df["trade_count"] = None
        return pl.from_pandas(
            df[
                [
                    "time",
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                    "vwap",
                    "trade_count",
                ]
            ]
        )

    # ------------------------------------------------------------------
    @classmethod
    def fetch_daily(cls, ticker: str, start: date, end: date) -> pl.DataFrame:
        """Fetch end-of-day bars between [start, end). Returns Polars DF."""
        logger.info(f"yfinance: daily {ticker} {start} → {end}")
        raw = yf.download(
            ticker,
            start=start,
            end=end,
            interval="1d",
            auto_adjust=False,
            progress=False,
            threads=False,
        )
        return cls._yf_to_polars(raw)

    # ------------------------------------------------------------------
    @classmethod
    def fetch_intraday(
        cls,
        ticker: str,
        days_back: int = 5,
        interval: str = "1m",
    ) -> pl.DataFrame:
        """Fetch the most recent ``days_back`` days at ``interval`` resolution.

        yfinance only allows up to 7d of 1-minute data per request.
        """
        if days_back > 7:
            raise ValueError("yfinance limits 1-minute requests to 7 days.")
        logger.info(f"yfinance: intraday {ticker} last {days_back}d @ {interval}")
        raw = yf.download(
            ticker,
            period=f"{days_back}d",
            interval=interval,
            auto_adjust=False,
            progress=False,
            threads=False,
        )
        return cls._yf_to_polars(raw)

    # ------------------------------------------------------------------
    @classmethod
    async def backfill_to_timescale(
        cls,
        store: TimescaleStore,
        tickers: list[str],
        start: date,
        end: date,
        rate_limit_seconds: float = 0.5,
    ) -> int:
        """Backfill historical daily bars into the ohlcv_1m hypertable.

        Each daily bar is written as a *single* 1-minute bar at 20:00 UTC
        (NYSE close). The result is a sparse hypertable but it lets the
        rest of the pipeline run unchanged.

        Parameters
        ----------
        rate_limit_seconds : float
            Sleep between tickers. yfinance has been hostile to bursts;
            0.5 s keeps us out of trouble.

        Returns
        -------
        Total number of rows inserted.
        """
        loop = asyncio.get_running_loop()
        total = 0
        for ticker in tickers:
            df = await loop.run_in_executor(None, cls.fetch_daily, ticker, start, end)
            if df.is_empty():
                logger.warning(f"yfinance returned no data for {ticker}")
                continue
            rows: list[OHLCVRow] = []
            for r in df.iter_rows(named=True):
                t = r["time"]
                # Anchor every daily bar at 20:00 UTC = NYSE close.
                anchored = t.replace(hour=20, minute=0, second=0, microsecond=0, tzinfo=UTC)
                rows.append(
                    OHLCVRow(
                        time=anchored,
                        ticker=ticker,
                        open=float(r["open"]),
                        high=float(r["high"]),
                        low=float(r["low"]),
                        close=float(r["close"]),
                        volume=int(r["volume"] or 0),
                        vwap=None,
                        trade_count=None,
                    )
                )
            inserted = await store.insert_ohlcv_batch(rows)
            total += inserted
            logger.success(f"{ticker}: backfilled {inserted} daily bars")
            await asyncio.sleep(rate_limit_seconds)
        return total
