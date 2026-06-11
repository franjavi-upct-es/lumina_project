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

import pandas as pd
import polars as pl
import yfinance as yf
from loguru import logger

from backend.data_engine.storage.timescale import OHLCVRow, TimescaleStore


class YFinanceCollector:
    """Synchronous-by-design wrapper around yfinance.

    yfinance is not async; we offload calls to a thread executor when we
    need them inside an event loop.
    """

<<<<<<< HEAD
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
        # yfinance 0.2.40+ returns a MultiIndex for columns even with a single ticker.
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
=======
    def __init__(self, rate_limit: int = 2000):
        super().__init__(name="YFinance", rate_limit=rate_limit)
        self._cache = {}  # type: ignore
>>>>>>> 994b45ea5c7f16817f4caea4d941fa54c203899e

        df = df.reset_index()
        # yfinance usually uses 'Date' (daily) or 'Datetime' (intraday).
        # After reset_index(), if the index had no name, it becomes 'index'.
        time_col = None
        for candidate in ["Date", "Datetime", "index"]:
            if candidate in df.columns:
                time_col = candidate
                break

        if time_col is None:
            # Fallback: take the first column if we can't find it by name
            time_col = df.columns[0]

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
            ticker_total = 0
            for r in df.iter_rows(named=True):
                t = r["time"]
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
                if len(rows) >= 1000:
                    ticker_total += await store.insert_ohlcv_batch(rows)
                    rows = []

            if rows:
                ticker_total += await store.insert_ohlcv_batch(rows)

<<<<<<< HEAD
            total += ticker_total
            logger.success(f"{ticker}: backfilled {ticker_total} daily bars")
            await asyncio.sleep(rate_limit_seconds)
        return total
=======
        if missing_columns:
            logger.error(f"Missing required columns: {missing_columns}")
            return False

        # Check for null values in critical columns
        null_counts = data.select(
            [pl.col(col).is_null().sum().alias(col) for col in ["close", "volume"]]
        )

        if any(null_counts.row(0)):
            logger.warning("Found null values in critical columns")

        # Validate price consistency (high >= low, etc.)
        invalid_prices = data.filter(
            (pl.col("high") < pl.col("low")) | (pl.col("close") < 0) | (pl.col("volume") < 0)
        )

        if invalid_prices.height > 0:
            logger.error(f"Found {invalid_prices.height} rows with invalid prices")
            return False

        return True

    async def get_company_info(self, ticker: str) -> dict[str, Any] | None:
        """
        Get company information and meta_data

        Args:
            ticker: Stock ticker symbol

        Returns:
            Dictionary with company information
        """
        try:
            loop = asyncio.get_event_loop()
            info = await loop.run_in_executor(None, self._fetch_company_info, ticker)
            return info
        except Exception as e:
            logger.error(f"Error fetching company info for {ticker}: {e}")
            return None

    def _fetch_company_info(self, ticker: str) -> dict[str, Any]:
        """
        Synchronous fetch of company info
        """
        stock = yf.Ticker(ticker)
        info = stock.info

        # Extract relevant fileds
        return {
            "ticker": ticker,
            "name": info.get("longName"),
            "sector": info.get("sector"),
            "industry": info.get("industry"),
            "market_cap": info.get("marketCap"),
            "country": info.get("country"),
            "currency": info.get("currency"),
            "exchange": info.get("exchange"),
            "website": info.get("website"),
            "description": info.get("longBusinessSummary"),
            # Fundamental metrics
            "pe_ratio": info.get("trailingPE"),
            "forward_pe": info.get("forwardPE"),
            "peg_ratio": info.get("pegRatio"),
            "price_to_book": info.get("priceToBook"),
            "dividend_yield": info.get("dividendYield"),
            "beta": info.get("beta"),
            # Profitability
            "profit_margin": info.get("profitMargins"),
            "operating_margin": info.get("operatingMargins"),
            "roe": info.get("returnOnEquity"),
            "roa": info.get("returnOnAssets"),
            # Financial health
            "debt_to_equity": info.get("debtToEquity"),
            "current_ratio": info.get("currentRatio"),
            "quick_ratio": info.get("quickRatio"),
            # Growth
            "revenue_growth": info.get("revenueGrowth"),
            "earnings_growth": info.get("earningsGrowth"),
            # Analyst recommendations
            "recommendation": info.get("recommendationKey"),
            "target_high_price": info.get("targetHighPrice"),
            "target_low_price": info.get("targetLowPrice"),
            "target_mean_price": info.get("targetMeanPrice"),
        }

    async def get_options_data(
        self, ticker: str, expiration_date: str | None = None
    ) -> dict[str, pl.DataFrame] | None:
        """
        Get options chain data

        Args:
            ticker: Stock ticker symbol
            expiration_date: Option expiration date (YYYY-MM-DD)

        Returns:
            Dictionary with 'calls' and 'puts' DataFrames
        """
        try:
            loop = asyncio.get_event_loop()
            options_data = await loop.run_in_executor(
                None, self._fetch_options_data, ticker, expiration_date
            )
            return options_data
        except Exception as e:
            logger.error(f"Error fetching options for {ticker}: {e}")
            return None

    def _fetch_options_data(
        self, ticker: str, expiration_date: str | None
    ) -> dict[str, pl.DataFrame] | None:
        """
        Synchronous fetch of options data
        """
        stock = yf.Ticker(ticker)

        # Get available expiration dates if not specified
        if expiration_date is None:
            expirations = stock.options
            if not expirations:
                return None
            expiration_date = expirations[0]  # Use nearest expiration

        # Get options chain
        opt = stock.option_chain(expiration_date)

        # Convert to Polars
        calls = pl.from_pandas(opt.calls)
        puts = pl.from_pandas(opt.puts)

        return {"calls": calls, "puts": puts, "expiration": expiration_date}  # type: ignore

    async def get_institutional_holders(self, ticker: str) -> pl.DataFrame | None:
        """
        Get institutional holders information
        """
        try:
            loop = asyncio.get_event_loop()
            data = await loop.run_in_executor(None, self._fetch_institutional_holders, ticker)
            return data
        except Exception as e:
            logger.error(f"Error fetching institutional holders for {ticker}: {e}")
            return None

    def _fetch_institutional_holders(self, ticker: str) -> pl.DataFrame | None:
        """
        Synchronous fetch of institutional holders
        """
        stock = yf.Ticker(ticker)
        holders = stock.institutional_holders

        if holders is None or holders.empty:
            return None

        return pl.from_pandas(holders)

    async def get_earnings_history(self, ticker: str) -> pl.DataFrame | None:
        """
        Get historical earnings data
        """
        try:
            loop = asyncio.get_event_loop()
            data = await loop.run_in_executor(None, self._fetch_earnings, ticker)
            return data
        except Exception as e:
            logger.error(f"Error fetching earnings for {ticker}: {e}")
            return None

    def _fetch_earnings(self, ticker: str) -> pl.DataFrame | None:
        """
        Synchronous fetch of earnings
        """
        stock = yf.Ticker(ticker)
        earnings = stock.earnings_history

        if earnings is None or earnings.empty:
            return None

        df = pl.from_pandas(earnings.reset_index())
        return self._standardize_columns(df)
>>>>>>> 994b45ea5c7f16817f4caea4d941fa54c203899e
