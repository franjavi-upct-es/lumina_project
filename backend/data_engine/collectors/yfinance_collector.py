# backend/data_engine/collectors/yfinance_collector.py
"""
TFinance data collector - Primary source for historical stock data
"""

from typing import Optional, Dict, Any
from datetime import datetime, timedelta
import polars as pl
import yfinance as yf
from loguru import logger
import asyncio

from backend.data_engine.collectors.base_collector import BaseDataCollector


class YFinanceCollector(BaseDataCollector):
    """
    Collector for Yahoo Finance data using yfinance library
    """

    def __init__(self, rate_limit: int = 2000):
        super().__init__(name="YFinance", rate_limit=rate_limit)
        self._cache = {}

    async def collect(
        self,
        ticker: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        interval: str = "1d",
        **kwargs,
    ) -> pl.DataFrame:
        """
        Collect historical price data from Yahoo Finance

        Args:
            ticker: Stocks ticker symbol
            start_date: Start date (default: 5 years ago)
            end_date: End date (default: today)
            interval: Data interval (1d, 1h, etc.)

        Returns:
            Polars DataFrame with OHLCV data
        """
        # Set defaults
        if end_date is None:
            end_date = datetime.now()
        if start_date is None:
            start_date = end_date - timedelta(days=365 * 5)  # 5 years

        try:
            # Run in executor to avoid blocking
            loop = asyncio.get_event_loop()
            data = await loop.run_in_executor(
                None, self._fetch_yfinance_data, ticker, start_date, end_date, interval
            )

            if data is None:
                return None

            # Convert to Polars
            df = pl.from_pandas(data.reset_index())

            # Standardize columns
            df = self._standardize_columns(df)

            # Add meta_data
            df = self._add_metadata(df, ticker, "yfinance")

            # Ensure datetime column
            if "Date" in df.columns:
                df = df.with_columns(
                    pl.col("Date").cast(pl.Datetime).alias("time")
                ).drop("Date")
            elif "datetime" in df.columns:
                df = df.rename({"datetime": "time"})

            return df

        except Exception as e:
            logger.error(f"Error fetching data for {ticker}: {e}")
            return None

    def _fetch_yfinance_data(
        self, ticker: str, start_date: datetime, end_date: datetime, interval: str
    ):
        """
        Synchronous fetch from yfinance
        """
        stock = yf.Ticker(ticker)
        data = stock.history(
            start=start_date,
            end=end_date,
            interval=interval,
            auto_adjust=True,  # Use adjusted prices
            actions=True,  # Include dividends and splits
        )

        if data.empty:
            logger.warning(f"No data returned for {ticker}")
            return None
        
        return data

    async def validate_data(self, data: pl.DataFrame) -> bool:
        """
        Validate the colleted data
        """
        if data is None or data.height == 0:
            return False

        required_columns = ["time", "open", "high", "low", "close", "volume"]
        missing_columns = [col for col in required_columns if col not in data.columns]

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
            (pl.col("high") < pl.col("low"))
            | (pl.col("close") < 0)
            | (pl.col("volume") < 0)
        )

        if invalid_prices.height > 0:
            logger.error(f"Found {invalid_prices.height} rows with invalid prices")
            return False

        return True

    async def get_company_info(self, ticker: str) -> Optional[Dict[str, Any]]:
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

    def _fetch_company_info(self, ticker: str) -> Dict[str, Any]:
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
        self, ticker: str, expiration_date: Optional[str] = None
    ) -> Optional[Dict[str, pl.DataFrame]]:
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
        self, ticker: str, expiration_date: Optional[str]
    ) -> Dict[str, pl.DataFrame]:
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

        return {"calls": calls, "puts": puts, "expiration": expiration_date}

    async def get_institutional_holders(self, ticker: str) -> Optional[pl.DataFrame]:
        """
        Get institutional holders information
        """
        try:
            loop = asyncio.get_event_loop()
            data = await loop.run_in_executor(
                None, self._fetch_institutional_holders, ticker
            )
            return data
        except Exception as e:
            logger.error(f"Error fetching institutional holders for {ticker}: {e}")
            return None

    def _fetch_institutional_holders(self, ticker: str) -> pl.DataFrame:
        """
        Synchronous fetch of institutional holders
        """
        stock = yf.Ticker(ticker)
        holders = stock.institutional_holders

        if holders is None or holders.empty:
            return None

        return pl.from_pandas(holders)

    async def get_earnings_history(self, ticker: str) -> Optional[pl.DataFrame]:
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

    def _fetch_earnings(self, ticker: str) -> pl.DataFrame:
        """
        Synchronous fetch of earnings
        """
        stock = yf.Ticker(ticker)
        earnings = stock.earnings_history

        if earnings is None or earnings.empty:
            return None

        df = pl.from_pandas(earnings.reset_index())
        return self._standardize_columns(df)
