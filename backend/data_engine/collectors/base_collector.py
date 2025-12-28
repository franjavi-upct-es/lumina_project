# backend/data_engine/collectors/base_collector.py
"""
Base class for all data collectors
Provides common interface and utilities
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import polars as pl
from loguru import logger
import asyncio
from tenacity import retry, stop_after_attempt, wait_exponential


class BaseDataCollector(ABC):
    """
    Abstract base class for data collectors
    All collectors must implement the collect() method
    """

    def __init__(self, name: str, rate_limit: int = 100):
        """
        Args:
            name: Collector identifier
            rate_limit: Max requests per minute
        """
        self.name = name
        self.rate_limit = rate_limit
        self._request_timestamps: List[datetime] = []
        logger.info(f"Initialized {name} collector with rate limit {rate_limit}/min")

    @abstractmethod
    async def collect(
        self,
        ticker: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        **kwargs,
    ) -> Optional[pl.DataFrame]:
        """
        Collect data for a specific ticker

        Args:
            ticker: Stock ticker symbol
            start_date: Start date for data collection
            end_date: End date for data collection
            **kwargs: Additional collector-specific parameters

        Returns:
            Polars DataFrame with collected data
        """
        pass

    @abstractmethod
    async def validate_data(self, data: Optional[pl.DataFrame]) -> bool:
        """
        Validate collected data

        Args:
            data: DataFrame to validate

        Returns:
            True if data is valid, False otherwise
        """
        pass

    async def _check_rate_limit(self):
        """
        Enforce rate limiting
        """
        now = datetime.now()
        minute_ago = now - timedelta(minutes=1)

        # Remove old timestamps
        self._request_timestamps = [
            ts for ts in self._request_timestamps if ts > minute_ago
        ]

        # Check if we're at the limit
        if len(self._request_timestamps) >= self.rate_limit:
            wait_time = (self._request_timestamps[0] - minute_ago).total_seconds()
            logger.warning(
                f"Rate limit reached for {self.name}, waiting {wait_time:.2f}s"
            )
            await asyncio.sleep(wait_time + 0.1)

        # Record this request
        self._request_timestamps.append(now)

    @retry(
        stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    async def collect_with_retry(
        self,
        ticker: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        **kwargs,
    ) -> Optional[pl.DataFrame]:
        """
        Collect data with automatic retry on failure

        Returns:
            DataFrame or None if collection failed
        """
        try:
            await self._check_rate_limit()

            logger.info(f"Collecting data for {ticker} from {self.name}")
            data = await self.collect(ticker, start_date, end_date, **kwargs)

            if data is None or data.height == 0:
                logger.warning(f"No data collected for {ticker} from {self.name}")
                return None

            # Validate data
            is_valid = await self.validate_data(data)
            if not is_valid:
                logger.error(f"Data validation failed for {ticker} from {self.name}")
                return None

            logger.success(f"Successfully collected {data.height} rows for {ticker}")
            return data

        except Exception as e:
            logger.error(f"Error collecting data for {ticker} from {self.name}: {e}")
            raise

    async def collect_batch(
        self,
        tickers: List[str],
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        max_concurrent: int = 5,
        **kwargs,
    ) -> Dict[str, pl.DataFrame]:
        """
        Collect data for multiple tickers concurrently

        Args:
            tickers: List of ticker symbols
            start_date: Start date
            end_date: End date
            max_concurrent: Maximum concurrent requests

        Returns:
            Dictionary mapping tickers to their DataFrames
        """
        results = {}
        semaphore = asyncio.Semaphore(max_concurrent)

        async def collect_one(ticker: str):
            async with semaphore:
                try:
                    data = await self.collect_with_retry(
                        ticker, start_date, end_date, **kwargs
                    )
                    if data is not None:
                        results[ticker] = data
                except Exception as e:
                    logger.error(f"Failed to collect {ticker}: {e}")

        tasks = [collect_one(ticker) for ticker in tickers]
        await asyncio.gather(*tasks)

        logger.info(
            f"Batch collection complete: {len(results)}/{len(tickers)} successful"
        )
        return results

    def _standardize_columns(self, data: pl.DataFrame) -> pl.DataFrame:
        """
        Standardize column names to lowercase with underscores

        Args:
            data: Input DataFrame

        Returns:
            DataFrame with standardized column names
        """
        column_mapping = {}
        for col in data.columns:
            # Convert to lowercase and replace spaces/special chars with underscore
            new_col = col.lower().replace(" ", "_").replace("-", "_")
            column_mapping[col] = new_col

        return data.rename(column_mapping)

    def _add_metadata(
        self, data: pl.DataFrame, ticker: str, source: str
    ) -> pl.DataFrame:
        """
        Add meta_data columns to the DataFrame

        Args:
            data: Input DataFrame
            ticker: Ticker symbol
            source: Data source identifier

        Returns:
            DataFrame with added meta_data
        """
        return data.with_columns(
            [
                pl.lit(ticker).alias("ticker"),
                pl.lit(source).alias("source"),
                pl.lit(datetime.now()).alias("collected_at"),
            ]
        )

    async def health_check(self) -> Dict[str, Any]:
        """
        Check if the data source is accessible

        Returns:
            Health status dictionary
        """
        try:
            # Try to collect a small amount of data
            test_data = await self.collect(
                ticker="AAPL",
                start_date=datetime.now() - timedelta(days=7),
                end_date=datetime.now(),
            )

            return {
                "collector": self.name,
                "status": "healthy" if test_data is not None else "degraded",
                "timestamp": datetime.now().isoformat(),
                "rate_limit": self.rate_limit,
                "requests_last_minute": len(self._request_timestamps),
            }
        except Exception as e:
            return {
                "collector": self.name,
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            }
