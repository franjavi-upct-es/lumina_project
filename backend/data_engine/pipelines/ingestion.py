# backend/data_engine/pipelines/ingestion.py
"""
Data Ingestion Pipeline for V3
==============================

Async ETL pipeline that collects data from multiple sources
and stores it in the feature store.

Pipeline Flow:
    1. Collect raw data from sources (YFinance, AlphaVantage, etc.)
    2. Validate and clean data
    3. Engineer features
    4. Store in feature store (hot + cold)

Version: 3.0.0
"""

import asyncio
from datetime import datetime
from typing import Any

import polars as pl
from loguru import logger

from backend.data_engine.collectors import (
    AlphaVantageCollector,
    FREDCollector,
    NewsCollector,
    RedditCollector,
    YFinanceCollector,
)
from backend.data_engine.feature_store.client import get_feature_store_client
from backend.data_engine.transformers.feature_engineering import get_feature_engineer


class IngestionPipeline:
    """
    Async ETL pipeline for data ingestion

    Orchestrates data collection, transformation, and storage.
    """

    def __init__(self):
        """Initialize ingestion pipeline"""
        self.yfinance = YFinanceCollector()
        self.alpha_vantage = AlphaVantageCollector()
        self.fred = FREDCollector()
        self.news = NewsCollector()
        self.reddit = RedditCollector()

        self.feature_store = get_feature_store_client()
        self.feature_engineer = get_feature_engineer()

        logger.info("IngestionPipeline initialized")

    async def ingest_ticker(
        self,
        ticker: str,
        start_date: datetime,
        end_date: datetime | None = None,
        include_fundamentals: bool = False,
        include_news: bool = False,
        include_social: bool = False,
    ) -> dict[str, Any]:
        """
        Ingest all data for a ticker

        Args:
            ticker: Asset ticker
            start_date: Start date for data collection
            end_date: End date (default: now)
            include_fundamentals: Fetch fundamental data
            include_news: Fetch news data
            include_social: Fetch social sentiment

        Returns:
            Status dict with results
        """
        try:
            end_date = end_date or datetime.utcnow()

            logger.info(f"Starting ingestion for {ticker}")

            # Collect price data
            price_data = await self.yfinance.collect_with_retry(
                ticker=ticker,
                start_date=start_date,
                end_date=end_date,
            )

            if price_data is None or len(price_data) == 0:
                logger.warning(f"No price data collected for {ticker}")
                return {"status": "failed", "reason": "no_data"}

            # Engineer features
            features = self.feature_engineer.create_all_features(price_data)

            # Store in feature store
            metadata = await self.feature_store.store_features(
                ticker=ticker,
                features=features,
                metadata={
                    "source": "yfinance",
                    "ingestion_date": datetime.utcnow().isoformat(),
                },
            )

            result = {
                "status": "success",
                "ticker": ticker,
                "data_points": len(features),
                "features": metadata.feature_count,
                "time_range": metadata.time_range,
            }

            # Optional: Collect fundamentals
            if include_fundamentals:
                fundamentals = await self._collect_fundamentals(ticker)
                result["fundamentals"] = fundamentals is not None

            # Optional: Collect news
            if include_news:
                news = await self._collect_news(ticker, start_date, end_date)
                result["news_articles"] = len(news) if news else 0

            # Optional: Collect social sentiment
            if include_social:
                social = await self._collect_social(ticker)
                result["social_mentions"] = len(social) if social else 0

            logger.success(f"Ingestion complete for {ticker}")
            return result

        except Exception as e:
            logger.error(f"Error ingesting {ticker}: {e}")
            return {"status": "error", "ticker": ticker, "error": str(e)}

    async def ingest_multiple(
        self,
        tickers: list[str],
        start_date: datetime,
        end_date: datetime | None = None,
        max_concurrent: int = 5,
    ) -> list[dict[str, Any]]:
        """
        Ingest multiple tickers concurrently

        Args:
            tickers: List of tickers
            start_date: Start date
            end_date: End date
            max_concurrent: Max concurrent tasks

        Returns:
            List of result dicts
        """
        results = []

        # Process in batches
        for i in range(0, len(tickers), max_concurrent):
            batch = tickers[i : i + max_concurrent]

            tasks = [self.ingest_ticker(ticker, start_date, end_date) for ticker in batch]

            batch_results = await asyncio.gather(*tasks, return_exceptions=True)

            for result in batch_results:
                if isinstance(result, Exception):
                    results.append({"status": "error", "error": str(result)})
                else:
                    results.append(result)

            # Small delay between batches
            await asyncio.sleep(1.0)

        logger.success(f"Ingested {len(tickers)} tickers")
        return results

    async def _collect_fundamentals(self, ticker: str) -> pl.DataFrame | None:
        """Collect fundamental data"""
        try:
            fundamentals = await self.alpha_vantage.get_company_overview(ticker)
            return fundamentals
        except Exception as e:
            logger.error(f"Error collecting fundamentals: {e}")
            return None

    async def _collect_news(
        self,
        ticker: str,
        start_date: datetime,
        end_date: datetime,
    ) -> pl.DataFrame | None:
        """Collect news data"""
        try:
            news = await self.news.collect(ticker, start_date, end_date)
            return news
        except Exception as e:
            logger.error(f"Error collecting news: {e}")
            return None

    async def _collect_social(self, ticker: str) -> pl.DataFrame | None:
        """Collect social sentiment"""
        try:
            social = await self.reddit.search_ticker(ticker)
            return social
        except Exception as e:
            logger.error(f"Error collecting social: {e}")
            return None


# Global instance
_pipeline: IngestionPipeline | None = None


def get_ingestion_pipeline() -> IngestionPipeline:
    """Get global ingestion pipeline"""
    global _pipeline
    if _pipeline is None:
        _pipeline = IngestionPipeline()
    return _pipeline
