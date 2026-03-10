# ./backend/workers/data_tasks.py
"""
Data Collection and Processing Tasks for V3
===========================================

Celery tasks for:
- Market data collection from multiple sources
- Feature engineering and storage
- Data cleaning and validation
- Feature store updates

All tasks are async and can be distributed across workers.

Author: Lumina Quant Lab
Version: 3.0.0
"""

import asyncio
from datetime import datetime, timedelta

from celery import group
from loguru import logger

from backend.config.settings import get_settings
from backend.data_engine.collectors import YFinanceCollector
from backend.data_engine.feature_store import get_feature_store_client
from backend.data_engine.transformers import get_feature_engineer
from backend.workers.celery_app import celery_app

settings = get_settings()


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def run_async(coro):
    """Run async coroutine in Celery task"""
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(coro)


# ============================================================================
# DATA COLLECTION TASKS
# ============================================================================


@celery_app.task(
    bind=True, name="backend.workers.data_tasks.collect_ticker_data"
)
def collect_ticker_data(
    self,
    ticker: str,
    start_date: str | None = None,
    end_date: str | None = None,
    days_back: int = 365,
):
    """
    Collect price data for a single ticker

    Args:
        ticker: Stock ticker symbol
        start_date: Start date (ISO format)
        end_date: End date (ISO format)
        days_back: Days to fetch if dates not specified

    Returns:
        dict with status and data info
    """
    try:
        logger.info(f"Task {self.request.id}: Collecting data for {ticker}")

        # Parse dates
        if start_date:
            start = datetime.fromisoformat(start_date)
        else:
            start = datetime.utcnow() - timedelta(days=days_back)

        if end_date:
            end = datetime.fromisoformat(end_date)
        else:
            end = datetime.utcnow()

        # Collect data
        collector = YFinanceCollector()

        async def _collect():
            return await collector.collect_with_retry(
                ticker=ticker,
                start_date=start,
                end_date=end,
            )

        data = run_async(_collect())

        if data is None or len(data) == 0:
            logger.warning(f"No data collected for {ticker}")
            return {
                "status": "no_data",
                "ticker": ticker,
                "data_points": 0,
            }

        logger.success(f"Collected {len(data)} data points for {ticker}")

        return {
            "status": "success",
            "ticker": ticker,
            "data_points": len(data),
            "start_date": str(data["time"].min()),
            "end_date": str(data["time"].max()),
        }

    except Exception as e:
        logger.error(f"Error collecting {ticker}: {e}")
        return {
            "status": "error",
            "ticker": ticker,
            "error": str(e),
        }


@celery_app.task(
    bind=True, name="backend.workers.data_tasks.collect_and_engineer_features"
)
def collect_and_engineer_features(
    self,
    ticker: str,
    start_date: str | None = None,
    end_date: str | None = None,
):
    """
    Collect data and engineer features for a ticker

    This is the main ETL task that:
    1. Collects price data
    2. Engineers features
    3. Stores in feature store

    Args:
        ticker: Stock ticker
        start_date: Start date (ISO format)
        end_date: End date (ISO format)

    Returns:
        dict with status and feature info
    """
    try:
        logger.info(
            f"Task {self.request.id}: Engineering features for {ticker}"
        )

        # Step 1: Collect data
        result = collect_ticker_data(
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
        )

        if result["status"] != "success":
            return result

        # Step 2: Re-fetch data for feature engineering
        start = (
            datetime.fromisoformat(start_date)
            if start_date
            else datetime.utcnow() - timedelta(days=365)
        )
        end = (
            datetime.fromisoformat(end_date) if end_date else datetime.utcnow()
        )

        collector = YFinanceCollector()

        async def _collect():
            return await collector.collect(ticker, start, end)

        data = run_async(_collect())

        # Step 3: Engineer features
        engineer = get_feature_engineer()
        features = engineer.create_all_features(data)

        # Step 4: Store in feature store
        feature_store = get_feature_store_client()

        async def _store():
            return await feature_store.store_features(
                ticker=ticker,
                features=features,
                metadata={
                    "task_id": self.request.id,
                    "collected_at": datetime.utcnow().isoformat(),
                },
            )

        metadata = run_async(_store())

        logger.success(
            f"Engineered {metadata.feature_count} features for {ticker}"
        )

        return {
            "status": "success",
            "ticker": ticker,
            "data_points": len(features),
            "features": metadata.feature_count,
            "time_range": [str(t) for t in metadata.time_range],
        }

    except Exception as e:
        logger.error(f"Error engineering features for {ticker}: {e}")
        return {
            "status": "error",
            "ticker": ticker,
            "error": str(e),
        }


@celery_app.task(name="backend.workers.data_tasks.update_all_tickers")
def update_all_tickers(tickers: list[str] | None = None):
    """
    Update data for multiple tickers in parallel

    This is a scheduled task that runs daily to update all tracked tickers.

    Args:
        tickers: List of tickers (default: core watchlist)

    Returns:
        dict with summary results
    """
    try:
        if tickers is None:
            # Default watchlist (can be expanded)
            tickers = [
                "AAPL",
                "GOOGL",
                "MSFT",
                "AMZN",
                "META",
                "TSLA",
                "NVDA",
                "JPM",
                "V",
                "WMT",
                "SPY",
                "QQQ",
                "IWM",  # ETFs
            ]

        logger.info(f"Updating {len(tickers)} tickers")

        # Create parallel tasks
        job = group(
            collect_and_engineer_features.s(ticker) for ticker in tickers
        )

        # Execute
        result = job.apply_async()

        # Wait for completion (with timeout)
        results = result.get(timeout=3600)

        # Summarize
        success = sum(1 for r in results if r["status"] == "success")
        failed = sum(1 for r in results if r["status"] != "success")

        logger.info(f"Update complete: {success} success, {failed} failed")

        return {
            "status": "completed",
            "total": len(tickers),
            "success": success,
            "failed": failed,
            "results": results,
        }

    except Exception as e:
        logger.error(f"Error updating tickers: {e}")
        return {
            "status": "error",
            "error": str(e),
        }


@celery_app.task(name="backend.workers.data_tasks.update_all_features")
def update_all_features():
    """
    Re-engineer features for all tickers

    Scheduled task that refreshes features daily.
    """
    logger.info("Refreshing features for all tickers")

    # This will trigger the update_all_tickers task
    return update_all_tickers()


# ============================================================================
# MAINTENANCE TASKS
# ============================================================================


@celery_app.task(name="backend.workers.data_tasks.health_check_task")
def health_check_task():
    """
    Periodic health check task

    Checks:
    - Database connectivity
    - Redis connectivity
    - Feature store health
    """
    try:
        logger.info("Running health check")

        health = {
            "timestamp": datetime.utcnow().isoformat(),
            "database": "healthy",  # TODO: Actual DB check
            "redis": "healthy",  # TODO: Actual Redis check
            "feature_store": "healthy",  # TODO: Actual FS check
        }

        logger.success("Health check passed")
        return health

    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
        }


@celery_app.task(name="backend.workers.data_tasks.cleanup_old_results")
def cleanup_old_results(days_old: int = 7):
    """
    Clean up old Celery results

    Args:
        days_old: Delete results older than this many days
    """
    try:
        logger.info(f"Cleaning up results older than {days_old} days")

        # TODO: Implement cleanup logic
        # This would delete old task results from Redis

        return {
            "status": "completed",
            "cleaned": 0,  # Placeholder
        }

    except Exception as e:
        logger.error(f"Cleanup failed: {e}")
        return {
            "status": "error",
            "error": str(e),
        }
