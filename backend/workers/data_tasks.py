# backend/workers/data_tasks.py
"""
Celery tasks for data collection and processing.
Scheduled tasks for updating market data, features, and maintenance.
"""

import asyncio
from collections.abc import Coroutine
from datetime import datetime, timedelta
from typing import Any, TypeVar

import pandas as pd
from celery import group, shared_task
from loguru import logger

from backend.config.settings import get_settings
from backend.data_engine.collectors.yfinance_collector import YFinanceCollector
from backend.data_engine.transformers.feature_engineering import FeatureEngineer
from backend.db.models import (
    bulk_insert_features,
    bulk_insert_price_data,
    close_db,
    execute_raw_sql,
    get_latest_price,
    reset_db_engine,
)

T = TypeVar("T")

settings = get_settings()


def run_async(coro: Coroutine[Any, Any, T]) -> T:
    """
    Run an async coroutine from a synchronous Celery task.

    Creates a fresh event loop and resets the global DB engine/session
    factory so that asyncpg connections are always bound to the current
    loop. This prevents 'Future attached to a different loop' and
    'Event loop is closed' errors in Celery's prefork worker pool.
    """
    # Discard any stale engine whose connections belong to a previous loop
    reset_db_engine()

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(coro)
    finally:
        # Dispose the engine while the loop is still open so asyncpg can
        # cleanly close its connections.
        try:
            loop.run_until_complete(close_db())
        except Exception:
            pass
        loop.close()


# Popular tickers to update automatically
DEFAULT_TICKERS = [
    # Tech gigants
    "AAPL",  # Apple Inc.
    "MSFT",  # Microsoft Corporation
    "GOOGL",  # Alphabet Inc.
    "AMZN",  # Amazon.com, Inc.
    "META",  # Meta Platforms, Inc.
    "NVDA",  # NVIDIA Corporation
    "TSLA",  # Tesla, Inc.
    # Finance
    "JPM",  # JPMorgan Chase & Co.
    "BAC",  # Bank of America Corporation
    "WFC",  # Wells Fargo & Company
    "GS",  # The Goldman Sachs Group, Inc.
    "C",  # Citigroup Inc.
    # Consumer
    "WMT",  # Walmart Inc.
    "HD",  # The Home Depot, Inc.
    "MCD",  # McDonald's Corporation
    "NKE",  # NIKE, Inc.
    "SBUX",  # Starbucks Corporation
    # Healthcare
    "JNJ",  # Johnson & Johnson
    "PFE",  # Pfizer Inc.
    "UNH",  # UnitedHealth Group Incorporated
    "ABBV",  # AbbVie Inc.
    "TMO",  # Thermo Fisher Scientific Inc.
    # Energy
    "XOM",  # Exxon Mobil Corporation
    "CVX",  # Chevron Corporation
    "COP",  # ConocoPhillips
    # ETFs
    "SPY",  # SPDR S&P 500 ETF Trust
    "QQQ",  # Invesco QQQ Trust
    "IWM",  # iShares Russell 2000 ETF
    "DIA",  # SPDR Dow Jones Industrial Average ETF Trust
]


@shared_task(
    bind=True,
    name="workers.data_tasks.update_ticker_data",
    max_retries=3,
    default_retry_delay=300,
)
def update_ticker_data(
    self, ticker: str, days: int = 7, include_features: bool = True
) -> dict[str, Any]:
    """
    Update price data and features for a single ticker.

    Args:
        ticker: Stock ticker symbol
        days: Number of days to fetch (deftault: 7 for daily updates)
        include_features: Whether to compute and store features

    Returns:
        Dictionary with update results
    """
    try:
        logger.info(f"Updating data for {ticker}")

        result = run_async(_update_ticker_data_async(ticker, days, include_features))
        return result

    except Exception as e:
        logger.error(f"Error updating {ticker}: {e}")
        # Retry the task
        raise self.retry(exc=e) from e


async def _update_ticker_data_async(
    ticker: str, days: int, include_features: bool
) -> dict[str, Any]:
    """Async implementation of update_ticker_data."""
    # Initialize collector
    collector = YFinanceCollector()

    # Get date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)

    # Collect data
    data = await collector.collect_with_retry(
        ticker=ticker, start_date=start_date, end_date=end_date
    )

    if data is None or data.height == 0:
        logger.warning(f"No data collected for {ticker}")
        return {
            "ticker": ticker,
            "status": "no_data",
            "price_rows": 0,
            "feature_rows": 0,
        }

    logger.info(f"Collected {data.height} rows for {ticker}")

    # Store price data
    data_pd = data.to_pandas()
    price_data = []
    for _, row in data_pd.iterrows():
        price_data.append(
            {
                "time": row["time"],
                "ticker": ticker,
                "open": float(row["open"]) if pd.notna(row["open"]) else None,
                "high": float(row["high"]) if pd.notna(row["high"]) else None,
                "low": float(row["low"]) if pd.notna(row["low"]) else None,
                "close": float(row["close"]) if pd.notna(row["close"]) else None,
                "volume": int(row["volume"]) if pd.notna(row["volume"]) else None,
                "adjusted_close": float(row["close"]) if pd.notna(row["close"]) else None,
                "dividends": 0.0,
                "stock_splits": 0.0,
            }
        )

    # Bulk insert price data
    price_count = await bulk_insert_price_data(price_data)
    logger.success(f"Inserted {price_count} price rows for {ticker}")

    feature_count = 0

    # Compute and store features if requested
    if include_features:
        logger.info(f"Computing features for {ticker}")

        fe = FeatureEngineer()
        enriched = fe.create_all_features(data, add_lags=True, add_rolling=True)

        # Fet top features (limit to prevent storage bloat)
        feature_names = fe.get_all_feature_names()[:50]

        enriched_pd = enriched.to_pandas()
        feature_data = []

        for _, row in enriched_pd.iterrows():
            for fname in feature_names:
                if fname in row.index:
                    val = row[fname]
                    if pd.notna(val):
                        feature_data.append(
                            {
                                "time": row["time"],
                                "ticker": ticker,
                                "feature_name": fname,
                                "feature_value": float(val),
                                "feature_category": _get_feature_category(fname),
                            }
                        )

        # Bulk insert features
        if feature_data:
            feature_count = await bulk_insert_features(feature_data)
            logger.success(f"Inserted {feature_count} feature rows for {ticker}")

    return {
        "ticker": ticker,
        "status": "success",
        "price_rows": price_count,
        "feature_rows": feature_count,
        "updated_at": datetime.now().isoformat(),
    }


@shared_task(name="workers.data_tasks.update_all_tickers")
def update_all_tickers(tickers: list[str] | None = None, days: int = 1) -> dict[str, Any]:
    """
    Update data for all tracked tickers in parallel

    This is a scheduled task that runs daily after market close

    Args:
        tickers: List of tickers to update (default: DEFAULT_TICKERS)
        days: Number of days to fetch

    Returns:
        Summary of updates
    """
    try:
        logger.info("=" * 60)
        logger.info("SCHEDULED TASK: Update All Tickers")
        logger.info("=" * 60)

        if tickers is None:
            tickers = DEFAULT_TICKERS

        logger.info(f"Updating {len(tickers)} tickers")

        # Create parallel tasks using Celery group
        job = group(
            update_ticker_data.s(ticker, days=days, include_features=True) for ticker in tickers
        )

        # Execute in parallel
        result = job.apply_async()

        # Wait for completition (with timeout)
        results = result.get(timeout=3600)  # 1 hour timeout

        # Aggregate results
        successful = sum(1 for r in results if r["status"] == "success")
        failed = len(results) - successful
        total_price_rows = sum(r["price_rows"] for r in results)
        total_feature_rows = sum(r["feature_rows"] for r in results)

        summary = {
            "task": "update_all_tickers",
            "total_tickers": len(tickers),
            "successful": successful,
            "failed": failed,
            "total_price_rows": total_price_rows,
            "total_feature_rows": total_feature_rows,
            "completed_at": datetime.now().isoformat(),
        }

        logger.success(f"✅ Update complete: {successful}/{len(tickers)} successful")
        logger.info(f"Total price rows: {total_price_rows}")
        logger.info(f"Total feature rows: {total_feature_rows}")

        return summary

    except Exception as e:
        logger.error(f"Error in update_all_tickers: {e}")
        raise


@shared_task(name="workers.data_tasks.update_all_features")
def update_all_features(tickers: list[str] | None = None, days: int = 90) -> dict[str, Any]:
    """
    Recalculate features for all tickers

    This is a scheduled task that runs weekly to ensure feature consistency

    Args:
        tickers: List of tickers (default: DEFAULT_TICKERS)
        days: Number of days of data to use for feature calculation

    Returns:
        Summary of feature updates
    """
    try:
        logger.info("=" * 60)
        logger.info("SCHEDULED TASK: Update All Features")
        logger.info("=" * 60)

        if tickers is None:
            tickers = DEFAULT_TICKERS

        summary = run_async(_update_all_features_async(tickers, days))

        logger.success(
            f"✅ Feature update complete: {summary['successful']}/{summary['total_tickers']}"
        )
        return summary

    except Exception as e:
        logger.error(f"Error in update_all_features: {e}")
        raise


async def _update_all_features_async(tickers: list[str], days: int) -> dict[str, Any]:
    """Async implementation of update_all_features."""
    collector = YFinanceCollector()
    fe = FeatureEngineer()

    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)

    total_rows = 0
    successful = 0

    for ticker in tickers:
        try:
            logger.info(f"Processing features for {ticker}")

            # Collect data
            data = await collector.collect_with_retry(
                ticker=ticker, start_date=start_date, end_date=end_date
            )

            if data is None or data.height == 0:
                logger.warning(f"No data for {ticker}")
                continue

            # Engineer features
            enriched = fe.create_all_features(data, add_lags=True, add_rolling=True)

            # Prepare feature data
            feature_names = fe.get_all_feature_names()[:50]
            enriched_pd = enriched.to_pandas()

            feature_data = []
            for _, row in enriched_pd.iterrows():
                for fname in feature_names:
                    if fname in row.index:
                        val = row[fname]
                        if pd.notna(val):
                            feature_data.append(
                                {
                                    "time": row["time"],
                                    "ticker": ticker,
                                    "feature_name": fname,
                                    "feature_value": float(val),
                                    "feature_category": _get_feature_category(fname),
                                }
                            )

            # Bulk insert
            if feature_data:
                count = await bulk_insert_features(feature_data)
                total_rows += count
                successful += 1
                logger.success(f"✅ {ticker}: {count} feature rows")
        except Exception as e:
            logger.error(f"Error processing {ticker}: {e}")

    return {
        "task": "update_all_features",
        "total_tickers": len(tickers),
        "successful": successful,
        "total_feature_rows": total_rows,
        "completed_at": datetime.now().isoformat(),
    }


@shared_task(name="workers.data_tasks.health_check_task")
def health_check_task() -> dict[str, Any]:
    """
    Periodic health check for data services

    Runs every hour to verify:
    - Database connectivity
    - Data freshness
    - Service availability

    Returns:
        Health check results
    """
    try:
        logger.info("Running health check...")

        health_status = run_async(_health_check_async())

        logger.info(f"Health check complete: {health_status['overall']}")
        return health_status

    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "timestamp": datetime.now().isoformat(),
            "overall": "unhealthy",
            "error": str(e),
        }


async def _health_check_async() -> dict[str, Any]:
    """Async implementation of health_check_task."""
    from backend.db.models import check_db_connection

    health_status = {
        "timestamp": datetime.now().isoformat(),
        "database": "unknown",
        "data_freshness": "unknown",
        "collectors": "unknown",
    }

    # Check database
    try:
        db_ok = await check_db_connection()
        health_status["database"] = "healthy" if db_ok else "unhealthy"
    except Exception as e:
        logger.error(f"Database check failed: {e}")
        health_status["database"] = "unhealthy"

    # Check data freshness (most recent data should be within 24 hours)
    try:
        latest = await get_latest_price("AAPL")
        if latest:
            hours_old = (datetime.now() - latest.time).total_seconds() / 3600
            if hours_old < 24:
                health_status["data_freshness"] = "fresh"
            elif hours_old < 72:
                health_status["data_freshness"] = "stale"
            else:
                health_status["data_freshness"] = "very_stale"

            health_status["latest_data_age_hours"] = round(hours_old, 2)
        else:
            health_status["data_freshness"] = "no_data"
    except Exception as e:
        logger.error(f"Data freshness check failed: {e}")
        health_status["data_freshness"] = "unknown"

    # Check collectors
    try:
        collector = YFinanceCollector()
        collector_health = await collector.health_check()
        health_status["collectors"] = collector_health["status"]
    except Exception as e:
        logger.error(f"Collector check failed: {e}")
        health_status["collectors"] = "unhealthy"

    # Overall status
    if all(
        v in ["healthy", "fresh"]
        for k, v in health_status.items()
        if k not in ["timestamp", "latest_data_age_hours"]
    ):
        health_status["overall"] = "healthy"
    elif "unhealthy" in health_status.values() or "no_data" in health_status.values():
        health_status["overall"] = "unhealthy"
    else:
        health_status["overall"] = "degraded"

    return health_status


@shared_task(name="workers.data_tasks.cleanup_old_results")
def cleanup_old_results(days_to_keep: int = 90) -> dict[str, Any]:
    """
    Cleanup old daa to manage storage

    Runs weekly to remove:
    - Old predictions (keep 90 days)
    - Old backtest results (keep 180 days)
    - Temporary data

    Args:
        days_to_keep: Number of days of data to retain

    Returns:
        Cleanup summary
    """
    try:
        logger.info("=" * 60)
        logger.info("SCHEDULED TASK: Cleanup Old Results")
        logger.info("=" * 60)

        summary = run_async(_cleanup_old_results_async(days_to_keep))

        logger.success(f"✅ Cleanup completed: {summary['total_deleted']} records deleted")
        return summary

    except Exception as e:
        logger.error(f"Cleanup task failed: {e}")
        raise


async def _cleanup_old_results_async(days_to_keep: int) -> dict[str, Any]:
    """Async implementation of cleanup_old_results."""
    cutoff_date = datetime.now() - timedelta(days=days_to_keep)
    deleted_counts = {}

    # Clean old predictions
    try:
        sql = f"""
        DELETE FROM predictions
        WHERE prediction_time < '{cutoff_date.isoformat()}'
        RETURNING *
        """
        result = await execute_raw_sql(sql)
        deleted_counts["predictions"] = len(result) if result else 0
        logger.info(f"Deleted {deleted_counts['predictions']} old predictions")
    except Exception as e:
        logger.error(f"Error cleaning sentiment: {e}")
        deleted_counts["sentiment"] = 0

    # Vacuum database to reclaim space
    try:
        logger.info("Running VACUUM ANALYZE...")
        await execute_raw_sql("VACUUM ANALYZE")
        logger.success("✅ Database vacuumed")
    except Exception as e:
        logger.error(f"Error running vacuum: {e}")

    return {
        "task": "cleanup_old_results",
        "cutoff_date": cutoff_date.isoformat(),
        "deleted_counts": deleted_counts,
        "total_deleted": sum(deleted_counts.values()),
        "completed_at": datetime.now().isoformat(),
    }


@shared_task(name="workers.data_tasks.sync_ticker_list")
def sync_ticker_list() -> dict[str, Any]:
    """
    Sync list of tracked tickers from various sources

    Can be extended to:
    - Pull tickers from S&P 500
    - Update from user watchlists
    - Add trending tickers

    Returns:
        Sync results
    """
    try:
        logger.info("Syncing ticker list...")

        # For now, just return the default list
        # In future, can add logic to:
        # 1. Fetch S&P 500 constitutents
        # 2. Add user watchlists
        # 3. Add trending tickers from social media

        current_tickers = DEFAULT_TICKERS

        return {
            "task": "sync_ticker_list",
            "total_tickers": len(current_tickers),
            "tickers": current_tickers,
            "updated_at": datetime.now().isoformat(),
        }

    except Exception as e:
        logger.error(f"Ticker sync failed: {e}")
        raise


# Helper functions


def _get_feature_category(feature_name: str) -> str:
    """Determine feature category from name"""
    feature_lower = feature_name.lower()

    if any(x in feature_lower for x in ["return", "price", "gap", "change", "typical", "weighted"]):
        return "price"
    elif any(x in feature_lower for x in ["volume", "obv", "vwap", "cmf"]):
        return "volume"
    elif any(x in feature_lower for x in ["volatility", "atr", "bb", "parkinson"]):
        return "volatility"
    elif any(x in feature_lower for x in ["rsi", "stoch", "williams", "roc", "cci", "mfi"]):
        return "momentum"
    elif any(
        x in feature_lower for x in ["sma", "ema", "macd", "adx", "psar", "trend", "cross", "above"]
    ):
        return "trend"
    else:
        return "statistical"


# Task for manual trigger
@shared_task(name="workers.data_tasks.force_update_ticker")
def force_update_ticker(ticker: str, days: int = 365) -> dict[str, Any]:
    """
    Force full update of a ticker (including historical data)

    Use this for:
    - Adding new tickers to system
    - Recovering from data gaps
    - Manual updates

    Args:
        ticker: Ticker symbol
        days: Days of historical data to fetch

    Returns:
        Update results
    """
    logger.info(f"Force updating {ticker} with {days} days of data")

    return update_ticker_data(ticker=ticker, days=days, include_features=True)


# Monitoring task
@shared_task(name="workers.data_tasks.generate_data_report")
def generate_data_report() -> dict[str, Any]:
    """
    Generate daily data quality report

    Returns:
        Report with data statistics
    """
    try:
        logger.info("Generating data quality report...")

        report = run_async(_generate_data_report_async())

        logger.info(f"Report generated: {report['metrics']}")
        return report

    except Exception as e:
        logger.error(f"Report generation failed: {e}")
        raise


async def _generate_data_report_async() -> dict[str, Any]:
    """Async implementation of generate_data_report."""
    report: dict[str, Any] = {
        "generated_at": datetime.now().isoformat(),
        "metrics": {},
    }

    # Count total records
    try:
        sql = "SELECT COUNT(*) as count FROM price_data"
        result = await execute_raw_sql(sql)
        report["metrics"]["total_price_rows"] = result[0][0] if result else 0
    except Exception as e:
        logger.error(f"Error counting price records: {e}")

    # Count by ticker
    try:
        sql = """
        SELECT ticker, COUNT(*) as count
        FROM price_data
        GROUP BY ticker
        ORDER BY count DESC
        LIMIT 10
        """
        result = await execute_raw_sql(sql)
        report["metrics"]["top_tickers"] = {r[0]: r[1] for r in result} if result else 0
    except Exception as e:
        logger.error(f"Error counting by ticker: {e}")

    # Count features
    try:
        sql = "SELECT COUNT(*) as count FROM features"
        result = await execute_raw_sql(sql)
        report["metrics"]["total_features"] = result[0][0] if result else 0
    except Exception as e:
        logger.error(f"Error counting features: {e}")

    # Latest data timestamp
    try:
        sql = "SELECT MAX(time) FROM price_data"
        result = await execute_raw_sql(sql)
        report["metrics"]["latest_data"] = result[0][0].isoformat()
    except Exception as e:
        logger.error(f"Error getting latest timestamp: {e}")

    return report
