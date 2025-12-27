#!/usr/bin/env python3
"""
Seed database with initial data
Populates TimescaleDB with historical data for testing
"""

import asyncio
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.data_engine.collectors.yfinance_collector import YFinanceCollector
from backend.data_engine.transformers.feature_engineering import FeatureEngineer
from backend.db.models import (
    init_db,
    bulk_insert_price_data,
    bulk_insert_features,
)
from loguru import logger
import pandas as pd


# Default tickers to populate
DEFAULT_TICKERS = [
    # Tech Giants
    "AAPL",
    "MSFT",
    "GOOGL",
    "AMZN",
    "META",
    "NVDA",
    "TSLA",
    # Finance
    "JPM",
    "BAC",
    "WFC",
    "GS",
    # Consumer
    "WMT",
    "HD",
    "MCD",
    "NKE",
    # Healthcare
    "JNJ",
    "PFE",
    "UNH",
    "ABBV",
    # ETFs
    "SPY",
    "QQQ",
    "IWM",
]


async def seed_price_data(tickers: list[str], days: int = 365):
    """
    Seed price data for specified tickers

    Args:
        tickers: List of ticker symbols
        days: Number of days of historical data
    """
    logger.info(f"Seeding price data for {len(tickers)} tickers ({days} days)")

    collector = YFinanceCollector()
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)

    total_rows = 0

    # Collect data for all tickers
    results = await collector.collect_batch(
        tickers=tickers, start_date=start_date, end_date=end_date, max_concurrent=5
    )

    logger.info(f"Collected data for {len(results)}/{len(tickers)} tickers")

    # Process each ticker
    for ticker, data in results.items():
        logger.info(f"Processing {ticker}...")

        if data is None or data.height == 0:
            logger.warning(f"  No data for {ticker}")
            continue

        # Convert to pandas
        data_pd = data.to_pandas()

        # Prepare for insertion
        price_data = []
        for _, row in data_pd.iterrows():
            price_data.append(
                {
                    "time": row["time"],
                    "ticker": ticker,
                    "open": float(row["open"]) if row["open"] else None,
                    "high": float(row["high"]) if row["high"] else None,
                    "low": float(row["low"]) if row["low"] else None,
                    "close": float(row["close"]) if row["close"] else None,
                    "volume": int(row["volume"]) if row["volume"] else None,
                    "adjusted_close": float(row["close"]) if row["close"] else None,
                    "dividends": 0.0,
                    "stock_splits": 0.0,
                }
            )

        # Bulk insert
        try:
            count = await bulk_insert_price_data(price_data)
            total_rows += count
            logger.success(f"  ✅ {ticker}: Inserted {count} rows")
        except Exception as e:
            logger.error(f"  ❌ {ticker}: Failed - {e}")

    logger.success(f"✅ Total price rows inserted: {total_rows}")
    return total_rows


async def seed_features(tickers: list[str], days: int = 90):
    """
    Seed feature data for specified tickers

    Args:
        tickers: List of ticker symbols
        days: Number of days of historical data
    """
    logger.info(f"Seeding features for {len(tickers)} tickers ({days} days)")

    collector = YFinanceCollector()
    fe = FeatureEngineer()

    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)

    total_rows = 0

    # Process each ticker
    for ticker in tickers:
        logger.info(f"Processing {ticker}...")

        # Collect data
        data = await collector.collect_with_retry(
            ticker=ticker, start_date=start_date, end_date=end_date
        )

        if data is None or data.height == 0:
            logger.warning(f"  No data for {ticker}")
            continue

        # Engineer features
        try:
            enriched = fe.create_all_features(data, add_lags=True, add_rolling=True)
            logger.info(f"  Created {len(enriched.columns)} features")
        except Exception as e:
            logger.error(f"  ❌ Feature engineering failed: {e}")
            continue

        # Get feature names (use top 50 to keep storage manageable)
        feature_names = fe.get_all_feature_names()[:50]

        # Convert to pandas
        enriched_pd = enriched.to_pandas()

        # Prepare feature data
        feature_data = []
        for _, row in enriched_pd.iterrows():
            for fname in feature_names:
                if fname in row.index:
                    val = row[fname]
                    if val is not None and not pd.isna(val):
                        feature_data.append(
                            {
                                "time": row["time"],
                                "ticker": ticker,
                                "feature_name": fname,
                                "feature_value": float(val),
                                "feature_category": _get_category(fname),
                            }
                        )

        # Bulk insert
        try:
            count = await bulk_insert_features(feature_data)
            total_rows += count
            logger.success(f"  ✅ {ticker}: Inserted {count} feature rows")
        except Exception as e:
            logger.error(f"  ❌ {ticker}: Feature insert failed - {e}")

    logger.success(f"✅ Total feature rows inserted: {total_rows}")
    return total_rows


def _get_category(feature_name: str) -> str:
    """Determine feature category from name"""
    if any(
        x in feature_name.lower()
        for x in ["return", "price", "gap", "change", "typical", "weighted"]
    ):
        return "price"
    elif any(x in feature_name.lower() for x in ["volume", "obv", "vwap", "cmf"]):
        return "volume"
    elif any(
        x in feature_name.lower() for x in ["volatility", "atr", "bb", "parkinson"]
    ):
        return "volatility"
    elif any(
        x in feature_name.lower()
        for x in ["rsi", "stoch", "williams", "roc", "cci", "mfi"]
    ):
        return "momentum"
    elif any(
        x in feature_name.lower() for x in ["sma", "ema", "macd", "adx", "psar", "bb_"]
    ):
        return "trend"
    else:
        return "statistical"


async def main():
    """Main seeding function"""
    logger.info("=" * 60)
    logger.info("🌱 LUMINA QUANT LAB - DATABASE SEEDING")
    logger.info("=" * 60)

    # Parse arguments
    import argparse

    parser = argparse.ArgumentParser(description="Seed Lumina database")
    parser.add_argument(
        "--tickers",
        nargs="+",
        default=DEFAULT_TICKERS,
        help="Tickers to seed (default: popular stocks)",
    )
    parser.add_argument(
        "--price-days",
        type=int,
        default=365,
        help="Days of price history (default: 365)",
    )
    parser.add_argument(
        "--feature-days",
        type=int,
        default=90,
        help="Days of feature history (default: 90)",
    )
    parser.add_argument(
        "--skip-price", action="store_true", help="Skip price data seeding"
    )
    parser.add_argument(
        "--skip-features", action="store_true", help="Skip feature seeding"
    )

    args = parser.parse_args()

    # Initialize database
    logger.info("\n📊 Step 1: Initialize database")
    try:
        await init_db()
        logger.success("✅ Database initialized")
    except Exception as e:
        logger.error(f"❌ Database initialization failed: {e}")
        return 1

    # Seed price data
    if not args.skip_price:
        logger.info("\n💰 Step 2: Seed price data")
        try:
            price_count = await seed_price_data(args.tickers, args.price_days)
            logger.success(f"✅ Seeded {price_count} price records")
        except Exception as e:
            logger.error(f"❌ Price seeding failed: {e}")
            return 1
    else:
        logger.info("\n⏭️  Skipping price data seeding")

    # Seed features
    if not args.skip_features:
        logger.info("\n🔬 Step 3: Seed feature data")
        try:
            feature_count = await seed_features(args.tickers, args.feature_days)
            logger.success(f"✅ Seeded {feature_count} feature records")
        except Exception as e:
            logger.error(f"❌ Feature seeding failed: {e}")
            return 1
    else:
        logger.info("\n⏭️  Skipping feature seeding")

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("SEEDING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Tickers: {len(args.tickers)}")
    logger.info(f"Price history: {args.price_days} days")
    logger.info(f"Feature history: {args.feature_days} days")
    logger.info("")
    logger.success("🎉 Database seeded successfully!")

    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
