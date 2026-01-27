# backend/db/__init__.py
"""
Database Module for Lumina Quant Lab

Provides database models and utilities for TimescaleDB:

Models:
- PriceData: Historical OHLCV data (hypertable)
- Feature: Computed features storage (hypertable)
- SentimentData: Sentiment scores (hypertable)
- Prediction: Model predictions (hypertable)
- BacktestResult: Backtest results
- BacktestTrade: Individual trades
- Model: Trained model metadata

Utilities:
- Database connection management
- Async session factories
- Bulk insert operations
- Query helpers

Usage:
    from backend.db import get_async_engine, get_async_session
    from backend.db.models import PriceData, Feature, BacktestResult

    # Get async session
    async with get_async_session() as session:
        # Query price data
        result = await session.execute(
            select(PriceData).where(PriceData.ticker == "AAPL")
        )
        prices = result.scalars().all()

        # Insert new data
        await bulk_insert_price_data(session, data)
"""

from backend.db.models import (
    # Models
    BacktestResult,
    BacktestTrade,
    Base,
    Feature,
    Model,
    Prediction,
    PriceData,
    SentimentData,
    # Utilities
    bulk_insert_features,
    bulk_insert_price_data,
    # Engine and session
    get_async_engine,
    get_async_session,
    init_db,
)

__all__ = [
    # Base
    "Base",
    # Models
    "PriceData",
    "Feature",
    "SentimentData",
    "Prediction",
    "BacktestResult",
    "BacktestTrade",
    "Model",
    # Engine/Session
    "get_async_engine",
    "get_async_session",
    "init_db",
    # Utilities
    "bulk_insert_price_data",
    "bulk_insert_features",
]
