# backend/data_engine/storage/feature_store.py
"""
Feature Store for storing and retrieving engineered features
Supports both in-memory and persistent storage with TimescaleDB
"""

from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
from pathlib import Path
import asyncio
import json

import polars as pl
import pandas as pd
from loguru import logger
from sqlalchemy import select, and_, delete

from backend.db.models import Feature, bulk_insert_features, get_async_session_factory
from backend.config.settings import get_settings

settings = get_settings()


class FeatureStore:
    """
    Feature Store for managing engineered features

    Features:
    - Store features in TimescaleDB
    - Cache frequently accessed features
    - Versioning support
    - Metadata tracking
    - Batch operations
    """

    def __init__(self):
        self.cache: Dict[str, pl.DataFrame] = {}
        self.cache_ttl = timedelta(hours=settings.FEATURE_CACHE_HOURS)
        self.cache_timestamps: Dict[str, datetime] = {}
        self.metadata: Dict[str, Dict[str, Any]] = {}

        # Storage path for local cache
        self.storage_path = Path(settings.FEATURE_STORE_PATH)
        self.storage_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"FeatureStore initialized with storage at {self.storage_path}")

    async def store_features(
        self,
        ticker: str,
        features: pl.DataFrame,
        feature_names: Optional[List[str]] = None,
        categories: Optional[Dict[str, str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        """
        Store features for a ticker

        Args:
            ticker: Stock ticker
            features: DataFrame with features (must have 'time' column)
            feature_names: List of feature column names to store
            categories: Dict mapping feature names to categories
            metadata: Additional metadata

        Returns:
            Number of feature records stored
        """
        try:
            logger.info(f"Storing features for {ticker}")

            if "time" not in features.columns:
                raise ValueError("Features DataFrame must have 'time' column")

            # Determine which features to store
            if feature_names is None:
                feature_names = [
                    col
                    for col in features.columns
                    if col not in ["time", "ticker", "source", "collected_at"]
                ]

            # Convert to pandas for easier iteration
            features_pd = features.to_pandas()

            # Prepare feature data
            feature_records = []
            for _, row in features_pd.iterrows():
                for fname in feature_names:
                    if fname in row.index:
                        value = row[fname]

                        # Skip NaN values
                        if pd.isna(value):
                            continue

                        # Get category
                        category = (
                            categories.get(fname, "unknown")
                            if categories
                            else "unknown"
                        )

                        feature_records.append(
                            {
                                "time": row["time"],
                                "ticker": ticker,
                                "feature_name": fname,
                                "feature_value": float(value),
                                "feature_category": category,
                            }
                        )

            if not feature_records:
                logger.warning(f"No valid features to store for {ticker}")
                return 0

            # Bulk insert
            count = await bulk_insert_features(feature_records)

            # Update cache
            cache_key = f"{ticker}_features"
            self.cache[cache_key] = features
            self.cache_timestamps[cache_key] = datetime.now()

            # Store metadata
            if metadata:
                self.metadata[ticker] = {
                    **metadata,
                    "feature_count": len(feature_names),
                    "stored_at": datetime.now().isoformat(),
                }

            logger.success(f"Stored {count} feature records for {ticker}")
            return count

        except Exception as e:
            logger.error(f"Error storing features for {ticker}: {e}")
            raise

    async def get_features(
        self,
        ticker: str,
        feature_names: Optional[List[str]] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        use_cache: bool = True,
    ) -> Optional[pl.DataFrame]:
        """
        Retrieve features for a ticker

        Args:
            ticker: Stock ticker
            feature_names: List of specific features to retrieve
            start_date: Start date filter
            end_date: End date filter
            use_cache: Whether to use cached data

        Returns:
            DataFrame with features or None
        """
        try:
            cache_key = f"{ticker}_features"

            # Check cache
            if use_cache and cache_key in self.cache:
                cache_age = datetime.now() - self.cache_timestamps.get(
                    cache_key, datetime.min
                )
                if cache_age < self.cache_ttl:
                    logger.debug(f"Using cached features for {ticker}")
                    return self.cache[cache_key]

            # Query from database
            session_factory = get_async_session_factory()

            async with session_factory() as session:
                query = select(Feature).where(Feature.ticker == ticker)

                # Apply date filters
                if start_date:
                    query = query.where(Feature.time >= start_date)
                if end_date:
                    query = query.where(Feature.time <= end_date)

                # Apply feature name filter
                if feature_names:
                    query = query.where(Feature.feature_name.in_(feature_names))

                query = query.order_by(Feature.time)

                result = await session.execute(query)
                features = result.scalars().all()

                if not features:
                    logger.warning(f"No features found for {ticker}")
                    return None

                # Convert to DataFrame
                data = []
                for f in features:
                    data.append(
                        {
                            "time": f.time,
                            "ticker": f.ticker,
                            "feature_name": f.feature_name,
                            "feature_value": f.feature_value,
                            "feature_category": f.feature_category,
                        }
                    )

                df = pd.DataFrame(data)

                # Pivot to wide format
                df_pivot = df.pivot_table(
                    index="time",
                    columns="feature_name",
                    values="feature_value",
                    aggfunc="first",
                ).reset_index()

                # Convert to Polars
                features_df = pl.from_pandas(df_pivot)

                # Update cache
                self.cache[cache_key] = features_df
                self.cache_timestamps[cache_key] = datetime.now()

                logger.info(
                    f"Retrieved {features_df.height} feature records for {ticker}"
                )
                return features_df

        except Exception as e:
            logger.error(f"Error retrieving features for {ticker}: {e}")
            return None

    async def get_features_batch(
        self,
        tickers: List[str],
        feature_names: Optional[List[str]] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> Dict[str, pl.DataFrame]:
        """
        Retrieve features for multiple tickers

        Args:
            tickers: List of tickers
            feature_names: List of features to retrieve
            start_date: Start date filter
            end_date: End date filter

        Returns:
            Dictionary mapping tickers to their features
        """
        try:
            logger.info(f"Retrieving features for {len(tickers)} tickers")

            results = {}

            # Gather all tasks
            tasks = [
                self.get_features(ticker, feature_names, start_date, end_date)
                for ticker in tickers
            ]

            # Execute concurrently
            features_list = await asyncio.gather(*tasks, return_exceptions=True)

            # Build results dict
            for ticker, features in zip(tickers, features_list):
                if isinstance(features, Exception):
                    logger.error(f"Error for {ticker}: {features}")
                elif features is not None:
                    results[ticker] = features

            logger.success(
                f"Retrieved features for {len(results)}/{len(tickers)} tickers"
            )
            return results

        except Exception as e:
            logger.error(f"Error in batch feature retrieval: {e}")
            return {}

    async def delete_features(
        self,
        ticker: str,
        feature_names: Optional[List[str]] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> int:
        """
        Delete features for a ticker

        Args:
            ticker: Stock ticker
            feature_names: Specific features to delete
            start_date: Delete from this date
            end_date: Delete until this date

        Returns:
            Number of records deleted
        """
        try:
            logger.info(f"Deleting features for {ticker}")

            session_factory = get_async_session_factory()

            async with session_factory() as session:
                query = delete(Feature).where(Feature.ticker == ticker)

                if feature_names:
                    query = query.where(Feature.feature_name.in_(feature_names))
                if start_date:
                    query = query.where(Feature.time >= start_date)
                if end_date:
                    query = query.where(Feature.time <= end_date)

                result = await session.execute(query)
                await session.commit()

                count = result.rowcount

                # Clear cache
                cache_key = f"{ticker}_features"
                if cache_key in self.cache:
                    del self.cache[cache_key]
                    del self.cache_timestamps[cache_key]

                logger.success(f"Deleted {count} feature records for {ticker}")
                return count

        except Exception as e:
            logger.error(f"Error deleting features: {e}")
            raise

    async def list_available_features(
        self, ticker: Optional[str] = None, category: Optional[str] = None
    ) -> Dict[str, List[str]]:
        """
        List available features

        Args:
            ticker: Filter by ticker
            category: Filter by category

        Returns:
            Dictionary with feature information
        """
        try:
            session_factory = get_async_session_factory()

            async with session_factory() as session:
                from sqlalchemy import func, distinct

                query = select(
                    Feature.ticker,
                    Feature.feature_name,
                    Feature.feature_category,
                    func.count(Feature.feature_name).label("count"),
                ).group_by(
                    Feature.ticker, Feature.feature_name, Feature.feature_category
                )

                if ticker:
                    query = query.where(Feature.ticker == ticker)
                if category:
                    query = query.where(Feature.feature_category == category)

                result = await session.execute(query)
                rows = result.all()

                # Organize results
                features_by_ticker = {}
                for row in rows:
                    if row.ticker not in features_by_ticker:
                        features_by_ticker[row.ticker] = []

                    features_by_ticker[row.ticker].append(
                        {
                            "name": row.feature_name,
                            "category": row.feature_category,
                            "count": row.count,
                        }
                    )

                return features_by_ticker

        except Exception as e:
            logger.error(f"Error listing features: {e}")
            return {}

    async def get_feature_statistics(
        self,
        ticker: str,
        feature_name: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> Optional[Dict[str, float]]:
        """
        Get statistics for a specific feature

        Args:
            ticker: Stock ticker
            feature_name: Feature name
            start_date: Start date
            end_date: End date

        Returns:
            Dictionary with statistics
        """
        try:
            session_factory = get_async_session_factory()

            async with session_factory() as session:
                from sqlalchemy import func

                query = select(
                    func.avg(Feature.feature_value).label("mean"),
                    func.stddev(Feature.feature_value).label("std"),
                    func.min(Feature.feature_value).label("min"),
                    func.max(Feature.feature_value).label("max"),
                    func.count(Feature.feature_value).label("count"),
                ).where(
                    and_(Feature.ticker == ticker, Feature.feature_name == feature_name)
                )

                if start_date:
                    query = query.where(Feature.time >= start_date)
                if end_date:
                    query = query.where(Feature.time <= end_date)

                result = await session.execute(query)
                row = result.first()

                if row:
                    return {
                        "mean": float(row.mean) if row.mean else 0.0,
                        "std": float(row.std) if row.std else 0.0,
                        "min": float(row.min) if row.min else 0.0,
                        "max": float(row.max) if row.max else 0.0,
                        "count": int(row.count),
                    }

                return None

        except Exception as e:
            logger.error(f"Error getting feature statistics: {e}")
            return None

    def clear_cache(self, ticker: Optional[str] = None):
        """
        Clear feature cache

        Args:
            ticker: Specific ticker to clear, or None for all
        """
        if ticker:
            cache_key = f"{ticker}_features"
            if cache_key in self.cache:
                del self.cache[cache_key]
                del self.cache_timestamps[cache_key]
                logger.info(f"Cleared cache for {ticker}")
        else:
            self.cache.clear()
            self.cache_timestamps.clear()
            logger.info("Cleared all feature cache")

    def save_metadata(self, filepath: Optional[str] = None):
        """Save metadata to disk"""
        if filepath is None:
            filepath = self.storage_path / "metadata.json"

        try:
            with open(filepath, "w") as f:
                json.dump(self.metadata, f, indent=2, default=str)
            logger.info(f"Saved metadata to {filepath}")
        except Exception as e:
            logger.error(f"Error saving metadata: {e}")

    def load_metadata(self, filepath: Optional[str] = None):
        """Load metadata from disk"""
        if filepath is None:
            filepath = self.storage_path / "metadata.json"

        try:
            if Path(filepath).exists():
                with open(filepath, "r") as f:
                    self.metadata = json.load(f)
                logger.info(f"Loaded metadata from {filepath}")
        except Exception as e:
            logger.error(f"Error loading metadata: {e}")

    async def update_features(
        self,
        ticker: str,
        features: pl.DataFrame,
        feature_names: List[str],
        upsert: bool = True,
    ) -> int:
        """
        Update existing features or insert new ones

        Args:
            ticker: Stock ticker
            features: DataFrame with updated features
            feature_names: Features to update
            upsert: If True, insert if not exists

        Returns:
            Number of records updated/inserted
        """
        try:
            if upsert:
                # Delete existing
                await self.delete_features(ticker, feature_names)

                # Insert new
                return await self.store_features(ticker, features, feature_names)
            else:
                # Only update existing (not implemented - would need UPDATE query)
                logger.warning("Non-upsert update not yet implemented")
                return 0

        except Exception as e:
            logger.error(f"Error updating features: {e}")
            raise

    async def health_check(self) -> Dict[str, Any]:
        """
        Check health of feature store

        Returns:
            Health status
        """
        try:
            session_factory = get_async_session_factory()

            async with session_factory() as session:
                from sqlalchemy import func

                # Count total features
                query = select(func.count(Feature.feature_name))
                result = await session.execute(query)
                total_features = result.scalar()

                # Count unique tickers
                query = select(func.count(func.distinct(Feature.ticker)))
                result = await session.execute(query)
                unique_tickers = result.scalar()

                return {
                    "status": "healthy",
                    "total_features": total_features,
                    "unique_tickers": unique_tickers,
                    "cache_size": len(self.cache),
                    "storage_path": str(self.storage_path),
                }

        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {"status": "unhealthy", "error": str(e)}


# Global feature store instance
_feature_store: Optional[FeatureStore] = None


def get_feature_store() -> FeatureStore:
    """Get global feature store instance"""
    global _feature_store
    if _feature_store is None:
        _feature_store = FeatureStore()
    return _feature_store
