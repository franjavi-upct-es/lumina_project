# backend/data_engine/feature_store/offline.py
"""
Offline Feature Store - Cold Storage (TimescaleDB + Parquet)
=============================================================

The "Long-Term Memory" of the Chimera Agent.
Stores historical features for model training, backtesting, and analysis.

Storage Strategy:
    - TimescaleDB: Time-series optimized PostgreSQL for queryable features
    - Parquet Files: Compressed columnar storage for bulk access
    - Hybrid approach: Recent data in DB, older data in Parquet

Directory Structure:
    /features/
        AAPL/
            2024/
                01/
                    features_20240101_20240107.parquet
                    features_20240108_20240114.parquet
                02/
                    features_20240201_20240207.parquet

Author: Lumina Quant Lab
Version: 3.0.0
"""

from datetime import datetime
from pathlib import Path
from typing import Any

import polars as pl
from loguru import logger
from sqlalchemy import and_, delete, func, select

from backend.config.settings import get_settings
from backend.data_engine.feature_store.definitions import FeatureMetadata
from backend.data_engine.storage.parquet_writer import ParquetWriter
from backend.data_engine.storage.timescale_adapter import TimescaleAdapter
from backend.db.models import Feature, bulk_insert_features, get_async_session_factory

settings = get_settings()


class OfflineFeatureStore:
    """
    Cold storage for historical features

    Optimized for training data retrieval and batch analytics.
    """

    def __init__(
        self,
        storage_path: str | None = None,
        use_parquet: bool = True,
        parquet_partition_days: int = 7,
    ):
        """
        Initialize offline feature store

        Args:
            storage_path: Base path for Parquet files
            use_parquet: Enable Parquet storage (in addition to DB)
            parquet_partition_days: Days per Parquet file
        """
        self.storage_path = Path(storage_path or settings.FEATURE_STORE_PATH)
        self.storage_path.mkdir(parents=True, exist_ok=True)

        self.use_parquet = use_parquet
        self.parquet_partition_days = parquet_partition_days

        # Initialize adapters
        self.timescale = TimescaleAdapter()
        self.parquet_writer = ParquetWriter(base_path=self.storage_path)

        logger.info(f"OfflineFeatureStore initialized at {self.storage_path}")

    async def store_features(
        self,
        ticker: str,
        features: pl.DataFrame,
        feature_names: list[str] | None = None,
        categories: dict[str, str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> FeatureMetadata:
        """
        Store features in cold storage (DB + Parquet)

        Args:
            ticker: Asset ticker
            features: DataFrame with features (must have 'time' column)
            feature_names: List of feature column names
            categories: Feature name -> category mapping
            metadata: Additional metadata

        Returns:
            FeatureMetadata object
        """
        try:
            logger.info(f"Storing features for {ticker} (cold storage)")

            if "time" not in features.columns:
                raise ValueError("Features DataFrame must have 'time' column")

            # Determine features to store
            if feature_names is None:
                feature_names = [
                    col
                    for col in features.columns
                    if col not in ["time", "ticker", "source", "collected_at"]
                ]

            # Store in TimescaleDB
            db_count = await self._store_to_timescale(ticker, features, feature_names, categories)

            # Store in Parquet (if enabled)
            parquet_path = None
            if self.use_parquet:
                parquet_path = await self._store_to_parquet(ticker, features)

            # Create metadata
            time_col = features["time"]
            metadata_obj = FeatureMetadata(
                ticker=ticker,
                feature_count=len(feature_names),
                feature_names=feature_names,
                categories=categories or {},
                time_range=(time_col.min(), time_col.max()),
                data_points=len(features),
                missing_ratio=self._calculate_missing_ratio(features, feature_names),
                storage_location=str(parquet_path) if parquet_path else None,
                **(metadata or {}),
            )

            logger.success(
                f"Stored {db_count} records for {ticker} "
                f"({len(feature_names)} features, {len(features)} time steps)"
            )

            return metadata_obj

        except Exception as e:
            logger.error(f"Error storing features: {e}")
            raise

    async def _store_to_timescale(
        self,
        ticker: str,
        features: pl.DataFrame,
        feature_names: list[str],
        categories: dict[str, str] | None,
    ) -> int:
        """Store features in TimescaleDB"""
        try:
            # Convert to pandas for iteration
            features_pd = features.to_pandas()

            # Prepare records
            feature_records = []
            for _, row in features_pd.iterrows():
                for fname in feature_names:
                    if (
                        fname in row.index
                        and not pl.DataFrame({fname: [row[fname]]}).null_count().item() > 0
                    ):
                        category = categories.get(fname, "unknown") if categories else "unknown"

                        feature_records.append(
                            {
                                "time": row["time"],
                                "ticker": ticker,
                                "feature_name": fname,
                                "feature_value": float(row[fname]),
                                "feature_category": category,
                            }
                        )

            if not feature_records:
                logger.warning(f"No valid features to store for {ticker}")
                return 0

            # Bulk insert
            count = await bulk_insert_features(feature_records)
            logger.debug(f"Inserted {count} records into TimescaleDB")
            return count

        except Exception as e:
            logger.error(f"Error storing to TimescaleDB: {e}")
            return 0

    async def _store_to_parquet(
        self,
        ticker: str,
        features: pl.DataFrame,
    ) -> Path | None:
        """Store features in Parquet file"""
        try:
            # Add ticker column if not present
            if "ticker" not in features.columns:
                features = features.with_columns(pl.lit(ticker).alias("ticker"))

            # Determine file path based on date
            min_date = features["time"].min()
            file_path = self.parquet_writer.write(
                data=features,
                ticker=ticker,
                partition_date=min_date,
            )

            logger.debug(f"Wrote Parquet file: {file_path}")
            return file_path

        except Exception as e:
            logger.error(f"Error storing to Parquet: {e}")
            return None

    async def get_features(
        self,
        ticker: str,
        feature_names: list[str] | None = None,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        use_parquet_first: bool = True,
    ) -> pl.DataFrame | None:
        """
        Retrieve features from cold storage

        Args:
            ticker: Asset ticker
            feature_names: Specific features to retrieve
            start_date: Start of time range
            end_date: End of time range
            use_parquet_first: Try Parquet before DB (faster for bulk)

        Returns:
            Polars DataFrame with features or None
        """
        try:
            # Try Parquet first (faster for bulk access)
            if use_parquet_first and self.use_parquet:
                df = await self._read_from_parquet(ticker, start_date, end_date, feature_names)
                if df is not None:
                    return df

            # Fall back to TimescaleDB
            return await self._read_from_timescale(ticker, feature_names, start_date, end_date)

        except Exception as e:
            logger.error(f"Error retrieving features: {e}")
            return None

    async def _read_from_timescale(
        self,
        ticker: str,
        feature_names: list[str] | None,
        start_date: datetime | None,
        end_date: datetime | None,
    ) -> pl.DataFrame | None:
        """Read features from TimescaleDB"""
        try:
            session_factory = get_async_session_factory()

            async with session_factory() as session:
                query = select(Feature).where(Feature.ticker == ticker)

                if feature_names:
                    query = query.where(Feature.feature_name.in_(feature_names))
                if start_date:
                    query = query.where(Feature.time >= start_date)
                if end_date:
                    query = query.where(Feature.time <= end_date)

                query = query.order_by(Feature.time)

                result = await session.execute(query)
                rows = result.scalars().all()

                if not rows:
                    return None

                # Convert to wide format DataFrame
                records = [
                    {
                        "time": row.time,
                        "feature_name": row.feature_name,
                        "feature_value": row.feature_value,
                    }
                    for row in rows
                ]

                df = pl.DataFrame(records)

                # Pivot to wide format
                df_wide = df.pivot(
                    values="feature_value",
                    index="time",
                    columns="feature_name",
                )

                logger.success(f"Retrieved {len(df_wide)} records from TimescaleDB")
                return df_wide

        except Exception as e:
            logger.error(f"Error reading from TimescaleDB: {e}")
            return None

    async def _read_from_parquet(
        self,
        ticker: str,
        start_date: datetime | None,
        end_date: datetime | None,
        feature_names: list[str] | None,
    ) -> pl.DataFrame | None:
        """Read features from Parquet files"""
        try:
            df = self.parquet_writer.read(
                ticker=ticker,
                start_date=start_date,
                end_date=end_date,
                columns=feature_names,
            )

            if df is not None:
                logger.success(f"Retrieved {len(df)} records from Parquet")

            return df

        except Exception as e:
            logger.error(f"Error reading from Parquet: {e}")
            return None

    def _calculate_missing_ratio(
        self,
        features: pl.DataFrame,
        feature_names: list[str],
    ) -> float:
        """Calculate ratio of missing values"""
        try:
            feature_cols = [col for col in feature_names if col in features.columns]
            total_values = len(features) * len(feature_cols)

            if total_values == 0:
                return 0.0

            missing_count = features.select(feature_cols).null_count().sum().item()
            return missing_count / total_values

        except Exception as e:
            logger.error(f"Error calculating missing ratio: {e}")
            return 0.0

    async def delete_features(
        self,
        ticker: str,
        feature_names: list[str] | None = None,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> int:
        """
        Delete features from cold storage

        Args:
            ticker: Asset ticker
            feature_names: Specific features to delete
            start_date: Delete from this date
            end_date: Delete until this date

        Returns:
            Number of records deleted
        """
        try:
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
                logger.success(f"Deleted {count} feature records for {ticker}")
                return count

        except Exception as e:
            logger.error(f"Error deleting features: {e}")
            return 0

    async def get_feature_statistics(
        self,
        ticker: str,
        feature_name: str,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> dict[str, Any] | None:
        """
        Get statistical summary of a feature

        Returns:
            Dict with mean, std, min, max, count
        """
        try:
            session_factory = get_async_session_factory()

            async with session_factory() as session:
                query = select(
                    func.avg(Feature.feature_value).label("mean"),
                    func.stddev(Feature.feature_value).label("std"),
                    func.min(Feature.feature_value).label("min"),
                    func.max(Feature.feature_value).label("max"),
                    func.count(Feature.feature_value).label("count"),
                ).where(
                    and_(
                        Feature.ticker == ticker,
                        Feature.feature_name == feature_name,
                    )
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


# Global instance
_offline_store: OfflineFeatureStore | None = None


def get_offline_store() -> OfflineFeatureStore:
    """Get global offline feature store instance"""
    global _offline_store
    if _offline_store is None:
        _offline_store = OfflineFeatureStore()
    return _offline_store
