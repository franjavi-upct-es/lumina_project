# backend/data_engine/storage/timescale_adapter.py
"""
TimescaleDB adapter for time series data storage
Optimized for hypertables and continuous aggregates
"""

from typing import List, Dict, Any, Optional
from datetime import datetime

import polars as pl
import pandas as pd
from sqlalchemy import text, select, and_
from loguru import logger

from backend.db.models import (
    PriceData,
    get_async_session_factory,
    bulk_insert_price_data,
)
from backend.config.settings import get_settings

settings = get_settings()


class TimescaleAdapter:
    """
    Adapter for TimescaleDB operations

    Provides:
    - Optimized batch inserts
    - Time-based queries
    - Continuous aggregates
    - Compression management
    - Data retention policies
    """

    def __init__(self):
        self.session_factory = get_async_session_factory()
        logger.info("TimescaleAdapter initialized")

    async def insert_price_data_batch(self, data: pl.DataFrame, ticker: str) -> int:
        """
        Insert price data in batch

        Args:
            data: DataFrame with OHLCV data
            ticker: Stock ticker

        Returns:
            Number of rows inserted
        """
        try:
            data_pd = data.to_pandas()

            records = []
            for _, row in data_pd.iterrows():
                records.append(
                    {
                        "time": row["time"],
                        "ticker": ticker,
                        "open": float(row["open"]) if pd.notna(row["open"]) else None,
                        "high": float(row["high"]) if pd.notna(row["high"]) else None,
                        "low": float(row["low"]) if pd.notna(row["low"]) else None,
                        "close": float(row["close"])
                        if pd.notna(row["close"])
                        else None,
                        "volume": int(row["volume"])
                        if pd.notna(row["volume"])
                        else None,
                        "adjusted_close": float(row["close"])
                        if pd.notna(row["close"])
                        else None,
                        "dividends": 0.0,
                        "stock_splits": 0.0,
                    }
                )

            count = await bulk_insert_price_data(records)
            logger.info(f"Inserted {count} price records for {ticker}")
            return count

        except Exception as e:
            logger.error(f"Error inserting price data: {e}")
            raise

    async def query_price_data(
        self,
        ticker: str,
        start_date: datetime,
        end_date: datetime,
        columns: Optional[List[str]] = None,
    ) -> Optional[pl.DataFrame]:
        """
        Query price data from TimescaleDB

        Args:
            ticker: Stock ticker
            start_date: Start date
            end_date: End date
            columns: Specific columns to query

        Returns:
            DataFrame with results
        """
        try:
            async with self.session_factory() as session:
                query = (
                    select(PriceData)
                    .where(
                        and_(
                            PriceData.ticker == ticker,
                            PriceData.time >= start_date,
                            PriceData.time <= end_date,
                        )
                    )
                    .order_by(PriceData.time)
                )

                result = await session.execute(query)
                rows = result.scalars().all()

                if not rows:
                    return None

                # Convert to DataFrame
                data = []
                for row in rows:
                    data.append(
                        {
                            "time": row.time,
                            "ticker": row.ticker,
                            "open": row.open,
                            "high": row.high,
                            "low": row.low,
                            "close": row.close,
                            "volume": row.volume,
                        }
                    )

                df = pl.DataFrame(data)

                # Filter columns if specified
                if columns:
                    df = df.select([col for col in columns if col in df.columns])

                return df

        except Exception as e:
            logger.error(f"Error querying price data: {e}")
            return None

    async def create_continuous_aggregate(
        self,
        view_name: str,
        source_table: str,
        time_column: str,
        bucket_width: str,
        aggregations: Dict[str, str],
    ) -> bool:
        """
        Create TimescaleDB continuous aggregate

        Args:
            view_name: Name for the materialized view
            source_table: Source hypertable
            time_column: Time column name
            bucket_width: Bucket width (e.g., '1 hour', '1 day')
            aggregations: Dict of {column: aggregation_function}

        Returns:
            True if successful
        """
        try:
            # Build aggregation clauses
            agg_clauses = []
            for col, agg_func in aggregations.items():
                agg_clauses.append(f"{agg_func}({col}) AS {col}_{agg_func}")

            agg_str = ",\n    ".join(agg_clauses)

            sql = f"""
            CREATE MATERIALIZED VIEW IF NOT EXISTS {view_name}
            WITH (timescaledb.continuous) AS
            SELECT
                time_bucket('{bucket_width}', {time_column}) AS bucket,
                ticker,
                {agg_str}
            FROM {source_table}
            GROUP BY bucket, ticker
            WITH NO DATA;
            """

            async with self.session_factory() as session:
                await session.execute(text(sql))
                await session.commit()

            logger.success(f"Created continuous aggregate {view_name}")
            return True

        except Exception as e:
            logger.error(f"Error creating continuous aggregate: {e}")
            return False

    async def refresh_continuous_aggregate(
        self,
        view_name: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> bool:
        """
        Manually refresh continuous aggregate

        Args:
            view_name: Materialized view name
            start_time: Start time for refresh
            end_time: End time for refresh

        Returns:
            True if successful
        """
        try:
            if start_time and end_time:
                sql = f"""
                CALL refresh_continuous_aggregate(
                    '{view_name}',
                    '{start_time.isoformat()}',
                    '{end_time.isoformat()}'
                );
                """
            else:
                sql = f"CALL refresh_continuous_aggregate('{view_name}', NULL, NULL);"

            async with self.session_factory() as session:
                await session.execute(text(sql))
                await session.commit()

            logger.success(f"Refreshed continuous aggregate {view_name}")
            return True

        except Exception as e:
            logger.error(f"Error refreshing continuous aggregate: {e}")
            return False

    async def add_compression_policy(
        self, table_name: str, compress_after: str = "7 days"
    ) -> bool:
        """
        Add compression policy to hypertable

        Args:
            table_name: Hypertable name
            compress_after: Compress data older than this

        Returns:
            True if successful
        """
        try:
            sql = f"""
            SELECT add_compression_policy(
                '{table_name}',
                INTERVAL '{compress_after}'
            );
            """

            async with self.session_factory() as session:
                await session.execute(text(sql))
                await session.commit()

            logger.success(f"Added compression policy to {table_name}")
            return True

        except Exception as e:
            logger.error(f"Error adding compression policy: {e}")
            return False

    async def add_retention_policy(
        self, table_name: str, retain_for: str = "1 year"
    ) -> bool:
        """
        Add data retention policy

        Args:
            table_name: Hypertable name
            retain_for: Keep data for this duration

        Returns:
            True if successful
        """
        try:
            sql = f"""
            SELECT add_retention_policy(
                '{table_name}',
                INTERVAL '{retain_for}'
            );
            """

            async with self.session_factory() as session:
                await session.execute(text(sql))
                await session.commit()

            logger.success(f"Added retention policy to {table_name}")
            return True

        except Exception as e:
            logger.error(f"Error adding retention policy: {e}")
            return False

    async def get_hypertable_stats(self, table_name: str) -> Optional[Dict[str, Any]]:
        """
        Get statistics about hypertable

        Args:
            table_name: Hypertable name

        Returns:
            Dictionary with statistics
        """
        try:
            sql = f"""
            SELECT
                hypertable_size('{table_name}') AS total_size,
                hypertable_index_size('{table_name}') AS index_size,
                hypertable_compression_stats('{table_name}')
            """

            async with self.session_factory() as session:
                result = await session.execute(text(sql))
                row = result.first()

                if row:
                    return {
                        "table_name": table_name,
                        "total_size_mb": row[0] / (1024 * 1024) if row[0] else 0,
                        "index_size_mb": row[1] / (1024 * 1024) if row[1] else 0,
                    }

                return None

        except Exception as e:
            logger.error(f"Error getting hypertable stats: {e}")
            return None

    async def query_aggregated_data(
        self,
        ticker: str,
        start_date: datetime,
        end_date: datetime,
        interval: str = "1 day",
        aggregations: Optional[Dict[str, str]] = None,
    ) -> Optional[pl.DataFrame]:
        """
        Query aggregated time series data

        Args:
            ticker: Stock ticker
            start_date: Start date
            end_date: End date
            interval: Time bucket interval
            aggregations: Custom aggregations

        Returns:
            DataFrame with aggregated data
        """
        try:
            if aggregations is None:
                aggregations = {
                    "open": "FIRST",
                    "high": "MAX",
                    "low": "MIN",
                    "close": "LAST",
                    "volume": "SUM",
                }

            agg_clauses = []
            for col, agg in aggregations.items():
                agg_clauses.append(f"{agg}({col}, time) AS {col}")

            agg_str = ",\n        ".join(agg_clauses)

            sql = f"""
            SELECT
                time_bucket('{interval}', time) AS bucket,
                {agg_str}
            FROM price_data
            WHERE ticker = :ticker
              AND time >= :start_date
              AND time <= :end_date
            GROUP BY bucket
            ORDER BY bucket
            """

            async with self.session_factory() as session:
                result = await session.execute(
                    text(sql),
                    {"ticker": ticker, "start_date": start_date, "end_date": end_date},
                )

                rows = result.fetchall()

                if not rows:
                    return None

                # Convert to DataFrame
                data = [dict(zip(result.keys(), row)) for row in rows]
                return pl.DataFrame(data)

        except Exception as e:
            logger.error(f"Error querying aggregated data: {e}")
            return None


# Global adapter instance
_timescale_adapter: Optional[TimescaleAdapter] = None


def get_timescale_adapter() -> TimescaleAdapter:
    """Get global TimescaleDB adapter instance"""
    global _timescale_adapter
    if _timescale_adapter is None:
        _timescale_adapter = TimescaleAdapter()
    return _timescale_adapter
