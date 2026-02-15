# backend/data_engine/storage/timescale_adapter.py
"""
TimescaleDB Adapter for V3
==========================

Provides async interface to TimescaleDB for time-series data storage.
Optimized for high-frequency financial data with automatic compression.

Version: 3.0.0
"""

import polars as pl
from loguru import logger
from sqlalchemy import text

from backend.config.settings import get_settings
from backend.db.models import get_async_session_factory

settings = get_settings()


class TimescaleAdapter:
    """
    Async adapter for TimescaleDB operations

    Handles hypertable setup, compression policies, and bulk inserts.
    """

    def __init__(self):
        """Initialize TimescaleDB adapter"""
        self.session_factory = get_async_session_factory()
        logger.info("TimescaleAdapter initialized")

    async def create_hypertable(
        self,
        table_name: str,
        time_column: str = "time",
        chunk_interval: str = "1 day",
    ) -> bool:
        """
        Create hypertable for time-series data

        Args:
            table_name: Name of the table
            time_column: Name of the time column
            chunk_interval: Chunk size (e.g., "1 day", "1 hour")

        Returns:
            True if successful
        """
        try:
            async with self.session_factory() as session:
                query = text(f"""
                    SELECT create_hypertable(
                        '{table_name}',
                        '{time_column}',
                        chunk_time_interval => INTERVAL '{chunk_interval}',
                        if_not_exists => TRUE
                    );
                """)

                await session.execute(query)
                await session.commit()

                logger.success(f"Created hypertable: {table_name}")
                return True

        except Exception as e:
            logger.error(f"Error creating hypertable: {e}")
            return False

    async def enable_compression(
        self,
        table_name: str,
        segment_by_columns: list[str] | None = None,
        order_by_columns: list[str] | None = None,
    ) -> bool:
        """
        Enable TimescaleDB compression

        Args:
            table_name: Name of the hypertable
            segment_by_columns: Columns to segment by (e.g., ["ticker"])
            order_by_columns: Columns to order by (e.g., ["time"])

        Returns:
            True if successful
        """
        try:
            async with self.session_factory() as session:
                # Build compression settings
                segment_by = ", ".join(segment_by_columns) if segment_by_columns else "ticker"
                order_by = ", ".join(order_by_columns) if order_by_columns else "time DESC"

                query = text(f"""
                    ALTER TABLE {table_name} SET (
                        timescaledb.compress,
                        timescaledb.compress_segmentby = '{segment_by}',
                        timescaledb.compress_orderby = '{order_by}'
                    );
                """)

                await session.execute(query)
                await session.commit()

                logger.success(f"Enabled compression for {table_name}")
                return True

        except Exception as e:
            logger.error(f"Error enabling compression: {e}")
            return False

    async def add_compression_policy(
        self,
        table_name: str,
        compress_after: str = "7 days",
    ) -> bool:
        """
        Add automatic compression policy

        Args:
            table_name: Name of the hypertable
            compress_after: Compress chunks older than this (e.g., "7 days")

        Returns:
            True if successful
        """
        try:
            async with self.session_factory() as session:
                query = text(f"""
                    SELECT add_compression_policy(
                        '{table_name}',
                        INTERVAL '{compress_after}'
                    );
                """)

                await session.execute(query)
                await session.commit()

                logger.success(f"Added compression policy for {table_name}")
                return True

        except Exception as e:
            logger.error(f"Error adding compression policy: {e}")
            return False

    async def get_chunk_stats(self, table_name: str) -> pl.DataFrame | None:
        """
        Get hypertable chunk statistics

        Args:
            table_name: Name of the hypertable

        Returns:
            DataFrame with chunk stats or None
        """
        try:
            async with self.session_factory() as session:
                query = text(f"""
                    SELECT
                        chunk_name,
                        range_start,
                        range_end,
                        is_compressed,
                        pg_size_pretty(total_bytes) as size
                    FROM timescaledb_information.chunks
                    WHERE hypertable_name = '{table_name}'
                    ORDER BY range_start DESC
                    LIMIT 50;
                """)

                result = await session.execute(query)
                rows = result.fetchall()

                if rows:
                    data = [dict(row._mapping) for row in rows]
                    return pl.DataFrame(data)

                return None

        except Exception as e:
            logger.error(f"Error getting chunk stats: {e}")
            return None


# Global instance
_adapter: TimescaleAdapter | None = None


def get_timescale_adapter() -> TimescaleAdapter:
    """Get global TimescaleDB adapter"""
    global _adapter
    if _adapter is None:
        _adapter = TimescaleAdapter()
    return _adapter
