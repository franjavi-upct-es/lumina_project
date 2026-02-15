# backend/data_engine/storage/parquet_writer.py
"""
Parquet Writer for V3
====================

Handles writing and reading Parquet files for feature storage.
Provides efficient columnar storage with compression.

Directory Structure:
    /features/
        {ticker}/
            {year}/
                {month}/
                    features_{start}_{end}.parquet

Version: 3.0.0
"""

from datetime import datetime
from pathlib import Path

import polars as pl
from loguru import logger

from backend.config.settings import get_settings

settings = get_settings()


class ParquetWriter:
    """
    Parquet file writer and reader

    Manages partitioned Parquet storage for features.
    """

    def __init__(
        self,
        base_path: str | Path | None = None,
        partition_days: int = 7,
    ):
        """
        Initialize Parquet writer

        Args:
            base_path: Base directory for Parquet files
            partition_days: Days per partition file
        """
        self.base_path = Path(base_path or settings.FEATURE_STORE_PATH)
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.partition_days = partition_days

        logger.info(f"ParquetWriter initialized at {self.base_path}")

    def write(
        self,
        data: pl.DataFrame,
        ticker: str,
        partition_date: datetime,
    ) -> Path:
        """
        Write DataFrame to Parquet file

        Args:
            data: DataFrame to write
            ticker: Asset ticker
            partition_date: Date for partitioning

        Returns:
            Path to written file
        """
        try:
            # Create directory structure
            year = partition_date.year
            month = f"{partition_date.month:02d}"

            dir_path = self.base_path / ticker / str(year) / month
            dir_path.mkdir(parents=True, exist_ok=True)

            # Generate filename
            start_date = partition_date.strftime("%Y%m%d")
            file_path = dir_path / f"features_{start_date}.parquet"

            # Write with compression
            data.write_parquet(
                file_path,
                compression="zstd",
                compression_level=3,
            )

            logger.success(f"Wrote {len(data)} rows to {file_path}")
            return file_path

        except Exception as e:
            logger.error(f"Error writing Parquet: {e}")
            raise

    def read(
        self,
        ticker: str,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        columns: list[str] | None = None,
    ) -> pl.DataFrame | None:
        """
        Read Parquet files for a ticker

        Args:
            ticker: Asset ticker
            start_date: Filter start date
            end_date: Filter end date
            columns: Specific columns to read

        Returns:
            Combined DataFrame or None
        """
        try:
            ticker_path = self.base_path / ticker

            if not ticker_path.exists():
                logger.warning(f"No data found for {ticker}")
                return None

            # Find all Parquet files
            parquet_files = list(ticker_path.rglob("*.parquet"))

            if not parquet_files:
                return None

            # Read and combine
            dfs = []
            for file_path in parquet_files:
                df = pl.read_parquet(file_path, columns=columns)

                # Filter by date if provided
                if start_date or end_date:
                    if "time" in df.columns:
                        if start_date:
                            df = df.filter(pl.col("time") >= start_date)
                        if end_date:
                            df = df.filter(pl.col("time") <= end_date)

                if len(df) > 0:
                    dfs.append(df)

            if not dfs:
                return None

            # Combine and sort
            combined = pl.concat(dfs)
            if "time" in combined.columns:
                combined = combined.sort("time")

            logger.success(f"Read {len(combined)} rows for {ticker}")
            return combined

        except Exception as e:
            logger.error(f"Error reading Parquet: {e}")
            return None

    def delete(self, ticker: str, year: int | None = None) -> bool:
        """
        Delete Parquet files for a ticker

        Args:
            ticker: Asset ticker
            year: Specific year to delete (None = all)

        Returns:
            True if successful
        """
        try:
            ticker_path = self.base_path / ticker

            if not ticker_path.exists():
                return True

            if year:
                year_path = ticker_path / str(year)
                if year_path.exists():
                    import shutil

                    shutil.rmtree(year_path)
                    logger.info(f"Deleted {year} data for {ticker}")
            else:
                import shutil

                shutil.rmtree(ticker_path)
                logger.info(f"Deleted all data for {ticker}")

            return True

        except Exception as e:
            logger.error(f"Error deleting Parquet: {e}")
            return False


# Global instance
_writer: ParquetWriter | None = None


def get_parquet_writer() -> ParquetWriter:
    """Get global Parquet writer"""
    global _writer
    if _writer is None:
        _writer = ParquetWriter()
    return _writer
