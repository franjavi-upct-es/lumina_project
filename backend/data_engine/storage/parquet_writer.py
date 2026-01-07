# backend/data_engine/storage/parquet_writer.py
"""
Parquet file writer for efficient storage of time series data
Optimized for fast writes and efficient querying
"""

from typing import Optional, List, Dict, Any
from datetime import datetime
from pathlib import Path
import shutil

import polars as pl
import pyarrow.parquet as pq
from loguru import logger

from backend.config.settings import get_settings

settings = get_settings()


class ParquetWriter:
    """
    Parquet writer for time series data

    Features:
    - Partitioned storage by ticker and date
    - Compression (snappy/gzip/zstd)
    - Schema validation
    - Incremental writes
    - Efficient querying
    """

    def __init__(
        self,
        base_path: Optional[str] = None,
        compression: str = "snappy",
        partition_cols: Optional[List[str]] = None,
    ):
        """
        Initialize Parquet writer

        Args:
            base_path: Base directory for parquet files
            compression: Compression codec (snappy, gzip, zstd, none)
            partition_cols: Columns to partition by
        """
        self.base_path = Path(base_path or settings.PARQUET_STORAGE_PATH)
        self.base_path.mkdir(parents=True, exist_ok=True)

        self.compression = compression
        self.partition_cols = partition_cols or ["ticker"]

        self.max_file_size_mb = settings.MAX_PARQUET_FILE_SIZE_MB

        logger.info(f"ParquetWriter initialized at {self.base_path} with {compression} compression")

    def write(
        self,
        data: pl.DataFrame,
        dataset_name: str,
        mode: str = "append",
        partition_by: Optional[List[str]] = None,
    ) -> str:
        """
        Write DataFrame to parquet

        Args:
            data: DataFrame to write
            dataset_name: Name of dataset (e.g., 'price_data', 'features')
            mode: 'append' or 'overwrite'
            partition_by: Columns to partition by (overrides default)

        Returns:
            Path where data was written
        """
        try:
            if data.height == 0:
                logger.warning("Attempted to write empty DataFrame")
                return ""

            dataset_path = self.base_path / dataset_name
            dataset_path.mkdir(parents=True, exist_ok=True)

            # Use custom partition columns if provided
            partition_cols = partition_by or self.partition_cols

            logger.info(
                f"Writing {data.height} rows to {dataset_name} (partitioned by {partition_cols})"
            )

            # Convert to PyArrow Table
            arrow_table = data.to_arrow()

            # Write with partitioning
            pq.write_to_dataset(
                arrow_table,
                root_path=str(dataset_path),
                partition_cols=partition_cols,
                compression=self.compression,
                existing_data_behavior="overwrite_or_ignore"
                if mode == "append"
                else "delete_matching",
                max_rows_per_file=1000000,  # 1M rows per file
                max_file_size=self.max_file_size_mb * 1024 * 1024,
            )

            logger.success(f"Successfully wrote data to {dataset_path}")
            return str(dataset_path)

        except Exception as e:
            logger.error(f"Error writing parquet: {e}")
            raise

    def read(
        self,
        dataset_name: str,
        filters: Optional[List[tuple]] = None,
        columns: Optional[List[str]] = None,
    ) -> Optional[pl.DataFrame]:
        """
        Read parquet dataset

        Args:
            dataset_name: Name of dataset
            filters: Filters in format [('column', 'op', value)]
            columns: Specific columns to read

        Returns:
            DataFrame or None
        """
        try:
            dataset_path = self.base_path / dataset_name

            if not dataset_path.exists():
                logger.warning(f"Dataset {dataset_name} not found")
                return None

            logger.info(f"Reading {dataset_name}")

            # Read parquet dataset
            dataset = pq.ParquetDataset(str(dataset_path), filters=filters)

            # Read to Arrow table
            arrow_table = dataset.read(columns=columns)

            # Convert to Polars
            df = pl.from_arrow(arrow_table)

            logger.info(f"Read {df.height} rows from {dataset_name}")
            return df

        except Exception as e:
            logger.error(f"Error reading parquet: {e}")
            return None

    def read_ticker(
        self,
        dataset_name: str,
        ticker: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        columns: Optional[List[str]] = None,
    ) -> Optional[pl.DataFrame]:
        """
        Read data for specific ticker

        Args:
            dataset_name: Dataset name
            ticker: Stock ticker
            start_date: Start date filter
            end_date: End date filter
            columns: Columns to read

        Returns:
            DataFrame or None
        """
        try:
            # Build filters
            filters = [("ticker", "=", ticker)]

            if start_date:
                filters.append(("time", ">=", start_date))
            if end_date:
                filters.append(("time", "<=", end_date))

            return self.read(dataset_name, filters=filters, columns=columns)

        except Exception as e:
            logger.error(f"Error reading ticker data: {e}")
            return None

    def append(
        self,
        data: pl.DataFrame,
        dataset_name: str,
        partition_by: Optional[List[str]] = None,
    ) -> str:
        """
        Append data to existing dataset

        Args:
            data: DataFrame to append
            dataset_name: Dataset name
            partition_by: Partition columns

        Returns:
            Path where data was written
        """
        return self.write(data, dataset_name, mode="append", partition_by=partition_by)

    def overwrite(
        self,
        data: pl.DataFrame,
        dataset_name: str,
        partition_by: Optional[List[str]] = None,
    ) -> str:
        """
        Overwrite existing dataset

        Args:
            data: DataFrame to write
            dataset_name: Dataset name
            partition_by: Partition columns

        Returns:
            Path where data was written
        """
        return self.write(data, dataset_name, mode="overwrite", partition_by=partition_by)

    def delete_dataset(self, dataset_name: str) -> bool:
        """
        Delete entire dataset

        Args:
            dataset_name: Dataset to delete

        Returns:
            True if successful
        """
        try:
            dataset_path = self.base_path / dataset_name

            if dataset_path.exists():
                shutil.rmtree(dataset_path)
                logger.success(f"Deleted dataset {dataset_name}")
                return True
            else:
                logger.warning(f"Dataset {dataset_name} not found")
                return False

        except Exception as e:
            logger.error(f"Error deleting dataset: {e}")
            return False

    def delete_partition(
        self,
        dataset_name: str,
        ticker: Optional[str] = None,
        date: Optional[datetime] = None,
    ) -> bool:
        """
        Delete specific partition

        Args:
            dataset_name: Dataset name
            ticker: Ticker partition to delete
            date: Date partition to delete

        Returns:
            True if successful
        """
        try:
            dataset_path = self.base_path / dataset_name

            # Build partition path
            if ticker:
                partition_path = dataset_path / f"ticker={ticker}"
                if partition_path.exists():
                    shutil.rmtree(partition_path)
                    logger.success(f"Deleted partition for {ticker}")
                    return True

            logger.warning("Partition not found")
            return False

        except Exception as e:
            logger.error(f"Error deleting partition: {e}")
            return False

    def list_datasets(self) -> List[str]:
        """
        List available datasets

        Returns:
            List of dataset names
        """
        try:
            datasets = [
                d.name
                for d in self.base_path.iterdir()
                if d.is_dir() and not d.name.startswith(".")
            ]
            return sorted(datasets)

        except Exception as e:
            logger.error(f"Error listing datasets: {e}")
            return []

    def get_dataset_info(self, dataset_name: str) -> Optional[Dict[str, Any]]:
        """
        Get information about dataset

        Args:
            dataset_name: Dataset name

        Returns:
            Dictionary with dataset info
        """
        try:
            dataset_path = self.base_path / dataset_name

            if not dataset_path.exists():
                return None

            # Count files
            parquet_files = list(dataset_path.rglob("*.parquet"))

            # Calculate total size
            total_size = sum(f.stat().st_size for f in parquet_files)

            # Get partitions
            partitions = [d.name for d in dataset_path.iterdir() if d.is_dir()]

            # Sample schema from first file
            schema = None
            if parquet_files:
                table = pq.read_table(str(parquet_files[0]))
                schema = {field.name: str(field.type) for field in table.schema}

            return {
                "dataset_name": dataset_name,
                "num_files": len(parquet_files),
                "total_size_mb": total_size / (1024 * 1024),
                "partitions": partitions,
                "schema": schema,
            }

        except Exception as e:
            logger.error(f"Error getting dataset info: {e}")
            return None

    def compact_dataset(self, dataset_name: str) -> bool:
        """
        Compact dataset by merging small files

        Args:
            dataset_name: Dataset to compact

        Returns:
            True if successful
        """
        try:
            logger.info(f"Compacting dataset {dataset_name}")

            # Read all data
            data = self.read(dataset_name)

            if data is None or data.height == 0:
                logger.warning("No data to compact")
                return False

            # Delete old dataset
            self.delete_dataset(dataset_name)

            # Write compacted version
            self.write(data, dataset_name, mode="overwrite")

            logger.success(f"Successfully compacted {dataset_name}")
            return True

        except Exception as e:
            logger.error(f"Error compacting dataset: {e}")
            return False

    def validate_schema(self, data: pl.DataFrame, expected_schema: Dict[str, str]) -> bool:
        """
        Validate DataFrame schema

        Args:
            data: DataFrame to validate
            expected_schema: Expected column types

        Returns:
            True if schema matches
        """
        try:
            for col, expected_type in expected_schema.items():
                if col not in data.columns:
                    logger.error(f"Missing column: {col}")
                    return False

                actual_type = str(data[col].dtype)
                if expected_type not in actual_type:
                    logger.warning(
                        f"Type mismatch for {col}: expected {expected_type}, got {actual_type}"
                    )

            return True

        except Exception as e:
            logger.error(f"Error validating schema: {e}")
            return False

    def optimize_for_query(self, dataset_name: str, sort_by: List[str]) -> bool:
        """
        Optimize dataset for specific query pattern

        Args:
            dataset_name: Dataset to optimize
            sort_by: Columns to sort by

        Returns:
            True if successful
        """
        try:
            logger.info(f"Optimizing {dataset_name} for queries on {sort_by}")

            # Read data
            data = self.read(dataset_name)

            if data is None:
                return False

            # Sort data
            data = data.sort(sort_by)

            # Overwrite with sorted data
            self.overwrite(data, dataset_name)

            logger.success(f"Optimized {dataset_name}")
            return True

        except Exception as e:
            logger.error(f"Error optimizing dataset: {e}")
            return False

    def get_statistics(self, dataset_name: str) -> Optional[Dict[str, Any]]:
        """
        Get statistics about dataset

        Args:
            dataset_name: Dataset name

        Returns:
            Dictionary with statistics
        """
        try:
            data = self.read(dataset_name)

            if data is None:
                return None

            stats = {
                "num_rows": data.height,
                "num_columns": data.width,
                "column_names": data.columns,
                "memory_usage_mb": data.estimated_size() / (1024 * 1024),
            }

            # Add column statistics for numeric columns
            numeric_cols = [
                col
                for col in data.columns
                if data[col].dtype in [pl.Int64, pl.Float64, pl.Int32, pl.Float32]
            ]

            if numeric_cols:
                stats["column_stats"] = {}
                for col in numeric_cols[:10]:  # Limit to first 10
                    stats["column_stats"][col] = {
                        "mean": float(data[col].mean()) if data[col].mean() else 0,
                        "std": float(data[col].std()) if data[col].std() else 0,
                        "min": float(data[col].min()) if data[col].min() else 0,
                        "max": float(data[col].max()) if data[col].max() else 0,
                    }

            return stats

        except Exception as e:
            logger.error(f"Error getting statistics: {e}")
            return None


# Global parquet writer instance
_parquet_writer: Optional[ParquetWriter] = None


def get_parquet_writer() -> ParquetWriter:
    """Get global parquet writer instance"""
    global _parquet_writer
    if _parquet_writer is None:
        _parquet_writer = ParquetWriter()
    return _parquet_writer
