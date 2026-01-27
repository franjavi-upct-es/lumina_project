# backend/data_engine/storage/__init__.py
"""
Data Storage Module for Lumina Quant Lab

Provides efficient storage solutions for time-series financial data:

TimescaleAdapter:
- TimescaleDB connection and operations
- Hypertable management
- Efficient time-based queries
- Continuous aggregates

ParquetWriter:
- Columnar file storage with compression
- Partitioned storage by date/ticker
- Efficient batch writes

FeatureStore:
- Feature versioning and lineage
- Point-in-time feature retrieval
- Feature metadata management
- Cache integration

Usage:
    from backend.data_engine.storage import TimescaleAdapter, ParquetWriter

    # Database operations
    adapter = TimescaleAdapter()
    await adapter.insert_price_data(data)

    # File storage
    writer = ParquetWriter("/data/parquet")
    writer.write(data, ticker="AAPL", partition_by="date")
"""

from backend.data_engine.storage.feature_store import FeatureStore
from backend.data_engine.storage.parquet_writer import ParquetWriter
from backend.data_engine.storage.timescale_adapter import TimescaleAdapter

__all__ = [
    "TimescaleAdapter",
    "ParquetWriter",
    "FeatureStore",
]
