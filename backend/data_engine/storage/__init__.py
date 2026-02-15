# backend/data_engine/storage/__init__.py
"""
Storage Module for Lumina V3
============================

Provides adapters for different storage backends:
- TimescaleDB: Time-series optimized PostgreSQL
- Redis: In-memory caching and pub/sub
- Parquet: Compressed columnar file storage

Version: 3.0.0
"""

from backend.data_engine.storage.parquet_writer import ParquetWriter
from backend.data_engine.storage.redis_cache import RedisCache
from backend.data_engine.storage.timescale_adapter import TimescaleAdapter

__all__ = [
    "TimescaleAdapter",
    "RedisCache",
    "ParquetWriter",
]
