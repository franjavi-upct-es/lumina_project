# backend/data_engine/pipelines/__init__.py
"""
Data Pipelines for Lumina V3
============================

ETL pipelines for data ingestion and cleaning.

Modules:
- ingestion: Async data collection pipelines
- cleaning: Data quality and outlier detection

Version: 3.0.0
"""

from backend.data_engine.pipelines.cleaning import CleaningPipeline
from backend.data_engine.pipelines.ingestion import IngestionPipeline

__all__ = [
    "IngestionPipeline",
    "CleaningPipeline",
]
