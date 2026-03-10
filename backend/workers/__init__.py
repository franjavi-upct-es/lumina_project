# backend/workers/__init__.py
"""
Workers Module for Lumina V3
============================

Distributed task processing using Celery for:
- Data collection and ingestion
- Feature engineering and storage
- Model training and evaluation
- Backtesting simulations

Task Queues:
- default: General tasks
- data. Data collection and processing
- ml: Machine learning training (GPU)
- backtest: Backtesting simulations
- priority: High-priority urgent tasks

Usage:
    from backend.workers import celery_app
    from backend.workers.data_tasks import collect_ticker_data

    # Synchronous execution
    result = collect_ticker_data.apply(args=["APPL"])

    # Asynchronous execution
    task = collect_ticker_data.delay("APPL")
    result = task.get(timeout=60)

Version: 3.0.0
"""

from backend.workers.celery_app import celery_app, debug_task

__all__ = [
    "celery_app",
    "debug_task",
]

__version__ = "3.0.0"
