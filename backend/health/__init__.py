# backend/health/__init__.py
"""
System Health Check Module for Lumina Quant Lab

Provides comprehensive health checking for all system components.

Usage:
    from backend.health import check_system_health, check_module_imports

    # Check all modules can be imported
    import_status = check_module_imports()

    # Check full system health
    health = await check_system_health()
    print(health)
"""

import asyncio
import importlib
from typing import Any

from loguru import logger

# List of all modules that should be importable
REQUIRED_MODULES = [
    # Config
    "backend.config",
    "backend.config.settings",
    "backend.config.logging_config",
    # API
    "backend.api",
    "backend.api.main",
    "backend.api.dependencies",
    "backend.api.routes",
    "backend.api.routes.data",
    "backend.api.routes.ml",
    "backend.api.routes.backtest",
    "backend.api.routes.portfolio",
    "backend.api.routes.risk",
    # Data Engine
    "backend.data_engine",
    "backend.data_engine.collectors",
    "backend.data_engine.collectors.base_collector",
    "backend.data_engine.collectors.yfinance_collector",
    "backend.data_engine.transformers",
    "backend.data_engine.transformers.feature_engineering",
    # ML Engine
    "backend.ml_engine",
    "backend.ml_engine.models",
    "backend.ml_engine.models.base_model",
    "backend.ml_engine.models.lstm_advanced",
    "backend.ml_engine.models.transformer",
    # Backtesting
    "backend.backtesting",
    "backend.backtesting.event_driven",
    "backend.backtesting.monte_carlo",
    "backend.backtesting.strategies",
    # Workers
    "backend.workers",
    "backend.workers.celery_app",
    "backend.workers.data_tasks",
    "backend.workers.ml_tasks",
    "backend.workers.backtest_tasks",
    # Database
    "backend.db",
    "backend.db.models",
    # Utils
    "backend.utils",
    "backend.utils.validation",
    "backend.utils.formatting",
    "backend.utils.calculations",
    # Core
    "backend.core",
]


def check_module_imports() -> dict[str, bool | str]:
    """
    Check if all required modules can be imported.

    Returns:
        Dictionary of module -> success/error message
    """
    results = {}

    for module_name in REQUIRED_MODULES:
        try:
            importlib.import_module(module_name)
            results[module_name] = True
        except ImportError as e:
            results[module_name] = f"ImportError: {e}"
        except Exception as e:
            results[module_name] = f"Error: {e}"

    return results


def get_import_summary() -> dict[str, Any]:
    """
    Get summary of module import status.

    Returns:
        Summary with counts and details
    """
    results = check_module_imports()

    success = [m for m, status in results.items() if status is True]
    failed = {m: status for m, status in results.items() if status is not True}

    return {
        "total_modules": len(REQUIRED_MODULES),
        "successful": len(success),
        "failed": len(failed),
        "success_rate": len(success) / len(REQUIRED_MODULES) * 100,
        "failed_modules": failed,
    }


async def check_database_connection() -> dict[str, Any]:
    """
    Check database connectivity.

    Returns:
        Database health status
    """
    try:
        from sqlalchemy import text

        from backend.db import get_async_engine

        engine = get_async_engine()
        async with engine.connect() as conn:
            result = await conn.execute(text("SELECT 1"))
            _ = result.scalar()

        return {
            "status": "healthy",
            "connected": True,
            "message": "Database connection successful",
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "connected": False,
            "message": str(e),
        }


async def check_redis_connection() -> dict[str, Any]:
    """
    Check Redis connectivity.

    Returns:
        Redis health status
    """
    try:
        from redis import Redis

        from backend.config import settings

        client = Redis.from_url(
            settings.REDIS_URL,
            decode_responses=True,
            socket_connect_timeout=5,
        )
        client.ping()
        client.close()

        return {
            "status": "healthy",
            "connected": True,
            "message": "Redis connection successful",
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "connected": False,
            "message": str(e),
        }


async def check_celery_connection() -> dict[str, Any]:
    """
    Check Celery broker connectivity.

    Returns:
        Celery health status
    """
    try:
        from backend.workers import celery_app

        conn = celery_app.connection()
        conn.ensure_connection(max_retries=2)
        conn.close()

        return {
            "status": "healthy",
            "connected": True,
            "message": "Celery broker connection successful",
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "connected": False,
            "message": str(e),
        }


async def check_data_collection() -> dict[str, Any]:
    """
    Check data collection capability.

    Returns:
        Data collection health status
    """
    try:
        from datetime import datetime, timedelta

        from backend.data_engine import YFinanceCollector

        collector = YFinanceCollector()

        # Try to collect recent data for a common ticker
        data = await collector.collect_with_retry(
            ticker="AAPL",
            start_date=datetime.now() - timedelta(days=7),
            end_date=datetime.now(),
        )

        if data is not None and data.height > 0:
            return {
                "status": "healthy",
                "working": True,
                "message": f"Collected {data.height} data points",
            }
        else:
            return {
                "status": "degraded",
                "working": False,
                "message": "Data collection returned no data",
            }
    except Exception as e:
        return {
            "status": "unhealthy",
            "working": False,
            "message": str(e),
        }


async def check_feature_engineering() -> dict[str, Any]:
    """
    Check feature engineering capability.

    Returns:
        Feature engineering health status
    """
    try:
        import numpy as np
        import polars as pl

        from backend.data_engine import FeatureEngineer

        # Create sample data
        dates = pl.date_range(
            start=pl.date(2024, 1, 1),
            end=pl.date(2024, 1, 31),
            eager=True,
        )

        data = pl.DataFrame(
            {
                "time": dates,
                "open": np.random.randn(len(dates)) + 100,
                "high": np.random.randn(len(dates)) + 102,
                "low": np.random.randn(len(dates)) + 98,
                "close": np.random.randn(len(dates)) + 100,
                "volume": np.random.randint(1000000, 10000000, len(dates)),
                "ticker": ["TEST"] * len(dates),
            }
        )

        fe = FeatureEngineer()
        enriched = fe.create_all_features(data)
        feature_count = len(fe.get_all_feature_names())

        return {
            "status": "healthy",
            "working": True,
            "message": f"Generated {feature_count} features",
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "working": False,
            "message": str(e),
        }


async def check_ml_models() -> dict[str, Any]:
    """
    Check ML model initialization.

    Returns:
        ML models health status
    """
    try:
        import torch

        from backend.ml_engine import AdvancedLSTM

        model = AdvancedLSTM(
            input_dim=10,
            hidden_dim=32,
            num_layers=1,
        )

        # Check forward pass
        test_input = torch.randn(1, 5, 10)
        with torch.no_grad():
            output = model(test_input)

        return {
            "status": "healthy",
            "working": True,
            "message": "LSTM model initialized successfully",
            "cuda_available": torch.cuda.is_available(),
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "working": False,
            "message": str(e),
        }


async def check_system_health(
    check_external: bool = True,
) -> dict[str, Any]:
    """
    Perform comprehensive system health check.

    Args:
        check_external: Whether to check external services (DB, Redis, etc.)

    Returns:
        Complete system health status
    """
    logger.info("Starting system health check...")

    health = {
        "timestamp": asyncio.get_event_loop().time(),
        "status": "healthy",
        "components": {},
    }

    # Check module imports
    import_summary = get_import_summary()
    health["components"]["imports"] = {
        "status": "healthy" if import_summary["success_rate"] == 100 else "degraded",
        "details": import_summary,
    }

    # Check feature engineering
    fe_health = await check_feature_engineering()
    health["components"]["feature_engineering"] = fe_health

    # Check ML models
    ml_health = await check_ml_models()
    health["components"]["ml_models"] = ml_health

    if check_external:
        # Check database
        db_health = await check_database_connection()
        health["components"]["database"] = db_health

        # Check Redis
        redis_health = await check_redis_connection()
        health["components"]["redis"] = redis_health

        # Check Celery
        celery_health = await check_celery_connection()
        health["components"]["celery"] = celery_health

        # Check data collection
        data_health = await check_data_collection()
        health["components"]["data_collection"] = data_health

    # Determine overall status
    statuses = [comp.get("status", "unknown") for comp in health["components"].values()]

    if all(s == "healthy" for s in statuses):
        health["status"] = "healthy"
    elif any(s == "unhealthy" for s in statuses):
        health["status"] = "unhealthy"
    else:
        health["status"] = "degraded"

    logger.info(f"System health check complete: {health['status']}")

    return health


def print_health_report(health: dict[str, Any]) -> None:
    """
    Print formatted health report.

    Args:
        health: Health check results
    """
    print("\n" + "=" * 60)
    print("LUMINA QUANT LAB - SYSTEM HEALTH REPORT")
    print("=" * 60)

    status_emoji = {
        "healthy": "✅",
        "degraded": "⚠️",
        "unhealthy": "❌",
        "unknown": "❓",
    }

    overall_status = health.get("status", "unknown")
    print(f"\nOverall Status: {status_emoji.get(overall_status, '❓')} {overall_status.upper()}")

    print("\nComponents:")
    print("-" * 40)

    for name, component in health.get("components", {}).items():
        status = component.get("status", "unknown")
        emoji = status_emoji.get(status, "❓")
        message = component.get("message", component.get("details", ""))

        print(f"  {emoji} {name.replace('_', ' ').title()}: {status}")
        if isinstance(message, str) and message:
            print(f"      └── {message}")

    print("\n" + "=" * 60)


__all__ = [
    "check_module_imports",
    "get_import_summary",
    "check_database_connection",
    "check_redis_connection",
    "check_celery_connection",
    "check_data_collection",
    "check_feature_engineering",
    "check_ml_models",
    "check_system_health",
    "print_health_report",
]
