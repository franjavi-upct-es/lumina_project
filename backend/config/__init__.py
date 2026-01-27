# backend/config/__init__.py
"""
Configuration module for Lumina Quant Lab

Provides centralized configuration management using Pydantic settings,
including environment variable loading, validation, and type safety.

Usage:
    from backend.config import settings, get_settings

    # Access settings directly
    print(settings.DATABASE_URL)

    # Get cached settings instance
    config = get_settings()
"""

from backend.config.logging_config import (
    get_logger,
    log_api_request,
    log_backtest_run,
    log_context,
    log_exceptions,
    log_function_call,
    log_model_training,
    setup_development_logging,
    setup_logging,
    setup_production_logging,
    timed_operation,
)
from backend.config.settings import Settings, get_settings, settings

__all__ = [
    # Settings
    "Settings",
    "settings",
    "get_settings",
    # Logging
    "get_logger",
    "setup_logging",
    "setup_production_logging",
    "setup_development_logging",
    "log_context",
    "timed_operation",
    "log_function_call",
    "log_exceptions",
    "log_api_request",
    "log_backtest_run",
    "log_model_training",
]
