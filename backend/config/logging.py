"""Loguru JSON-structured logging for ELK/Splunk compatibility."""

from __future__ import annotations

import sys

from loguru import logger

from backend.config.settings import get_settings


def configure_logging() -> None:
    """Call once at application startup."""
    settings = get_settings()
    logger.remove()

    if settings.ENVIRONMENT == "development":
        logger.add(
            sys.stdout,
            level="DEBUG",
            format="<green>{time:HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        )
    else:
        logger.add(sys.stdout, level="INFO", serialize=True)
        logger.add(
            "logs/lumina_{time:YYYY-MM-DD}.log",
            level="INFO",
            rotation="100 MB",
            retention="30 days",
            compression="gz",
            serialize=True,
        )
