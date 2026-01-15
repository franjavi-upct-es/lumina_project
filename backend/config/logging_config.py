"""
Logging configuration for API and workers.
"""

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

from loguru import logger

from backend.config.settings import get_settings


def setup_logging() -> None:
    """
    Configure standard logging and Loguru sinks.
    """
    settings = get_settings()
    log_level = settings.LOG_LEVEL.upper()

    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]

    if settings.LOG_FILE_PATH:
        log_path = Path(settings.LOG_FILE_PATH)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(
            RotatingFileHandler(
                log_path,
                maxBytes=settings.LOG_MAX_BYTES,
                backupCount=settings.LOG_BACKUP_COUNT,
            )
        )

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)8s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
    )

    logger.remove()
    logger.add(
        sys.stdout,
        level=log_level,
        format="{time:YYYY-MM-DD HH:mm:ss} [{level: <8}] {message}",
        serialize=settings.LOG_FORMAT == "json",
    )

    if settings.LOG_FILE_PATH:
        logger.add(
            str(settings.LOG_FILE_PATH),
            level=log_level,
            rotation=settings.LOG_MAX_BYTES,
            retention=settings.LOG_BACKUP_COUNT,
            serialize=settings.LOG_FORMAT == "json",
        )
