# backend/config/logging.py
"""Loguru JSON-structured logging for ELK/Splunk compatibility."""

from __future__ import annotations

import logging
import re
import sys

from loguru import logger

from backend.config.settings import get_settings

# Browsers cannot set headers on the WebSocket handshake, so the API key
# travels as a ``?token=`` / ``?api_key=`` query param. uvicorn's access log
# prints the full request line including that query string, which would leak
# the secret into stdout/log files. Scrub it everywhere it could appear.
_TOKEN_QS = re.compile(r"((?:token|api_key)=)[^&\s\"']+", re.IGNORECASE)


def _redact_tokens(text: str) -> str:
    return _TOKEN_QS.sub(r"\1[REDACTED]", text)


class _RedactTokensFilter(logging.Filter):
    """Strip ``token=``/``api_key=`` values from stdlib log records.

    Attached to ``uvicorn.access``, whose records carry the request path
    (with query string) in ``record.args``. We rewrite both the args and any
    pre-rendered message so the secret never reaches a sink.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        if record.args:
            if isinstance(record.args, tuple):
                record.args = tuple(
                    _redact_tokens(a) if isinstance(a, str) else a for a in record.args
                )
            elif isinstance(record.args, dict):
                record.args = {
                    k: (_redact_tokens(v) if isinstance(v, str) else v)
                    for k, v in record.args.items()
                }
        if isinstance(record.msg, str):
            record.msg = _redact_tokens(record.msg)
        return True


def _install_access_log_redaction() -> None:
    """Idempotently attach the token-redaction filter to uvicorn's loggers."""
    for name in ("uvicorn.access", "uvicorn.error"):
        log = logging.getLogger(name)
        if not any(isinstance(f, _RedactTokensFilter) for f in log.filters):
            log.addFilter(_RedactTokensFilter())


def configure_logging() -> None:
    """Call once at application startup."""
    settings = get_settings()
    logger.remove()
    _install_access_log_redaction()

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
