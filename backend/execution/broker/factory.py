"""Broker selection shared by API and background trading services."""

from __future__ import annotations

from loguru import logger

from backend.config.settings import Settings, get_settings
from backend.execution.broker.base import BaseBroker
from backend.execution.broker.paper_adapter import PaperBroker

_BROKER_SINGLETON: BaseBroker | None = None


def get_broker(settings: Settings | None = None) -> BaseBroker:
    """Return the process-local broker selected by ``BROKER_MODE``."""
    global _BROKER_SINGLETON
    if _BROKER_SINGLETON is not None:
        return _BROKER_SINGLETON

    effective_settings = settings or get_settings()
    if effective_settings.BROKER_MODE == "alpaca":
        from backend.execution.broker.alpaca_adapter import AlpacaBroker

        _BROKER_SINGLETON = AlpacaBroker()
        logger.info("Broker singleton initialised: AlpacaBroker")
    else:
        _BROKER_SINGLETON = PaperBroker()
        logger.info("Broker singleton initialised: PaperBroker")
    return _BROKER_SINGLETON


def reset_broker_singleton() -> None:
    """Clear the cached broker; intended for tests."""
    global _BROKER_SINGLETON
    _BROKER_SINGLETON = None
