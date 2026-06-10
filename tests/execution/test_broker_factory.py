"""Broker factory selection tests."""

from __future__ import annotations

from backend.config.settings import Settings
from backend.execution.broker.factory import get_broker, reset_broker_singleton
from backend.execution.broker.paper_adapter import PaperBroker


def test_broker_factory_builds_paper_broker() -> None:
    reset_broker_singleton()
    broker = get_broker(Settings(_env_file=None, BROKER_MODE="paper"))
    assert isinstance(broker, PaperBroker)
    reset_broker_singleton()
