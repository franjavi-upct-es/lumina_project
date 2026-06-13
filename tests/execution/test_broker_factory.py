"""Broker factory selection tests."""

from __future__ import annotations

import pytest

from backend.config.settings import Settings
from backend.execution.broker import factory as broker_factory
from backend.execution.broker.factory import get_broker, reset_broker_singleton
from backend.execution.broker.paper_adapter import PaperBroker


def test_broker_factory_builds_paper_broker() -> None:
    reset_broker_singleton()
    broker = get_broker(Settings(_env_file=None, BROKER_MODE="paper"))
    assert isinstance(broker, PaperBroker)
    reset_broker_singleton()


def test_broker_factory_builds_alpaca_broker(monkeypatch: pytest.MonkeyPatch) -> None:
    """``BROKER_MODE=alpaca`` must select :class:`AlpacaBroker`.

    The real ``AlpacaBroker.__init__`` instantiates ``TradingClient`` which
    needs API credentials and a live HTTP roundtrip. The factory contract
    is "pick the right class," not "successfully connect," so we stub the
    constructor and assert the picked type only.
    """
    reset_broker_singleton()

    class _StubAlpacaBroker:
        def __init__(self) -> None:
            self.kind = "alpaca"

    # Patch the import-target the factory resolves.
    import backend.execution.broker.alpaca_adapter as alpaca_module

    monkeypatch.setattr(alpaca_module, "AlpacaBroker", _StubAlpacaBroker)
    monkeypatch.setattr(broker_factory, "_BROKER_SINGLETON", None)

    broker = get_broker(Settings(_env_file=None, BROKER_MODE="alpaca"))
    assert isinstance(broker, _StubAlpacaBroker)
    reset_broker_singleton()
