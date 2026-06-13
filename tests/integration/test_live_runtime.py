"""Focused tests for the fully-composed live runtime services."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from backend.config.constants import NEXUS_OUTPUT_DIM
from backend.config.settings import Settings
from backend.data_engine.collectors.synthetic_feed import SyntheticFeedService
from backend.data_engine.storage.redis_cache import k_tick_latest
from backend.execution.broker.base import AccountSnapshot, Order, Position
from backend.execution.orchestrator import ExecutionOrchestrator
from backend.execution.safety.arbitrator import SafetyArbitrator, SafetyDecision
from backend.execution.safety.kill_switch import KillSwitchState
from backend.integration.agent_loop import LiveAgentLoop, k_market_state


def test_live_tickers_parse_from_comma_string() -> None:
    settings = Settings(_env_file=None, LIVE_TICKERS="spy, aapl,MSFT")
    assert settings.LIVE_TICKERS == ["SPY", "AAPL", "MSFT"]


class _RedisClient:
    def __init__(self, data: dict[str, Any] | None = None) -> None:
        self.data = data or {}
        self.published: list[tuple[str, str]] = []
        self.history: list[str] = []

    async def get(self, key: str) -> Any:
        return self.data.get(key)

    async def set(self, key: str, value: Any, **_kwargs: Any) -> bool:
        self.data[key] = value
        return True

    async def publish(self, channel: str, payload: str) -> int:
        self.published.append((channel, payload))
        return 1

    async def lpush(self, key: str, value: str) -> int:
        self.history.insert(0, value)
        self.data[key] = list(self.history)
        return len(self.history)

    async def ltrim(self, _key: str, _start: int, _stop: int) -> bool:
        return True


class _Redis:
    def __init__(self, data: dict[str, Any] | None = None) -> None:
        self.client = _RedisClient(data)
        self.ticks: list[tuple[str, dict[str, Any]]] = []

    async def publish_tick(self, ticker: str, tick: dict[str, Any]) -> int:
        self.ticks.append((ticker, tick))
        await self.client.set(k_tick_latest(ticker), json.dumps(tick).encode("utf-8"))
        return 1


class _Timescale:
    def __init__(self) -> None:
        self.rows = []

    async def insert_ohlcv_batch(self, rows) -> int:
        self.rows.extend(rows)
        return len(rows)


@pytest.mark.anyio
async def test_synthetic_feed_bootstraps_history_and_publishes_tick_and_news() -> None:
    settings = Settings(
        _env_file=None,
        LIVE_TICKERS="SPY,AAPL",
        SYNTHETIC_FEED_BOOTSTRAP_DAYS=90,
        SYNTHETIC_FEED_SEED=3,
    )
    redis = _Redis()
    timescale = _Timescale()
    service = SyntheticFeedService(redis, timescale, settings)

    inserted = await service.bootstrap_history()
    await service._publish_ticks(datetime.now(UTC))
    await service._publish_news()

    assert inserted == 180
    assert len(timescale.rows) == 180
    assert {row.ticker for row in timescale.rows} == {"SPY", "AAPL"}
    assert len(redis.ticks) == 2
    assert redis.client.published[-1][0] == "channel:news.global"


@dataclass
class _Broker:
    equity: float = 100_000.0
    cash: float = 100_000.0
    positions: dict[str, Position] = field(default_factory=dict)
    prices: dict[str, float] = field(default_factory=dict)

    def update_price(self, ticker: str, price: float) -> None:
        self.prices[ticker] = price

    async def get_account(self) -> AccountSnapshot:
        return AccountSnapshot(
            equity=self.equity,
            cash=self.cash,
            buying_power=self.cash,
            positions=self.positions,
        )


class _Agent:
    def act(self, _state: np.ndarray, deterministic: bool = False):
        assert deterministic is True
        return np.array([0.2, 0.0, 0.0, 0.0], dtype=np.float32), 0.0, 0.0, 0.1, False


class _Orchestrator:
    def __init__(self) -> None:
        self.latest_price: float | None = None
        self.peak_equity: float | None = None
        self.consecutive_losses: int | None = None

    async def execute(
        self,
        *,
        ticker: str,
        proposed_action: np.ndarray,
        uncertainty: float,
        latest_price: float | None = None,
        peak_equity: float | None = None,
        consecutive_losses: int = 0,
    ):
        self.latest_price = latest_price
        self.peak_equity = peak_equity
        self.consecutive_losses = consecutive_losses
        return SimpleNamespace(
            decision=SafetyDecision(True, 0.1, [], "approved"),
            final_action=0.1,
            orders=[],
        )


@pytest.mark.anyio
async def test_agent_loop_uses_state_market_latest_tick_and_updates_broker() -> None:
    data = {
        k_market_state("SPY"): np.ones(NEXUS_OUTPUT_DIM, dtype=np.float32).tobytes(),
        k_tick_latest("SPY"): json.dumps({"c": 450.0}).encode("utf-8"),
    }
    redis = _Redis(data)
    broker = _Broker()
    orchestrator = _Orchestrator()
    settings = Settings(_env_file=None, LIVE_TICKERS="SPY")
    loop = LiveAgentLoop(
        agent=_Agent(),
        orchestrator=orchestrator,
        broker=broker,
        redis=redis,
        settings=settings,
    )

    await loop._cycle("SPY")

    assert broker.prices["SPY"] == 450.0
    assert orchestrator.latest_price == 450.0
    assert orchestrator.peak_equity == 100_000.0
    payload = json.loads(redis.client.data["agent:last_action"])
    assert payload["ticker"] == "SPY"
    assert payload["final_action"] == 0.1


@pytest.mark.anyio
async def test_agent_loop_skips_when_market_state_missing() -> None:
    """No ``state:market:SPY`` key → cycle must short-circuit cleanly.

    The orchestrator must NOT be called and ``agent:last_action`` must
    NOT be written. This is the contract the state_assembler relies on
    to safely throttle assembly without starving the agent loop.
    """
    redis = _Redis({})  # empty — neither state nor tick keys present
    broker = _Broker()
    orchestrator = _Orchestrator()
    settings = Settings(_env_file=None, LIVE_TICKERS="SPY")
    loop = LiveAgentLoop(
        agent=_Agent(),
        orchestrator=orchestrator,
        broker=broker,
        redis=redis,
        settings=settings,
    )

    await loop._cycle("SPY")

    assert orchestrator.latest_price is None
    assert "agent:last_action" not in redis.client.data
    assert broker.prices == {}


@pytest.mark.anyio
async def test_agent_loop_skips_when_latest_tick_missing() -> None:
    """Market state present but no tick → cycle must skip without executing.

    This is the "stale state" guard: the assembler may have flushed a
    state vector hours ago, but we refuse to act on it without a
    confirmed live price.
    """
    redis = _Redis({k_market_state("SPY"): np.ones(NEXUS_OUTPUT_DIM, dtype=np.float32).tobytes()})
    broker = _Broker()
    orchestrator = _Orchestrator()
    settings = Settings(_env_file=None, LIVE_TICKERS="SPY")
    loop = LiveAgentLoop(
        agent=_Agent(),
        orchestrator=orchestrator,
        broker=broker,
        redis=redis,
        settings=settings,
    )

    await loop._cycle("SPY")

    assert orchestrator.latest_price is None
    assert "agent:last_action" not in redis.client.data


class _KillSwitch:
    async def get_state(self) -> KillSwitchState:
        return KillSwitchState.NORMAL


class _OrderBroker(_Broker):
    def __init__(self) -> None:
        super().__init__()
        self.orders: list[Order] = []

    async def submit_order(self, ticker: str, qty: float, side: str, client_order_id: str) -> Order:
        order = Order(ticker=ticker, qty=qty, side=side, client_order_id=client_order_id)
        self.orders.append(order)
        return order

    async def liquidate_all(self) -> list[Order]:
        return []


@pytest.mark.anyio
async def test_execution_orchestrator_sizes_order_from_latest_price_and_peak_equity() -> None:
    broker = _OrderBroker()
    orchestrator = ExecutionOrchestrator(
        broker=broker,
        arbitrator=SafetyArbitrator(),
        kill_switch=_KillSwitch(),
    )

    result = await orchestrator.execute(
        ticker="SPY",
        proposed_action=np.array([0.5, 0.0, 0.0, 0.0], dtype=np.float32),
        uncertainty=0.1,
        latest_price=100.0,
        peak_equity=100_000.0,
    )

    assert result.orders
    assert result.orders[0].side == "buy"
    assert result.orders[0].qty == pytest.approx(250.0)
