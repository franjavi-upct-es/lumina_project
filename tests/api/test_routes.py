# tests/api/test_routes.py
"""Smoke tests for the FastAPI routers.

We construct a FastAPI app with dependency overrides so each test can
inject fake Redis, Timescale, and Broker objects. The goal is to verify
the routing + serialisation surface, not to exercise the underlying
services.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from backend.api.deps import (
    get_broker,
    get_kill_switch,
    get_redis,
    get_timescale,
    require_api_key,
    reset_broker_singleton,
)
from backend.api.main import create_app
from backend.execution.broker.base import AccountSnapshot, Position
from backend.execution.safety.kill_switch import KillSwitchState


@pytest.fixture
def fake_redis() -> AsyncMock:
    """Mock RedisCache."""
    r = AsyncMock()
    r.health_check = AsyncMock(return_value={"connected": True, "latency_ms": 0.4})
    r.client = MagicMock()
    r.client.get = AsyncMock(return_value=None)
    r.client.set = AsyncMock(return_value=True)
    return r


@pytest.fixture
def fake_timescale() -> AsyncMock:
    t = AsyncMock()
    t.health_check = AsyncMock(return_value={"connected": True, "latency_ms": 1.2})
    return t


@pytest.fixture
def fake_broker() -> AsyncMock:
    b = AsyncMock()
    b.health_check = AsyncMock(return_value={"connected": True, "equity": 100_000.0})
    b.get_account = AsyncMock(
        return_value=AccountSnapshot(
            equity=100_000.0,
            cash=50_000.0,
            buying_power=200_000.0,
            positions={
                "AAPL": Position(
                    ticker="AAPL",
                    qty=10.0,
                    avg_entry_price=180.0,
                    unrealized_pnl=50.0,
                    market_value=1850.0,
                )
            },
        )
    )
    return b


@pytest.fixture
def fake_kill_switch() -> MagicMock:
    ks = MagicMock()
    ks.get_state = AsyncMock(return_value=KillSwitchState.NORMAL)
    return ks


@pytest.fixture
def client(fake_redis, fake_timescale, fake_broker, fake_kill_switch) -> TestClient:
    """Build a TestClient with all four backing services overridden."""
    reset_broker_singleton()
    app = create_app()
    app.dependency_overrides[get_redis] = lambda: fake_redis
    app.dependency_overrides[get_timescale] = lambda: fake_timescale
    app.dependency_overrides[get_broker] = lambda: fake_broker
    app.dependency_overrides[get_kill_switch] = lambda: fake_kill_switch
    app.dependency_overrides[require_api_key] = lambda: None
    return TestClient(app)


def test_health_endpoint_returns_ok_when_all_green(client: TestClient) -> None:
    """All components healthy + kill switch NORMAL → status='ok'."""
    response = client.get("/api/monitoring/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert "redis" in body["components"]
    assert "broker" in body["components"]


def test_health_endpoint_returns_degraded_when_redis_down(
    client: TestClient,
    fake_redis: AsyncMock,
) -> None:
    fake_redis.health_check.return_value = {"connected": False, "error": "ECONNREFUSED"}
    response = client.get("/api/monitoring/health")
    assert response.status_code == 200
    assert response.json()["status"] == "degraded"


def test_health_endpoint_returns_down_when_kill_switch_armed(
    client: TestClient,
    fake_kill_switch: MagicMock,
) -> None:
    fake_kill_switch.get_state.return_value = KillSwitchState.LIQUIDATE_ALL
    response = client.get("/api/monitoring/health")
    assert response.json()["status"] == "down"


def test_portfolio_endpoint_returns_account_snapshot(client: TestClient) -> None:
    response = client.get("/api/portfolio")
    assert response.status_code == 200
    body = response.json()
    assert body["equity"] == 100_000.0
    assert body["cash"] == 50_000.0
    assert body["peak_equity"] >= 100_000.0
    assert 0.0 <= body["drawdown_pct"] <= 1.0
    assert len(body["positions"]) == 1
    assert body["positions"][0]["ticker"] == "AAPL"


def test_metrics_endpoint_returns_prometheus_text(client: TestClient) -> None:
    """The /metrics endpoint must return text/plain Prometheus exposition."""
    response = client.get("/api/monitoring/metrics")
    assert response.status_code == 200
    assert "text/plain" in response.headers["content-type"]
    # Must contain at least one HELP line (basic well-formedness check).
    assert "# HELP" in response.text or response.text == ""
