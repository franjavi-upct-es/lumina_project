"""Unit tests for the Timescale storage gateway."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any

import pytest

from backend.data_engine.storage.timescale import TimescaleStore


class FakeConnection:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows
        self.calls: list[tuple[str, tuple[Any, ...]]] = []

    async def fetch(self, query: str, *args: Any) -> list[dict[str, Any]]:
        self.calls.append((query, args))
        return self.rows


def _row() -> dict[str, Any]:
    return {
        "time": datetime(2024, 1, 2, 14, 30, tzinfo=UTC),
        "open": Decimal("100.10"),
        "high": Decimal("101.20"),
        "low": Decimal("99.90"),
        "close": Decimal("100.80"),
        "volume": Decimal("1234"),
        "vwap": Decimal("100.70"),
        "trade_count": Decimal("42"),
    }


def _patch_connection(
    monkeypatch: pytest.MonkeyPatch,
    store: TimescaleStore,
    conn: FakeConnection,
) -> None:
    @asynccontextmanager
    async def fake_conn() -> AsyncIterator[FakeConnection]:
        yield conn

    monkeypatch.setattr(store, "_conn", fake_conn)


@pytest.mark.asyncio
async def test_get_historical_window_rows_returns_plain_dicts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = TimescaleStore()
    conn = FakeConnection([_row()])
    _patch_connection(monkeypatch, store, conn)
    start = datetime(2024, 1, 2, 14, 0, tzinfo=UTC)
    end = datetime(2024, 1, 2, 15, 0, tzinfo=UTC)

    rows = await store.get_historical_window_rows("AAPL", start, end, freq="5m")

    assert rows == [_row()]
    query, args = conn.calls[0]
    assert "time_bucket" in query
    assert args == (timedelta(minutes=5), "AAPL", start, end)


@pytest.mark.asyncio
async def test_get_historical_window_preserves_polars_dataframe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = TimescaleStore()
    conn = FakeConnection([_row()])
    _patch_connection(monkeypatch, store, conn)
    start = datetime(2024, 1, 2, 14, 0, tzinfo=UTC)
    end = datetime(2024, 1, 2, 15, 0, tzinfo=UTC)

    df = await store.get_historical_window("AAPL", start, end, freq="1m")

    assert df.to_dicts() == [_row()]
    assert conn.calls[0][1] == (timedelta(minutes=1), "AAPL", start, end)
