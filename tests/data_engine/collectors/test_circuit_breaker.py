# tests/data_engine/collectors/test_circuit_breaker.py
"""Tests for the CircuitBreaker state machine.

We do NOT exercise the Redis-backed monitor loop — that's an
integration concern. The unit-level guarantees are:

* Newly-constructed breakers are closed (open == False).
* ``trip`` opens; ``reset`` closes.
* ``wait_if_open`` returns immediately when closed.
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest

from backend.data_engine.collectors.circuit_breaker import CircuitBreaker


@pytest.fixture
def breaker() -> CircuitBreaker:
    """A breaker over a mocked RedisCache (the breaker only touches Redis
    in the monitor loop, which we don't exercise here)."""
    return CircuitBreaker(MagicMock())


def test_initial_state_is_closed(breaker: CircuitBreaker) -> None:
    assert breaker.is_open is False


def test_trip_opens_the_breaker(breaker: CircuitBreaker) -> None:
    breaker.trip("test reason")
    assert breaker.is_open is True


def test_reset_closes_the_breaker(breaker: CircuitBreaker) -> None:
    breaker.trip("test")
    breaker.reset()
    assert breaker.is_open is False


@pytest.mark.asyncio
async def test_wait_if_open_returns_immediately_when_closed(breaker: CircuitBreaker) -> None:
    """A closed breaker must NOT block the caller."""
    # If wait_if_open ever did block, this test would hang and time-out
    # under pytest-asyncio's default 60s. We use a tight 1s timeout for
    # belt-and-suspenders safety.
    await asyncio.wait_for(breaker.wait_if_open(), timeout=1.0)
