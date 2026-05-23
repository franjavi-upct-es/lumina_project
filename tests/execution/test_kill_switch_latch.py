# tests/execution/test_kill_switch_latch.py
"""Unit tests for the in-process kill-switch latch.

These tests cover the LocalKillSwitch logic in isolation (no Redis
needed). The pub/sub propagation path requires a live Redis instance
and is exercised by the integration suite.
"""

from __future__ import annotations

import pytest

from backend.execution.safety.kill_switch import (
    KillSwitchState,
    LocalKillSwitch,
    LocalKillSwitchListener,
)


@pytest.fixture(autouse=True)
def _reset_singleton():
    """Fresh singleton per test so state from previous tests does not leak."""
    LocalKillSwitch._instance = None
    yield
    LocalKillSwitch._instance = None


def test_initial_state_is_not_liquidate() -> None:
    latch = LocalKillSwitch.instance()
    assert latch.is_liquidate() is False


def test_trip_arms_the_latch() -> None:
    latch = LocalKillSwitch.instance()
    latch.trip()
    assert latch.is_liquidate() is True


def test_reset_disarms_the_latch() -> None:
    latch = LocalKillSwitch.instance()
    latch.trip()
    latch.reset()
    assert latch.is_liquidate() is False


def test_singleton_returns_same_instance_across_calls() -> None:
    a = LocalKillSwitch.instance()
    b = LocalKillSwitch.instance()
    assert a is b


def test_listener_apply_arms_only_on_liquidate() -> None:
    """The listener's _apply method must arm the latch for LIQUIDATE_ALL
    and clear it for everything else."""
    latch = LocalKillSwitch.instance()
    # Build a listener with a mock redis — we never start its task here,
    # we only exercise the _apply method directly.
    listener = LocalKillSwitchListener.__new__(LocalKillSwitchListener)
    listener._latch = latch

    listener._apply(KillSwitchState.LIQUIDATE_ALL)
    assert latch.is_liquidate() is True

    listener._apply(KillSwitchState.CLOSE_ONLY)
    assert latch.is_liquidate() is False

    listener._apply(KillSwitchState.LIQUIDATE_ALL)
    assert latch.is_liquidate() is True

    listener._apply(KillSwitchState.NORMAL)
    assert latch.is_liquidate() is False
