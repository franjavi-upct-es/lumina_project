# tests/execution/test_safety_rules.py
"""Per-rule unit tests for the SafetyArbitrator's seven atomic rules.

Each rule is pure — input is a SafetyContext, output is (allow, reason).
The atomic-rule design makes these tests trivially independent of each
other, which is exactly the property a production safety layer should
have.
"""

from __future__ import annotations

import numpy as np
import pytest

from backend.execution.safety.rules import (
    SafetyContext,
    rule_kill_switch,
    rule_loss_streak,
    rule_max_drawdown,
    rule_no_flip_high_uncertainty,
    rule_position_limit,
    rule_size_cap,
    rule_uncertainty,
)


def _ctx(**overrides) -> SafetyContext:
    """Build a benign baseline context that every rule allows."""
    base = SafetyContext(
        proposed_action=np.array([0.2, 0.0, 0.0, 0.0], dtype=np.float32),
        current_position=0.0,
        equity=100_000.0,
        peak_equity=100_000.0,
        uncertainty=0.10,
        kill_switch_state="NORMAL",
    )
    for k, v in overrides.items():
        setattr(base, k, v)
    return base


# --- rule_kill_switch ------------------------------------------------------


def test_kill_switch_liquidate_all_vetoes_everything() -> None:
    allow, reason = rule_kill_switch(_ctx(kill_switch_state="LIQUIDATE_ALL"))
    assert allow is False
    assert "LIQUIDATE_ALL" in reason


def test_kill_switch_normal_permits() -> None:
    allow, _ = rule_kill_switch(_ctx())
    assert allow is True


def test_kill_switch_close_only_allows_position_reduction() -> None:
    """In CLOSE_ONLY mode, |new| < |current| must pass."""
    allow, _ = rule_kill_switch(
        _ctx(
            current_position=0.5,
            proposed_action=np.array([0.2, 0.0, 0.0, 0.0], dtype=np.float32),
            kill_switch_state="CLOSE_ONLY",
        )
    )
    assert allow is True


def test_kill_switch_close_only_rejects_increase() -> None:
    allow, _ = rule_kill_switch(
        _ctx(
            current_position=0.2,
            proposed_action=np.array([0.7, 0.0, 0.0, 0.0], dtype=np.float32),
            kill_switch_state="CLOSE_ONLY",
        )
    )
    assert allow is False


# --- rule_max_drawdown -----------------------------------------------------


def test_max_drawdown_within_limit_permits() -> None:
    allow, _ = rule_max_drawdown(_ctx(equity=90_000.0, peak_equity=100_000.0))
    assert allow is True


def test_max_drawdown_above_limit_vetoes() -> None:
    allow, reason = rule_max_drawdown(_ctx(equity=70_000.0, peak_equity=100_000.0))
    assert allow is False
    assert "Drawdown" in reason


# --- rule_uncertainty ------------------------------------------------------


def test_uncertainty_below_threshold_permits() -> None:
    allow, _ = rule_uncertainty(_ctx(uncertainty=0.5))
    assert allow is True


def test_uncertainty_above_threshold_vetoes() -> None:
    allow, _ = rule_uncertainty(_ctx(uncertainty=0.95))
    assert allow is False


# --- rule_position_limit ---------------------------------------------------


def test_position_limit_within_cap_permits() -> None:
    # direction=0.5, sizing=0 → size_factor=0.5 → target=0.25 < 1.0
    allow, _ = rule_position_limit(
        _ctx(proposed_action=np.array([0.5, 0.0, 0.0, 0.0], dtype=np.float32))
    )
    assert allow is True


# --- rule_no_flip_high_uncertainty -----------------------------------------


def test_no_flip_low_uncertainty_permits_flip() -> None:
    """Below the 0.7 uncertainty bar, flips are allowed."""
    allow, _ = rule_no_flip_high_uncertainty(
        _ctx(
            current_position=0.5,
            proposed_action=np.array([-0.5, 0.0, 0.0, 0.0], dtype=np.float32),
            uncertainty=0.5,
        )
    )
    assert allow is True


def test_no_flip_high_uncertainty_blocks_flip() -> None:
    allow, _ = rule_no_flip_high_uncertainty(
        _ctx(
            current_position=0.5,
            proposed_action=np.array([-0.5, 0.0, 0.0, 0.0], dtype=np.float32),
            uncertainty=0.85,
        )
    )
    assert allow is False


# --- rule_loss_streak ------------------------------------------------------


def test_loss_streak_under_5_permits() -> None:
    allow, _ = rule_loss_streak(_ctx(consecutive_losses=4))
    assert allow is True


def test_loss_streak_at_or_over_5_vetoes() -> None:
    allow, _ = rule_loss_streak(_ctx(consecutive_losses=5))
    assert allow is False


# --- rule_size_cap ---------------------------------------------------------


def test_size_cap_below_limit_permits() -> None:
    # sizing=-1 → size_factor=0 → under any cap
    allow, _ = rule_size_cap(
        _ctx(proposed_action=np.array([0.0, 0.0, -1.0, 0.0], dtype=np.float32))
    )
    assert allow is True


@pytest.mark.parametrize("size_raw,expected_allow", [(-1.0, True), (0.0, True), (1.0, False)])
def test_size_cap_parametrised(size_raw: float, expected_allow: bool) -> None:
    """size_factor = (size_raw + 1) / 2 → -1→0, 0→0.5, 1→1.0; cap = 0.5."""
    allow, _ = rule_size_cap(
        _ctx(proposed_action=np.array([0.0, 0.0, size_raw, 0.0], dtype=np.float32))
    )
    assert allow is expected_allow
