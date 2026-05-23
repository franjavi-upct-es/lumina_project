# tests/execution/test_arbitrator.py
"""Unit tests for the SafetyArbitrator with the 4-D action vector."""

from __future__ import annotations

import numpy as np

from backend.execution.safety.arbitrator import SafetyArbitrator
from backend.execution.safety.rules import SafetyContext


def _ctx(
    action_dir: float = 0.3,
    action_size: float = 0.0,
    current_position: float = 0.0,
    equity: float = 100_000.0,
    peak: float = 100_000.0,
    uncertainty: float = 0.20,
    kill: str = "NORMAL",
    losses: int = 0,
) -> SafetyContext:
    """Helper to build a 4-D-action SafetyContext quickly."""
    return SafetyContext(
        proposed_action=np.array([action_dir, 0.0, action_size, 0.0], dtype=np.float32),
        current_position=current_position,
        equity=equity,
        peak_equity=peak,
        uncertainty=uncertainty,
        kill_switch_state=kill,
        consecutive_losses=losses,
    )


def test_arbitrator_approves_normal_action() -> None:
    arb = SafetyArbitrator()
    decision = arb.evaluate(_ctx(action_dir=0.3, action_size=-0.5))
    assert decision.approved


def test_arbitrator_vetoes_on_liquidate() -> None:
    arb = SafetyArbitrator()
    decision = arb.evaluate(_ctx(kill="LIQUIDATE_ALL"))
    assert not decision.approved
    assert "kill switch" in decision.reason.lower()


def test_arbitrator_vetoes_on_drawdown() -> None:
    arb = SafetyArbitrator()
    decision = arb.evaluate(_ctx(equity=75_000.0, peak=100_000.0))
    assert not decision.approved
    assert "drawdown" in decision.reason.lower()


def test_arbitrator_vetoes_on_high_uncertainty() -> None:
    arb = SafetyArbitrator()
    decision = arb.evaluate(_ctx(uncertainty=0.95))
    assert not decision.approved
    assert "uncertainty" in decision.reason.lower()
