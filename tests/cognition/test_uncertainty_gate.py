# tests/cognition/test_uncertainty_gate.py
"""Unit tests for UncertaintyGate hysteresis behaviour."""

from __future__ import annotations

import numpy as np

from backend.cognition.agent.uncertainty_gate import (
    UncertaintyGate,
    UncertaintyGateConfig,
)


def test_gate_does_not_veto_during_warmup() -> None:
    cfg = UncertaintyGateConfig(
        warmup_steps=5, threshold_high=0.8, threshold_low=0.5, rolling_window=3
    )
    gate = UncertaintyGate(cfg)
    for _ in range(4):
        assert gate.should_veto(0.99) is False


def test_gate_engages_above_threshold_then_releases_below() -> None:
    cfg = UncertaintyGateConfig(
        warmup_steps=0, threshold_high=0.8, threshold_low=0.5, rolling_window=2
    )
    gate = UncertaintyGate(cfg)
    # Fill the rolling buffer with high values.
    gate.should_veto(0.9)
    assert gate.should_veto(0.9) is True
    # Now feed low values.
    gate.should_veto(0.4)
    assert gate.should_veto(0.3) is False


def test_aggregate_action_samples_returns_mean_per_dim_std() -> None:
    samples = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0, 0.0],
        ]
    )
    u = UncertaintyGate.aggregate_action_samples(samples)
    # Only dim 0 has variance; mean across 4 dims = (std_0 + 0 + 0 + 0)/4
    assert u > 0.2
    assert u < 0.5


def test_defensive_action_is_zeros_with_4_dims_by_default() -> None:
    out = UncertaintyGate.defensive_action()
    assert out.shape == (4,)
    assert np.allclose(out, 0.0)
