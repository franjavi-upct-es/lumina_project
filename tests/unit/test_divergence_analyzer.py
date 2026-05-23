# tests/unit/test_divergence_analyzer.py
"""Unit tests for the DivergenceAnalyzer."""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

import pytest

from backend.simulation.arena.divergence_analyzer import DivergenceAnalyzer
from backend.simulation.arena.schemas import (
    ActionKind,
    AttributionPayload,
    CrossModalWeights,
    DecisionRecord,
)


def _rec(traj_id: int, run_id, action: list[float]) -> DecisionRecord:
    return DecisionRecord(
        run_id=run_id,
        trajectory_id=traj_id,
        step_index=0,
        sim_timestamp=datetime.now(UTC),
        wall_timestamp=datetime.now(UTC),
        ticker="AAPL",
        ohlcv={"open": 1, "high": 1, "low": 1, "close": 1, "volume": 0},
        action_kind=ActionKind.BUY,
        action_vector=action,
        confidence=0.6,
        uncertainty=0.4,
        state_artifact_path="x",
        attribution=AttributionPayload(
            cross_modal=CrossModalWeights(price=1 / 3, news=1 / 3, graph=1 / 3),
        ),
        mc_seed=traj_id,
    )


def test_no_divergence_for_aligned_trajectories() -> None:
    run_id = uuid4()
    analyzer = DivergenceAnalyzer(n_trajectories=3)
    decisions = {t: _rec(t, run_id, [0.1, 0.0, 0.0, 0.0]) for t in range(3)}
    analyzer.ingest_step(0, datetime.now(UTC), decisions)
    result = analyzer.finalize_step(0, {t: [0.001] * 30 for t in range(3)})
    assert result is None


def test_divergence_emitted_above_thresholds() -> None:
    run_id = uuid4()
    analyzer = DivergenceAnalyzer(n_trajectories=3)
    decisions = {
        0: _rec(0, run_id, [0.9, 0.0, 0.0, 0.0]),
        1: _rec(1, run_id, [-0.9, 0.0, 0.0, 0.0]),
        2: _rec(2, run_id, [0.5, 0.0, 0.0, 0.0]),
    }
    analyzer.ingest_step(5, datetime.now(UTC), decisions)
    returns = {
        0: [0.002] * 30,
        1: [-0.002] * 30,
        2: [0.0] * 30,
    }
    result = analyzer.finalize_step(5, returns)
    assert result is not None
    assert result.best_trajectory_id == 0
    assert result.worst_trajectory_id == 1
    assert result.action_l2_distance > 0.0


def test_sub_threshold_action_distance_no_emit() -> None:
    run_id = uuid4()
    analyzer = DivergenceAnalyzer(n_trajectories=3)
    decisions = {
        0: _rec(0, run_id, [0.10, 0.0, 0.0, 0.0]),
        1: _rec(1, run_id, [0.12, 0.0, 0.0, 0.0]),
        2: _rec(2, run_id, [0.11, 0.0, 0.0, 0.0]),
    }
    analyzer.ingest_step(1, datetime.now(UTC), decisions)
    returns = {
        0: [0.005] * 30,
        1: [-0.005] * 30,
        2: [0.0] * 30,
    }
    result = analyzer.finalize_step(1, returns)
    assert result is None


def test_sub_threshold_sharpe_delta_no_emit() -> None:
    run_id = uuid4()
    analyzer = DivergenceAnalyzer(n_trajectories=2)
    decisions = {
        0: _rec(0, run_id, [0.9, 0.0, 0.0, 0.0]),
        1: _rec(1, run_id, [-0.9, 0.0, 0.0, 0.0]),
    }
    analyzer.ingest_step(2, datetime.now(UTC), decisions)
    # Use slightly varied but similar-distribution returns so the Sharpe
    # values stay close together (high std, same mean).
    returns = {
        0: [0.001, -0.001] * 15,
        1: [-0.001, 0.001] * 15,
    }
    result = analyzer.finalize_step(2, returns)
    assert result is None


def test_pending_step_indices_tracks_buffer() -> None:
    run_id = uuid4()
    analyzer = DivergenceAnalyzer(n_trajectories=2)
    analyzer.ingest_step(0, datetime.now(UTC), {0: _rec(0, run_id, [0.1, 0, 0, 0])})
    analyzer.ingest_step(1, datetime.now(UTC), {0: _rec(0, run_id, [0.1, 0, 0, 0])})
    assert analyzer.pending_step_indices() == [0, 1]


@pytest.mark.parametrize("n_trajectories", [0, 1])
def test_constructor_rejects_tiny_n(n_trajectories: int) -> None:
    with pytest.raises(ValueError):
        DivergenceAnalyzer(n_trajectories=n_trajectories)
