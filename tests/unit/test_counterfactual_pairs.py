# tests/unit/test_counterfactual_pairs.py
"""Unit tests for the counterfactual-pair builder."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from backend.simulation.arena.schemas import (
    ActionKind,
    AttributionPayload,
    CrossModalWeights,
    DecisionRecord,
    DivergencePoint,
)
from backend.simulation.feedback.counterfactual_pairs import (
    _confidence_score,
    build_pairs,
    write_pairs_jsonl,
)


def _decision(run_id, traj_id: int, step: int) -> DecisionRecord:
    return DecisionRecord(
        run_id=run_id,
        trajectory_id=traj_id,
        step_index=step,
        sim_timestamp=datetime.now(UTC),
        wall_timestamp=datetime.now(UTC),
        ticker="AAPL",
        ohlcv={"open": 1, "high": 1, "low": 1, "close": 1, "volume": 0},
        action_kind=ActionKind.BUY,
        action_vector=[0.1, 0.0, 0.0, 0.0],
        confidence=0.5,
        uncertainty=0.5,
        state_artifact_path=f"r/states/{traj_id}/{step}.npy",
        attribution=AttributionPayload(
            cross_modal=CrossModalWeights(price=1 / 3, news=1 / 3, graph=1 / 3),
        ),
        mc_seed=traj_id,
    )


def _divergence(run_id, step: int, sharpe_delta: float, l2: float) -> DivergencePoint:
    return DivergencePoint(
        run_id=run_id,
        step_index=step,
        sim_timestamp=datetime.now(UTC),
        best_trajectory_id=0,
        worst_trajectory_id=1,
        best_action_vector=[0.5, 0.0, 0.0, 0.0],
        worst_action_vector=[-0.5, 0.0, 0.0, 0.0],
        action_l2_distance=l2,
        best_subsequent_sharpe=sharpe_delta,
        worst_subsequent_sharpe=0.0,
        sharpe_delta=sharpe_delta,
    )


def test_build_pairs_one_per_divergence() -> None:
    run_id = uuid4()
    decisions = {
        0: [_decision(run_id, 0, 10), _decision(run_id, 0, 20)],
        1: [_decision(run_id, 1, 10), _decision(run_id, 1, 20)],
    }
    divergences = [
        _divergence(run_id, 10, sharpe_delta=1.0, l2=1.0),
        _divergence(run_id, 20, sharpe_delta=0.8, l2=0.8),
    ]
    pairs = build_pairs(run_id, divergences, decisions)
    assert len(pairs) == 2


def test_confidence_score_clamped() -> None:
    assert _confidence_score(sharpe_delta=0.0, action_l2_distance=0.0) == 0.0
    assert _confidence_score(sharpe_delta=100.0, action_l2_distance=100.0) == 1.0


def test_write_pairs_jsonl_format(tmp_path: Path) -> None:
    run_id = uuid4()
    decisions = {
        0: [_decision(run_id, 0, 10)],
        1: [_decision(run_id, 1, 10)],
    }
    divergences = [_divergence(run_id, 10, sharpe_delta=1.0, l2=1.0)]
    pairs = build_pairs(run_id, divergences, decisions)
    out = tmp_path / "pairs.jsonl"
    write_pairs_jsonl(pairs, out)
    lines = out.read_text().strip().splitlines()
    assert len(lines) == 1
    parsed = json.loads(lines[0])
    assert "good_action_vector" in parsed
    assert "bad_action_vector" in parsed
    assert "confidence_score" in parsed
