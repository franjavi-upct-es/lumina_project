# tests/unit/test_step_explainer.py
"""Unit tests for the step explainer."""

from __future__ import annotations

import importlib
import sys
from datetime import UTC, datetime
from uuid import uuid4

from backend.simulation.arena.schemas import (
    ActionKind,
    AttributionPayload,
    CrossModalWeights,
    DecisionRecord,
    DivergencePoint,
    GATEdgeCoefficient,
    VSNWeight,
)
from backend.simulation.xai.step_explainer import format_decision


def _base_record(*, uncertainty: float = 0.2, confidence: float = 0.8) -> DecisionRecord:
    return DecisionRecord(
        run_id=uuid4(),
        trajectory_id=0,
        step_index=10,
        sim_timestamp=datetime(2024, 1, 15, 14, 32, tzinfo=UTC),
        wall_timestamp=datetime.now(UTC),
        ticker="AAPL",
        ohlcv={"open": 150, "high": 151, "low": 149, "close": 150.5, "volume": 1e4},
        action_kind=ActionKind.BUY,
        action_vector=[0.4, 0.1, 0.5, 0.2],
        confidence=confidence,
        uncertainty=uncertainty,
        state_artifact_path="x",
        attribution=AttributionPayload(
            cross_modal=CrossModalWeights(price=0.32, news=0.51, graph=0.17),
            tft_vsn_top=[VSNWeight(feature="rsi_14", weight=0.41, state="overbought")],
            gat_edges_top=[
                GATEdgeCoefficient(source_ticker="AAPL", target_ticker="MSFT", coefficient=0.72)
            ],
        ),
        mc_seed=42,
    )


def test_format_decision_without_divergence() -> None:
    record = _base_record()
    exp = format_decision(record)
    assert "VS:" not in exp.text
    assert "BUY" in exp.text
    assert "conf=0.80" in exp.text


def test_format_decision_with_divergence() -> None:
    record = _base_record()
    div = DivergencePoint(
        run_id=record.run_id,
        step_index=record.step_index,
        sim_timestamp=record.sim_timestamp,
        best_trajectory_id=2,
        worst_trajectory_id=0,
        best_action_vector=[-0.6, 0.0, 0.5, 0.0],
        worst_action_vector=list(record.action_vector),
        action_l2_distance=1.05,
        best_subsequent_sharpe=1.2,
        worst_subsequent_sharpe=0.4,
        sharpe_delta=0.84,
    )
    exp = format_decision(record, divergence=div)
    assert "VS:  T2" in exp.text
    assert "SELL" in exp.text or "HOLD" in exp.text  # peer direction was negative


def test_tags_overbought() -> None:
    record = _base_record()
    exp = format_decision(record)
    assert "overbought" in exp.tags


def test_tags_low_confidence() -> None:
    record = _base_record(uncertainty=0.7)
    exp = format_decision(record)
    assert "low_confidence" in exp.tags


def test_no_model_calls() -> None:
    """The step_explainer module must not depend on torch."""
    # Ensure no torch hook is needed; the module imports cleanly without it.
    module = importlib.import_module("backend.simulation.xai.step_explainer")
    referenced = set(dir(module))
    # If torch had been imported, it would appear as a module attribute (we
    # use ``import torch`` not ``from torch import ...`` everywhere).
    assert "torch" not in referenced, "step_explainer must remain torch-free"
    # And the module's own ``__name__`` must already be in sys.modules.
    assert "backend.simulation.xai.step_explainer" in sys.modules
