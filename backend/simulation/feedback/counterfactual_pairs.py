# backend/simulation/feedback/counterfactual_pairs.py
"""Convert pivotal :class:`DivergencePoint`s into :class:`CounterfactualPair`s.

A counterfactual pair is the training-grade artifact of the arena:
same state, two actions, two outcomes. The XAI panel renders them; the
Behavioral-Cloning trainer consumes them.
"""

from __future__ import annotations

import json
from pathlib import Path
from uuid import UUID

from loguru import logger

from backend.config.constants import (
    ARENA_DIVERGENCE_ACTION_THRESHOLD,
    ARENA_PIVOTAL_SHARPE_DELTA,
)
from backend.simulation.arena.schemas import (
    CounterfactualPair,
    DecisionRecord,
    DivergencePoint,
)


def build_pairs(
    run_id: UUID,
    divergences: list[DivergencePoint],
    decisions_by_trajectory: dict[int, list[DecisionRecord]],
) -> list[CounterfactualPair]:
    """One :class:`CounterfactualPair` per pivotal divergence.

    Both trajectories saw the same input data at the divergence step, so the
    super-state ``.npy`` file is shared. We pull it from the best
    trajectory's record (either side works; the paths are identical).
    """
    pairs: list[CounterfactualPair] = []
    for div in divergences:
        if div.sharpe_delta < ARENA_PIVOTAL_SHARPE_DELTA:
            continue
        best_rec = _find_record(
            decisions_by_trajectory.get(div.best_trajectory_id, []), div.step_index
        )
        worst_rec = _find_record(
            decisions_by_trajectory.get(div.worst_trajectory_id, []), div.step_index
        )
        if best_rec is None or worst_rec is None:
            logger.warning(
                "Skipping divergence at step {}: decision record missing", div.step_index
            )
            continue
        confidence = _confidence_score(
            sharpe_delta=div.sharpe_delta,
            action_l2_distance=div.action_l2_distance,
        )
        pairs.append(
            CounterfactualPair(
                run_id=run_id,
                divergence_step_index=div.step_index,
                sim_timestamp=div.sim_timestamp,
                state_artifact_path=best_rec.state_artifact_path,
                good_action_vector=list(div.best_action_vector),
                bad_action_vector=list(div.worst_action_vector),
                good_outcome_sharpe=div.best_subsequent_sharpe,
                bad_outcome_sharpe=div.worst_subsequent_sharpe,
                confidence_score=confidence,
            )
        )
    return pairs


def write_pairs_jsonl(pairs: list[CounterfactualPair], output_path: Path) -> None:
    """One JSON-serialized pair per line, sorted by confidence_score desc."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sorted_pairs = sorted(pairs, key=lambda p: -p.confidence_score)
    with output_path.open("w", encoding="utf-8") as fh:
        for pair in sorted_pairs:
            fh.write(json.dumps(pair.model_dump(mode="json")))
            fh.write("\n")


# ----------------------------------------------------------------------
def _find_record(records: list[DecisionRecord], step_index: int) -> DecisionRecord | None:
    for rec in records:
        if rec.step_index == step_index:
            return rec
    return None


def _confidence_score(*, sharpe_delta: float, action_l2_distance: float) -> float:
    """Two-term confidence in [0, 1].

    Half mass from the normalised Sharpe delta (above the pivotal threshold
    up to 2.0), half from the normalised action L2 distance (above the
    divergence threshold up to 2.0). Both terms are clamped before
    averaging so a single outlier cannot push the score outside [0, 1].
    """
    sharpe_norm = _normalize(sharpe_delta, lo=ARENA_PIVOTAL_SHARPE_DELTA, hi=2.0)
    l2_norm = _normalize(action_l2_distance, lo=ARENA_DIVERGENCE_ACTION_THRESHOLD, hi=2.0)
    return max(0.0, min(1.0, 0.5 * sharpe_norm + 0.5 * l2_norm))


def _normalize(value: float, *, lo: float, hi: float) -> float:
    if hi <= lo:
        return 1.0 if value >= hi else 0.0
    return max(0.0, min(1.0, (value - lo) / (hi - lo)))
