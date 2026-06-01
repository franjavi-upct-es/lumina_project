# backend/simulation/arena/divergence_analyzer.py
"""Detect and emit DivergencePoints across trajectories.

For each step, the analyzer buffers all N trajectories' decisions. When
the K-bar horizon has elapsed (K = ARENA_DIVERGENCE_HORIZON_BARS), the
analyzer is given the subsequent per-trajectory returns and emits a
:class:`DivergencePoint` iff the trajectories' actions diverged AND the
subsequent Sharpe ratios diverged (both thresholds in
``backend.config.constants``).
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from datetime import datetime

import numpy as np
from loguru import logger

from backend.config.constants import (
    ARENA_DIVERGENCE_ACTION_THRESHOLD,
    ARENA_DIVERGENCE_HORIZON_BARS,
    ARENA_PIVOTAL_SHARPE_DELTA,
)
from backend.simulation.arena.schemas import DecisionRecord, DivergencePoint

# 1-minute bars * regular US session (390 bars/day) * 252 trading days/year.
_BARS_PER_YEAR: float = 252.0 * 390.0
_SHARPE_ANNUALIZER: float = math.sqrt(_BARS_PER_YEAR)


class DivergenceAnalyzer:
    """Maintains the per-step buffer and emits pivotal divergence records.

    Not thread-safe. The arena runner calls ``ingest_step`` and
    ``finalize_step`` from the same coroutine.
    """

    def __init__(
        self,
        n_trajectories: int,
        annualization_periods: float = _BARS_PER_YEAR,
    ) -> None:
        if n_trajectories < 2:
            raise ValueError(
                f"DivergenceAnalyzer needs at least 2 trajectories, got {n_trajectories}"
            )
        self.n_trajectories = n_trajectories
        self._annualizer = math.sqrt(max(float(annualization_periods), 1.0))
        self._buffer: dict[int, dict[int, DecisionRecord]] = {}
        self._sim_timestamps: dict[int, datetime] = {}
        self._emitted: list[DivergencePoint] = []
        self.horizon = ARENA_DIVERGENCE_HORIZON_BARS

    def ingest_step(
        self,
        step_index: int,
        sim_timestamp: datetime,
        decisions: dict[int, DecisionRecord],
    ) -> None:
        """Stash one step's decisions, keyed by trajectory_id."""
        self._buffer[step_index] = dict(decisions)
        self._sim_timestamps[step_index] = sim_timestamp

    def finalize_step(
        self,
        step_index: int,
        subsequent_returns: dict[int, list[float]],
    ) -> DivergencePoint | None:
        """Decide whether to emit a divergence and pop the step from the buffer.

        Returns the new :class:`DivergencePoint` when both thresholds are
        crossed, else ``None``. Always pops the step from the internal
        buffer so memory stays bounded.
        """
        decisions = self._buffer.pop(step_index, None)
        sim_timestamp = self._sim_timestamps.pop(step_index, None)
        if decisions is None or sim_timestamp is None:
            return None
        if len(decisions) < 2:
            return None

        # --- Pairwise max L2 distance between action vectors ---------------
        traj_ids = sorted(decisions.keys())
        actions = np.array([decisions[t].action_vector for t in traj_ids], dtype=np.float64)
        max_l2 = 0.0
        for i in range(len(traj_ids)):
            for j in range(i + 1, len(traj_ids)):
                d = float(np.linalg.norm(actions[i] - actions[j]))
                if d > max_l2:
                    max_l2 = d

        # --- Per-trajectory Sharpe over the horizon window -----------------
        sharpe_by_traj: dict[int, float] = {}
        for tid in traj_ids:
            returns = subsequent_returns.get(tid, [])
            sharpe_by_traj[tid] = _sharpe_ratio(returns, annualizer=self._annualizer)

        best_tid = max(sharpe_by_traj, key=lambda t: sharpe_by_traj[t])
        worst_tid = min(sharpe_by_traj, key=lambda t: sharpe_by_traj[t])
        sharpe_delta = sharpe_by_traj[best_tid] - sharpe_by_traj[worst_tid]

        if max_l2 < ARENA_DIVERGENCE_ACTION_THRESHOLD:
            return None
        if sharpe_delta < ARENA_PIVOTAL_SHARPE_DELTA:
            return None

        # The best/worst pair's *own* L2 distance — the spec stores their pair
        # rather than the global maximum, so the counterfactual training pair
        # later compares like-for-like.
        bw_l2 = float(
            np.linalg.norm(
                np.array(decisions[best_tid].action_vector)
                - np.array(decisions[worst_tid].action_vector)
            )
        )

        point = DivergencePoint(
            run_id=decisions[best_tid].run_id,
            step_index=step_index,
            sim_timestamp=sim_timestamp,
            best_trajectory_id=best_tid,
            worst_trajectory_id=worst_tid,
            best_action_vector=list(decisions[best_tid].action_vector),
            worst_action_vector=list(decisions[worst_tid].action_vector),
            action_l2_distance=bw_l2,
            best_subsequent_sharpe=sharpe_by_traj[best_tid],
            worst_subsequent_sharpe=sharpe_by_traj[worst_tid],
            sharpe_delta=sharpe_delta,
        )
        self._emitted.append(point)
        logger.info(
            "Divergence detected at step {}: traj {} vs traj {} | L2={:.3f} sharpe_delta={:.3f}",
            step_index,
            best_tid,
            worst_tid,
            bw_l2,
            sharpe_delta,
        )
        return point

    def all_divergences(self) -> list[DivergencePoint]:
        return list(self._emitted)

    def pending_step_indices(self) -> list[int]:
        """Step indices still awaiting horizon-window data — useful at shutdown."""
        return sorted(self._buffer.keys())


def _sharpe_ratio(returns: Iterable[float], annualizer: float = _SHARPE_ANNUALIZER) -> float:
    arr = np.asarray(list(returns), dtype=np.float64)
    if arr.size == 0:
        return 0.0
    std = float(arr.std(ddof=0))
    mean = float(arr.mean())
    return (mean / (std + 1e-9)) * annualizer
