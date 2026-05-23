# backend/simulation/feedback/curriculum_updater.py
"""Long-horizon feedback for the adversarial curriculum.

Analyses one (or more) arena run(s), classifies each pivotal divergence
into a market regime, and updates the scenario weights consumed by
``backend.simulation.generators.adversarial.AdversarialGenerator``. The
class never imports the generator; it only reads and writes a JSON
config file the generator already knows how to load.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from loguru import logger

from backend.simulation.arena.schemas import (
    ArenaRunMetadata,
    DecisionRecord,
    DivergencePoint,
)

REGIMES: tuple[str, ...] = (
    "high_vol",
    "news_event",
    "trend_reversal",
    "sideways",
    "low_vol",
)
_SMOOTHING_FACTOR: float = 0.2
_WEIGHT_MIN: float = 0.05
_WEIGHT_MAX: float = 0.50


class CurriculumUpdater:
    """Read/modify/write the scenario-weight JSON consumed by the curriculum.

    The config file holds one floating weight per scenario name. When the
    file is missing, the updater seeds it with a uniform distribution
    over :data:`REGIMES`.
    """

    def __init__(self, config_path: Path) -> None:
        self.config_path = Path(config_path)

    def analyze_run(
        self,
        metadata: ArenaRunMetadata,
        decisions_by_trajectory: dict[int, list[DecisionRecord]],
        divergences: list[DivergencePoint],
    ) -> dict[str, float]:
        """Return per-regime weight *deltas* (not absolute weights).

        Positive deltas push the curriculum to over-represent regimes
        where the bottom-K trajectories disproportionately failed.
        Other regimes receive a small negative delta to keep the total
        budget conserved over time.
        """
        if not divergences:
            return {regime: 0.0 for regime in REGIMES}

        sharpes = _per_trajectory_sharpe(decisions_by_trajectory)
        bottom_k_tids = _bottom_k_trajectories(sharpes)
        regime_counts: Counter[str] = Counter()
        regime_bottom_counts: Counter[str] = Counter()

        for divergence in divergences:
            regime = self._classify_regime(divergence, decisions_by_trajectory)
            regime_counts[regime] += 1
            if divergence.worst_trajectory_id in bottom_k_tids:
                regime_bottom_counts[regime] += 1

        deltas: dict[str, float] = {}
        total_pivotal = sum(regime_counts.values())
        for regime in REGIMES:
            if total_pivotal == 0:
                deltas[regime] = 0.0
                continue
            failure_share = regime_bottom_counts[regime] / max(total_pivotal, 1)
            deltas[regime] = 0.05 if failure_share > 0.40 else -0.02

        logger.info(
            "Curriculum deltas for run {}: {}",
            metadata.run_id,
            {k: round(v, 3) for k, v in deltas.items()},
        )
        return deltas

    def commit_deltas(self, deltas: dict[str, float]) -> None:
        """Smooth, clamp, and persist the new weights."""
        current = self._load_or_seed()
        for regime, delta in deltas.items():
            base = current.get(regime, 1.0 / len(REGIMES))
            updated = base + _SMOOTHING_FACTOR * delta
            current[regime] = max(_WEIGHT_MIN, min(_WEIGHT_MAX, updated))
        # Normalise so the weights still describe a probability distribution.
        total = sum(current.values()) or 1.0
        normalised = {k: v / total for k, v in current.items()}
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        with self.config_path.open("w", encoding="utf-8") as fh:
            json.dump(normalised, fh, indent=2, sort_keys=True)

    # ------------------------------------------------------------------
    def _load_or_seed(self) -> dict[str, float]:
        if not self.config_path.exists():
            return {regime: 1.0 / len(REGIMES) for regime in REGIMES}
        try:
            with self.config_path.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
        except Exception as exc:
            logger.warning("Failed to parse curriculum config {} ({})", self.config_path, exc)
            return {regime: 1.0 / len(REGIMES) for regime in REGIMES}
        # Backfill any missing regime.
        for regime in REGIMES:
            data.setdefault(regime, 1.0 / len(REGIMES))
        return data

    @staticmethod
    def _classify_regime(
        divergence: DivergencePoint,
        decisions_by_trajectory: dict[int, list[DecisionRecord]],
    ) -> str:
        # Use the best trajectory's recent OHLCV history as the regime witness.
        records = decisions_by_trajectory.get(divergence.best_trajectory_id, [])
        if not records:
            return "low_vol"
        # Build a tail of close prices ending at the divergence step.
        tail = [
            float(r.ohlcv.get("close", 0.0))
            for r in records
            if r.step_index <= divergence.step_index
        ]
        if len(tail) < 5:
            return "low_vol"

        closes = np.asarray(tail, dtype=np.float64)
        returns = np.diff(closes) / np.maximum(closes[:-1], 1e-9)
        window = returns[-30:] if returns.size >= 30 else returns
        wider_returns = returns[-60:] if returns.size >= 60 else returns

        vol = float(window.std()) if window.size else 0.0
        all_vol_p90 = float(np.quantile(np.abs(returns) + 1e-12, 0.90)) if returns.size else 0.0
        cm_news = _pick_news_weight(records, divergence.step_index)

        if cm_news > 0.45:
            return "news_event"
        if vol > all_vol_p90 and all_vol_p90 > 0.0:
            return "high_vol"
        if _trend_reversed(returns, divergence.step_index):
            return "trend_reversal"
        if float(np.abs(wider_returns.sum())) < 0.005:
            return "sideways"
        return "low_vol"


def _per_trajectory_sharpe(
    decisions_by_trajectory: dict[int, list[DecisionRecord]],
) -> dict[int, float]:
    """Cheap proxy Sharpe — relies on realized rewards if present."""
    result: dict[int, float] = {}
    for tid, records in decisions_by_trajectory.items():
        rewards = np.asarray(
            [r.realized_reward for r in records if r.realized_reward is not None],
            dtype=np.float64,
        )
        if rewards.size < 2:
            result[tid] = 0.0
            continue
        std = float(rewards.std(ddof=0)) or 1e-9
        result[tid] = float(rewards.mean()) / std
    return result


def _bottom_k_trajectories(sharpes: dict[int, float], k: int = 2) -> set[int]:
    if not sharpes:
        return set()
    ordered = sorted(sharpes.items(), key=lambda kv: kv[1])
    return {tid for tid, _s in ordered[: max(1, k)]}


def _pick_news_weight(records: list[DecisionRecord], step_index: int) -> float:
    """Find the news (semantic) cross-modal weight at ``step_index``."""
    for r in records:
        if r.step_index == step_index:
            return float(r.attribution.cross_modal.news)
    return 0.0


def _trend_reversed(returns: np.ndarray, _step_index: int) -> bool:
    if returns.size < 6:
        return False
    pre = returns[-30:-5] if returns.size >= 35 else returns[:-5]
    post = returns[-5:]
    if pre.size == 0 or post.size == 0:
        return False
    return np.sign(pre.mean()) != np.sign(post.mean()) and np.sign(post.mean()) != 0


# Group together for the runner's "feedback pipeline" convenience.
def feedback_pipeline_paths(artifact_root: Path) -> dict[str, Path]:
    """Standard subpaths used by the feedback module within ``artifact_root``."""
    return {
        "pairs_jsonl": Path(artifact_root) / "counterfactual_pairs.jsonl",
        "bc_dataset": Path(artifact_root) / "bc_dataset.npz",
        "curriculum_config": Path(artifact_root) / "curriculum_weights.json",
    }


# Defaultdict helper kept here so feedback consumers don't re-import it.
def empty_decisions_index() -> defaultdict:
    return defaultdict(list)
