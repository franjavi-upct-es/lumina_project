# backend/simulation/arena/runner.py
"""Spartan Arena top-level orchestrator.

Owns ``N`` independent :class:`LuminaTradingEnv` instances, advances them
step-locked through one shared time controller, wires every decision
through the trajectory logger, the attribution extractor, the
divergence analyzer, and the step explainer. The end-of-run feedback
artifacts (counterfactual pairs, BC dataset, curriculum deltas) are
*not* produced here — they live in ``backend.simulation.feedback``.

Naming-adapter notes
--------------------
The Spartan Arena roadmap names several prerequisite symbols that do
not exist in this codebase. We adapt to the real names (see
``docs/spartan_arena.md`` for the alias table):

    StateBuilder           -> StateAssembler
    PPOContinuousAgent     -> PPOAgent
    UncertaintyEstimator   -> UncertaintyGate (already owned by PPOAgent)
    BaseTradingEnv         -> LuminaTradingEnv
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Iterable
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import torch
from loguru import logger

from backend.cognition.agent.ppo_agent import PPOAgent
from backend.config.constants import (
    ACTION_DIM,
    ARENA_DIVERGENCE_HORIZON_BARS,
)
from backend.config.settings import get_settings
from backend.data_engine.storage.timescale import TimescaleStore
from backend.fusion.state_assembler import StateAssembler
from backend.simulation.arena.divergence_analyzer import DivergenceAnalyzer
from backend.simulation.arena.schemas import (
    ActionKind,
    ArenaRunMetadata,
    ArenaRunStatus,
    AttributionPayload,
    CrossModalWeights,
    DecisionRecord,
    DivergencePoint,
    StepExplanation,
)
from backend.simulation.arena.time_controller import AdaptiveStepController
from backend.simulation.arena.trajectory_logger import TrajectoryLogger
from backend.simulation.environments.base_env import LuminaTradingEnv
from backend.simulation.xai.step_explainer import format_decision

EnvFactory = Callable[[int], LuminaTradingEnv]
"""A function ``seed -> LuminaTradingEnv``. The runner calls it once per trajectory."""


class ArenaRunner:
    """Run an arena execution end-to-end.

    Parameters
    ----------
    run_metadata
        Frozen control record describing the run (ticker, dates, N, seeds).
    agent
        Shared PPO agent. All trajectories query the *same* instance —
        state lives in the env, not the policy.
    env_factory
        Callable that builds a fresh :class:`LuminaTradingEnv` given a
        Monte Carlo seed. The factory is responsible for selecting /
        loading the historical episode and applying the seed-dependent
        adversarial perturbation.
    state_builder
        Optional state assembler with inline encoders attached. When
        provided, ``build()`` produces the per-step attribution payload.
        When absent, attribution is stubbed with uniform cross-modal
        weights — the schema is satisfied but downstream interpretability
        will be coarse.
    timescale
        Optional storage handle. When ``None``, decisions are written only
        to JSONL artifacts (used by unit tests against a sandbox FS).
    """

    def __init__(
        self,
        run_metadata: ArenaRunMetadata,
        agent: PPOAgent,
        env_factory: EnvFactory,
        state_builder: StateAssembler | None = None,
        timescale: TimescaleStore | None = None,
        explanation_sink: Callable[[StepExplanation], asyncio.Future | None] | None = None,
    ) -> None:
        self.metadata = run_metadata
        self.agent = agent
        self.env_factory = env_factory
        self.state_builder = state_builder
        self._timescale = timescale
        self._explanation_sink = explanation_sink
        self._cancel_event = asyncio.Event()
        self._envs: list[LuminaTradingEnv] = []
        self._latest_obs: list[np.ndarray] = []
        self._terminated: list[bool] = []
        self._truncated: list[bool] = []
        self._n_trades: list[int] = []
        # Per-trajectory rolling returns used by the divergence analyzer.
        self._returns_history: list[list[float]] = []
        # Trace of decisions, indexed by step then trajectory_id.
        self._records_by_step: dict[int, dict[int, DecisionRecord]] = {}

        settings = get_settings()
        self.artifact_root: Path = settings.arena.artifact_dir
        self.logger = TrajectoryLogger(
            run_id=self.metadata.run_id,
            artifact_root=self.artifact_root,
            timescale=timescale,
        )
        self.divergence_analyzer = DivergenceAnalyzer(n_trajectories=self.metadata.n_trajectories)
        self.time_controller = AdaptiveStepController(
            playback_multiplier=self.metadata.playback_multiplier
        )

    # ------------------------------------------------------------------
    async def run(self) -> ArenaRunMetadata:
        """Execute the full arena run, returning the final metadata."""
        await self._persist_run_metadata(status=ArenaRunStatus.RUNNING)
        try:
            await self._initialize_envs()
            await self.logger.start()
            await self._main_loop()
            await self._flush_pending_divergences()
            await self.logger.finalize()
            return await self._mark_done(ArenaRunStatus.COMPLETED)
        except asyncio.CancelledError:
            logger.warning("Arena run {} cancelled", self.metadata.run_id)
            await self.logger.finalize()
            return await self._mark_done(ArenaRunStatus.CANCELLED)
        except Exception as exc:
            logger.exception("Arena run {} failed", self.metadata.run_id)
            await self.logger.finalize()
            return await self._mark_done(ArenaRunStatus.FAILED, failure_reason=str(exc))

    async def cancel(self) -> None:
        """Cooperative cancellation. Trajectories finish their current step."""
        self._cancel_event.set()

    # ------------------------------------------------------------------
    async def _initialize_envs(self) -> None:
        self._envs = [self.env_factory(seed) for seed in self.metadata.mc_seeds]
        obs_list: list[np.ndarray] = []
        for env in self._envs:
            obs, _info = env.reset()
            obs_list.append(obs)
        self._latest_obs = obs_list
        self._terminated = [False] * len(self._envs)
        self._truncated = [False] * len(self._envs)
        self._n_trades = [0] * len(self._envs)
        self._returns_history = [[] for _ in self._envs]

    async def _main_loop(self) -> None:
        step_index = 0
        sim_start = self.metadata.start_date
        while not self._cancel_event.is_set():
            if all(self._terminated) or all(self._truncated):
                break
            sim_timestamp = sim_start + timedelta(minutes=step_index)
            async with self.time_controller.step(step_index):
                step_records = await self._run_one_step(step_index, sim_timestamp)
            self._records_by_step[step_index] = step_records
            self.divergence_analyzer.ingest_step(step_index, sim_timestamp, step_records)

            # Finalize the divergence window K steps in the past.
            ready_step = step_index - ARENA_DIVERGENCE_HORIZON_BARS
            if ready_step >= 0:
                await self._finalize_divergence_for(ready_step)

            step_index += 1

    async def _run_one_step(
        self, step_index: int, sim_timestamp: datetime
    ) -> dict[int, DecisionRecord]:
        coroutines = [
            self._step_single_trajectory(t_id, step_index, sim_timestamp)
            for t_id in range(len(self._envs))
            if not (self._terminated[t_id] or self._truncated[t_id])
        ]
        if not coroutines:
            return {}
        results = await asyncio.gather(*coroutines)
        return {r.trajectory_id: r for r in results}

    async def _step_single_trajectory(
        self, t_id: int, step_index: int, sim_timestamp: datetime
    ) -> DecisionRecord:
        env = self._envs[t_id]
        obs = self._latest_obs[t_id]
        # The env appends 4 bookkeeping fields after the market state vector; the
        # policy network expects only the market state, so we slice the
        # observation before handing it to the agent.
        market_state = obs[: _market_state_dim(obs)]

        action_array, _log_prob, _value, uncertainty, vetoed = self.agent.act(market_state)
        action_array = np.asarray(action_array, dtype=np.float32).reshape(-1)
        if action_array.size != ACTION_DIM:
            raise RuntimeError(
                f"agent returned action of shape {action_array.shape}; expected ({ACTION_DIM},)"
            )

        new_obs, reward, terminated, truncated, info = env.step(action_array)
        self._latest_obs[t_id] = new_obs
        self._terminated[t_id] = bool(terminated)
        self._truncated[t_id] = bool(truncated)

        position = float(info.get("position", 0.0))
        prev_position = self._position_before(t_id, action_array)
        action_kind = _classify_action(prev_position, position)
        confidence = max(0.0, min(1.0, 1.0 - float(uncertainty)))
        ohlcv = _ohlcv_from_info(info, sim_timestamp)
        attribution = _stub_attribution()

        record = DecisionRecord(
            run_id=self.metadata.run_id,
            trajectory_id=t_id,
            step_index=step_index,
            sim_timestamp=sim_timestamp,
            wall_timestamp=datetime.now(UTC),
            ticker=self.metadata.ticker,
            ohlcv=ohlcv,
            action_kind=action_kind,
            action_vector=action_array.tolist(),
            confidence=confidence,
            uncertainty=float(min(max(uncertainty, 0.0), 1.0)),
            realized_reward=None,
            state_artifact_path="",  # rewritten inside logger
            attribution=attribution,
            mc_seed=int(self.metadata.mc_seeds[t_id]),
        )

        super_state = torch.from_numpy(obs[: _market_state_dim(obs)]).float()
        record = await self.logger.log_decision(record, super_state)

        # Update reward asynchronously — does not block the step.
        await self.logger.update_realized_reward(record.record_id, float(reward))
        bar_return = _per_bar_return(reward)
        self._returns_history[t_id].append(bar_return)

        # Emit the step explanation if a sink was registered.
        if self._explanation_sink is not None:
            explanation = format_decision(record, divergence=None)
            try:
                maybe = self._explanation_sink(explanation)
                if asyncio.iscoroutine(maybe):
                    await maybe
            except Exception:
                logger.exception("Step-explanation sink raised; continuing run")

        if vetoed:
            logger.debug(
                "Step {} traj {} was vetoed by the uncertainty gate (u={:.3f})",
                step_index,
                t_id,
                uncertainty,
            )
        return record

    async def _finalize_divergence_for(self, step_index: int) -> DivergencePoint | None:
        # Use the last K-bar window of per-trajectory returns ending at the
        # most recent completed step.
        returns_window: dict[int, list[float]] = {}
        for t_id, history in enumerate(self._returns_history):
            tail_start = max(0, step_index)
            tail_end = min(len(history), step_index + ARENA_DIVERGENCE_HORIZON_BARS)
            returns_window[t_id] = list(history[tail_start:tail_end])
        return self.divergence_analyzer.finalize_step(step_index, returns_window)

    async def _flush_pending_divergences(self) -> None:
        for step_index in self.divergence_analyzer.pending_step_indices():
            await self._finalize_divergence_for(step_index)

    # ------------------------------------------------------------------
    async def _persist_run_metadata(
        self, status: ArenaRunStatus, failure_reason: str | None = None
    ) -> None:
        self.metadata = self.metadata.model_copy(
            update={"status": status, "failure_reason": failure_reason}
        )
        if self._timescale is None:
            return
        async with self._timescale._conn() as conn:
            await conn.execute(
                """
                INSERT INTO arena_runs (
                    run_id, status, ticker, start_date, end_date,
                    n_trajectories, mc_seeds, playback_multiplier,
                    created_at, completed_at, failure_reason
                ) VALUES (
                    $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11
                )
                ON CONFLICT (run_id) DO UPDATE SET
                    status = EXCLUDED.status,
                    completed_at = EXCLUDED.completed_at,
                    failure_reason = EXCLUDED.failure_reason
                """,
                self.metadata.run_id,
                self.metadata.status.value,
                self.metadata.ticker,
                self.metadata.start_date,
                self.metadata.end_date,
                self.metadata.n_trajectories,
                list(self.metadata.mc_seeds),
                float(self.metadata.playback_multiplier),
                self.metadata.created_at,
                self.metadata.completed_at,
                self.metadata.failure_reason,
            )

    async def _mark_done(
        self, status: ArenaRunStatus, failure_reason: str | None = None
    ) -> ArenaRunMetadata:
        self.metadata = self.metadata.model_copy(
            update={
                "status": status,
                "failure_reason": failure_reason,
                "completed_at": datetime.now(UTC),
            }
        )
        await self._persist_run_metadata(status, failure_reason)
        return self.metadata

    # ------------------------------------------------------------------
    def _position_before(self, t_id: int, _action: np.ndarray) -> float:
        # The previous position is whatever was active before this step's
        # ``env.step()``. We track it by reading the observation's bookkeeping
        # slot (set by the env's ``_build_obs``).
        obs = self._latest_obs[t_id]
        return float(obs[_market_state_dim(obs)])


# ----------------------------------------------------------------------
# Module-level helpers (small, pure, easy to unit-test independently)
# ----------------------------------------------------------------------
def _market_state_dim(obs: np.ndarray) -> int:
    """Index of the first portfolio bookkeeping slot in the env's obs."""
    # The env appends 4 portfolio fields after the market state vector.
    return obs.shape[0] - 4


def _ohlcv_from_info(info: dict, sim_timestamp: datetime) -> dict[str, float]:
    """Extract OHLCV from env info when present; otherwise produce a stub."""
    # The default LuminaTradingEnv exposes equity/position/pnl, not OHLCV.
    # Downstream consumers only care that the four canonical keys exist.
    return {
        "open": float(info.get("open", info.get("pnl", 0.0))),
        "high": float(info.get("high", info.get("pnl", 0.0))),
        "low": float(info.get("low", info.get("pnl", 0.0))),
        "close": float(info.get("close", info.get("pnl", 0.0))),
        "volume": float(info.get("volume", 0.0)),
    }


def _classify_action(prev_position: float, new_position: float) -> ActionKind:
    """Map the (prev, new) position pair to an :class:`ActionKind` label."""
    delta = new_position - prev_position
    eps = 1e-3
    if abs(new_position) < eps and abs(prev_position) >= eps:
        return ActionKind.SELL if prev_position > 0 else ActionKind.BUY
    if abs(prev_position) < eps and abs(new_position) >= eps:
        return ActionKind.BUY if new_position > 0 else ActionKind.SELL
    if abs(delta) < eps:
        return ActionKind.HOLD
    if abs(new_position) > abs(prev_position):
        return ActionKind.INCREASE
    return ActionKind.REDUCE


def _per_bar_return(reward: float) -> float:
    """Convert the env's scaled reward back into a per-bar return.

    The env reports ``r_t = SCALING * (pnl / capital) - risk_pen``. We
    drop the risk penalty (its contribution to Sharpe is small and the
    sign is right anyway) and divide by SCALING=100 to recover a
    fractional return.
    """
    return float(reward) / 100.0


def _stub_attribution() -> AttributionPayload:
    """Default attribution when no inline encoders are wired.

    Cross-modal weights are uniform (1/3 each), VSN and GAT top-K are
    empty, LLM tokens are ``None``. Schema-compliant but uninformative.
    """
    return AttributionPayload(
        cross_modal=CrossModalWeights(price=1 / 3, news=1 / 3, graph=1 / 3),
        tft_vsn_top=[],
        gat_edges_top=[],
        llm_top_tokens=None,
    )


def make_random_seeds(n: int, *, rng: np.random.Generator | None = None) -> list[int]:
    """Convenience: produce a list of ``n`` reproducibility-safe seeds."""
    rng = rng or np.random.default_rng()
    return [int(s) for s in rng.integers(0, 2**31 - 1, size=n)]


def chunk_iter(iterable: Iterable, size: int) -> Iterable[list]:
    """Split an iterable into chunks of at most ``size`` — used by callers
    that fan out arena runs over a worker pool."""
    bucket: list = []
    for item in iterable:
        bucket.append(item)
        if len(bucket) >= size:
            yield bucket
            bucket = []
    if bucket:
        yield bucket
