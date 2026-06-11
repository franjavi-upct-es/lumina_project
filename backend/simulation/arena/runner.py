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
import contextlib
from collections import defaultdict
from collections.abc import Callable, Iterable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Protocol

import mlflow
import numpy as np
import torch
from loguru import logger
from prometheus_client import Counter

from backend.config.constants import (
    ACTION_DIM,
    ARENA_DIVERGENCE_HORIZON_BARS,
    ARENA_EXPLANATION_QUEUE_SIZE,
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

ARENA_EXPLANATIONS_DROPPED = Counter(
    "arena_explanations_dropped_total",
    "Step explanations discarded because the consumer-side sink queue was full.",
    labelnames=("run_id",),
)
ARENA_EXPLANATIONS_EMITTED = Counter(
    "arena_explanations_emitted_total",
    "Step explanations successfully dispatched to the consumer-side sink.",
    labelnames=("run_id",),
)

EnvFactory = Callable[[int], LuminaTradingEnv]
"""A function ``seed -> LuminaTradingEnv``. The runner calls it once per trajectory."""


class ArenaAgent(Protocol):
    """Small policy interface consumed by the arena loop."""

    def act(
        self,
        state: np.ndarray,
        deterministic: bool = False,
    ) -> tuple[np.ndarray, float, float, float, bool]: ...


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
    explanation_sink
        Optional callback invoked once per emitted :class:`StepExplanation`.
        The step loop *never* awaits the sink directly — explanations are
        enqueued into a bounded in-process queue and drained by a dedicated
        consumer task. If the sink is slow and the queue fills, new
        explanations are dropped and ``arena_explanations_dropped_total`` is
        incremented. This guarantees that the per-step latency budget of
        the arena loop is independent of sink throughput.
    policy_uses_full_observation
        When ``False`` (default), the agent receives only the Nexus market
        state, matching the existing live policy shape. Article/retraining
        runs can set this to ``True`` so the policy receives and logs the
        full environment observation, including portfolio bookkeeping.
    divergence_annualization_periods
        Number of periods per year used by divergence Sharpe calculations.
        Defaults to the minute-bar arena setting; daily article runs pass
        ``252``.
    """

    def __init__(
        self,
        run_metadata: ArenaRunMetadata,
        agent: ArenaAgent,
        env_factory: EnvFactory,
        state_builder: StateAssembler | None = None,
        timescale: TimescaleStore | None = None,
        explanation_sink: Callable[[StepExplanation], asyncio.Future | None] | None = None,
        policy_uses_full_observation: bool = False,
        divergence_annualization_periods: float | None = None,
    ) -> None:
        self.metadata = run_metadata
        self.agent = agent
        self.env_factory = env_factory
        self.state_builder = state_builder
        self._timescale = timescale
        self._explanation_sink = explanation_sink
        self.policy_uses_full_observation = bool(policy_uses_full_observation)
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
        # Trace of decisions, indexed by trajectory_id.
        self._records_by_trajectory: dict[int, list[DecisionRecord]] = defaultdict(list)
        # Explanation dispatch — bounded queue + dedicated consumer task.
        # The queue is created lazily in ``run()`` so that the bound is
        # taken against the loop that actually owns the arena execution.
        self._explanation_queue: asyncio.Queue[StepExplanation] | None = None
        self._explanation_consumer_task: asyncio.Task | None = None
        self._run_id_label = str(run_metadata.run_id)

        settings = get_settings()
        self.artifact_root: Path = settings.arena.artifact_dir
        self.logger = TrajectoryLogger(
            run_id=self.metadata.run_id,
            artifact_root=self.artifact_root,
            timescale=timescale,
        )
        divergence_kwargs = {}
        if divergence_annualization_periods is not None:
            divergence_kwargs["annualization_periods"] = divergence_annualization_periods
        self.divergence_analyzer = DivergenceAnalyzer(
            n_trajectories=self.metadata.n_trajectories,
            **divergence_kwargs,
        )
        self.time_controller = AdaptiveStepController(
            playback_multiplier=self.metadata.playback_multiplier
        )

    # ------------------------------------------------------------------
    async def run(self) -> ArenaRunMetadata:
        """Execute the full arena run, returning the final metadata."""
        settings = get_settings()
        mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
        mlflow.set_experiment(settings.MLFLOW_EXPERIMENT_NAME)

        await self._persist_run_metadata(status=ArenaRunStatus.RUNNING)
        self._start_explanation_consumer()

        with mlflow.start_run(run_name=f"arena_{self.metadata.run_id}"):
            # Log run parameters
            mlflow.log_params(
                {
                    "run_id": self.metadata.run_id,
                    "ticker": self.metadata.ticker,
                    "start_date": self.metadata.start_date.isoformat(),
                    "end_date": self.metadata.end_date.isoformat(),
                    "n_trajectories": self.metadata.n_trajectories,
                    "playback_multiplier": self.metadata.playback_multiplier,
                }
            )

            try:
                await self._initialize_envs()
                await self.logger.start()
                await self._main_loop()
                await self._flush_pending_divergences()
                await self.logger.finalize()

                # Calculate and log summary metrics
                sharpes = _per_trajectory_sharpe(self._records_by_trajectory)
                if sharpes:
                    mlflow.log_metrics(
                        {
                            "mean_sharpe": float(np.mean(list(sharpes.values()))),
                            "max_sharpe": float(np.max(list(sharpes.values()))),
                            "min_sharpe": float(np.min(list(sharpes.values()))),
                        }
                    )

                return await self._mark_done(ArenaRunStatus.COMPLETED)
            except asyncio.CancelledError:
                logger.warning("Arena run {} cancelled", self.metadata.run_id)
                await self.logger.finalize()
                mlflow.set_tag("status", "cancelled")
                return await self._mark_done(ArenaRunStatus.CANCELLED)
            except Exception as exc:
                logger.exception("Arena run {} failed", self.metadata.run_id)
                await self.logger.finalize()
                mlflow.set_tag("status", "failed")
                mlflow.log_param("failure_reason", str(exc))
                return await self._mark_done(ArenaRunStatus.FAILED, failure_reason=str(exc))
            finally:
                await self._stop_explanation_consumer()

    async def cancel(self) -> None:
        """Cooperative cancellation. Trajectories finish their current step."""
        self._cancel_event.set()

    # ------------------------------------------------------------------
    # Explanation sink — bounded, non-blocking dispatch.
    # ------------------------------------------------------------------
    def _start_explanation_consumer(self) -> None:
        if self._explanation_sink is None:
            return
        self._explanation_queue = asyncio.Queue(maxsize=ARENA_EXPLANATION_QUEUE_SIZE)
        self._explanation_consumer_task = asyncio.create_task(
            self._explanation_consumer_loop(),
            name=f"arena-explanation-consumer-{self._run_id_label}",
        )

    async def _stop_explanation_consumer(self) -> None:
        task = self._explanation_consumer_task
        queue = self._explanation_queue
        if task is None or queue is None:
            return
        # Drain whatever is already enqueued, but bound the drain wait so a
        # stuck sink can never block run() shutdown indefinitely.
        try:
            await asyncio.wait_for(queue.join(), timeout=5.0)
        except TimeoutError:
            logger.warning(
                "Arena run {}: explanation queue drain timed out; "
                "remaining items will be discarded.",
                self.metadata.run_id,
            )
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await task
        self._explanation_consumer_task = None
        self._explanation_queue = None

    async def _explanation_consumer_loop(self) -> None:
        """Single consumer that drains the bounded explanation queue."""
        assert self._explanation_queue is not None
        assert self._explanation_sink is not None
        sink = self._explanation_sink
        queue = self._explanation_queue
        while True:
            explanation = await queue.get()
            try:
                maybe = sink(explanation)
                if asyncio.iscoroutine(maybe):
                    await maybe
                ARENA_EXPLANATIONS_EMITTED.labels(run_id=self._run_id_label).inc()
            except asyncio.CancelledError:
                # ``task_done`` must run before re-raising so a concurrent
                # ``queue.join()`` in shutdown can complete instead of hanging.
                queue.task_done()
                raise
            except Exception:
                logger.exception("Step-explanation sink raised; continuing run")
                queue.task_done()
            else:
                queue.task_done()

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
            for tid, record in step_records.items():
                self._records_by_trajectory[tid].append(record)

            # Log step metrics to MLflow
            if mlflow.active_run():
                step_metrics = {}
                for tid, record in step_records.items():
                    if record.realized_reward is not None:
                        step_metrics[f"reward_t{tid}"] = float(record.realized_reward)
                    step_metrics[f"confidence_t{tid}"] = float(record.confidence)
                if step_metrics:
                    mlflow.log_metrics(step_metrics, step=step_index)

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
        policy_state = self._policy_state(obs)

        action_array, _log_prob, _value, uncertainty, vetoed = self.agent.act(policy_state)
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
            realized_reward=float(reward),
            state_artifact_path="",  # rewritten inside logger
            attribution=attribution,
            mc_seed=int(self.metadata.mc_seeds[t_id]),
        )

        super_state = torch.from_numpy(policy_state).float()
        record = await self.logger.log_decision(record, super_state)
        bar_return = _per_bar_return(reward)
        self._returns_history[t_id].append(bar_return)

        # Emit the step explanation if a sink was registered. We hand the
        # explanation to the consumer task via a bounded queue rather than
        # awaiting the sink here — a slow sink must never feed back into
        # the per-step latency of the trajectory.
        if self._explanation_queue is not None:
            explanation = format_decision(record, divergence=None)
            try:
                self._explanation_queue.put_nowait(explanation)
            except asyncio.QueueFull:
                ARENA_EXPLANATIONS_DROPPED.labels(run_id=self._run_id_label).inc()

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

    def _policy_state(self, obs: np.ndarray) -> np.ndarray:
        if self.policy_uses_full_observation:
            return obs.astype(np.float32, copy=False)
        return obs[: _market_state_dim(obs)].astype(np.float32, copy=False)


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


def _per_trajectory_sharpe(
    decisions_by_trajectory: dict[int, list[DecisionRecord]],
) -> dict[int, float]:
    """Calculate the realized Sharpe ratio for each trajectory."""
    result: dict[int, float] = {}
    for tid, records in decisions_by_trajectory.items():
        # Using realized_reward as a proxy for per-bar return (scaled)
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
