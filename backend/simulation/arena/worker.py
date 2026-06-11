"""Redis-backed worker for dashboard-submitted Spartan Arena runs."""

from __future__ import annotations

import asyncio
import contextlib
import json
import signal
import zlib
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

import numpy as np
from loguru import logger

from backend.cognition.agent.ppo_agent import PPOAgent
from backend.cognition.agent.runtime import load_agent, pick_device
from backend.config.constants import NEXUS_OUTPUT_DIM
from backend.config.logging import configure_logging
from backend.config.settings import Settings, get_settings
from backend.data_engine.storage.redis_cache import RedisCache
from backend.data_engine.storage.timescale import TimescaleStore
from backend.simulation.arena.runner import ArenaRunner
from backend.simulation.arena.schemas import (
    ArenaRunMetadata,
    ArenaRunStatus,
    CounterfactualPair,
    DivergencePoint,
    RunSummary,
    StepExplanation,
)
from backend.simulation.environments.base_env import LuminaTradingEnv
from backend.simulation.feedback.counterfactual_pairs import build_pairs
from backend.simulation.feedback.replay_buffer_writer import BCDatasetWriter
from backend.simulation.generators.synthetic_data import jump_diffusion_episode
from backend.simulation.xai.run_summarizer import summarize_run
from backend.simulation.xai.step_explainer import format_decision

_REQUEST_PREFIX = "arena:request:"
_LOCK_PREFIX = "arena:lock:"
_CANCEL_PREFIX = "arena:cancel:"
_PUBSUB_PREFIX = "arena:"
_LOCK_TTL_SECONDS = 12 * 3600


class _SingleEpisode:
    def __init__(self, episode: dict[str, Any]) -> None:
        self.episode = episode

    def __iter__(self):
        yield self.episode


class ArenaWorker:
    """Poll Redis for API-submitted arena jobs and execute them."""

    def __init__(
        self,
        *,
        redis: RedisCache,
        timescale: TimescaleStore,
        agent: PPOAgent,
        settings: Settings,
    ) -> None:
        self.redis = redis
        self.timescale = timescale
        self.agent = agent
        self.settings = settings

    async def run(self, stop_event: asyncio.Event) -> None:
        logger.info("ArenaWorker started")
        while not stop_event.is_set():
            try:
                processed = await self.run_once()
                if processed:
                    logger.info("ArenaWorker processed {} job(s)", processed)
            except Exception:
                logger.exception("ArenaWorker poll cycle failed")
            await asyncio.sleep(self.settings.ARENA_WORKER_POLL_SECONDS)
        logger.info("ArenaWorker stopped")

    async def run_once(self) -> int:
        processed = 0
        async for raw_key in self.redis.client.scan_iter(match=f"{_REQUEST_PREFIX}*", count=100):
            key = _decode(raw_key)
            if await self._process_key(key):
                processed += 1
        return processed

    async def _process_key(self, key: str) -> bool:
        run_id = key.removeprefix(_REQUEST_PREFIX)
        lock_key = _LOCK_PREFIX + run_id
        claimed = await self.redis.client.set(lock_key, b"1", ex=_LOCK_TTL_SECONDS, nx=True)
        if not claimed:
            return False

        try:
            raw = await self.redis.client.get(key)
            if raw is None:
                return False
            metadata = await _load_metadata(self.timescale, UUID(run_id))
            if metadata.status in _TERMINAL_STATUSES:
                await self.redis.client.delete(key)
                return False

            result = await execute_arena(
                metadata=metadata,
                redis=self.redis,
                timescale=self.timescale,
                agent=self.agent,
                settings=self.settings,
            )
            if result.status in _TERMINAL_STATUSES:
                await self.redis.client.delete(key)
            return True
        except Exception as exc:
            logger.exception("Arena run {} failed before runner startup", run_id)
            await _mark_failed(self.timescale, UUID(run_id), str(exc))
            await self.redis.client.delete(key)
            return True
        finally:
            await self.redis.client.delete(lock_key)


_TERMINAL_STATUSES = {
    ArenaRunStatus.COMPLETED,
    ArenaRunStatus.FAILED,
    ArenaRunStatus.CANCELLED,
}


async def execute_arena(
    *,
    metadata: ArenaRunMetadata,
    redis: RedisCache,
    timescale: TimescaleStore,
    agent: PPOAgent,
    settings: Settings,
) -> ArenaRunMetadata:
    """Run one dashboard-submitted Arena job and persist dashboard artifacts."""
    env_factory = await _build_env_factory(
        metadata=metadata,
        timescale=timescale,
        settings=settings,
    )
    runner = ArenaRunner(
        run_metadata=metadata,
        agent=agent,
        env_factory=env_factory,
        timescale=timescale,
        policy_uses_full_observation=True,
    )
    cancel_task = asyncio.create_task(
        _watch_cancel(redis, runner, metadata.run_id),
        name=f"arena-cancel-watch:{metadata.run_id}",
    )
    try:
        result = await runner.run()
    finally:
        cancel_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await cancel_task

    divergences = runner.divergence_analyzer.all_divergences()
    decisions = runner._records_by_trajectory
    pairs = build_pairs(result.run_id, divergences, decisions)
    explanations = [
        format_decision(record, divergence=None)
        for records in decisions.values()
        for record in records
    ]
    summary = summarize_run(result, decisions, divergences)

    await _persist_divergences(timescale, divergences)
    await _persist_pairs(timescale, pairs)
    await _persist_explanations(timescale, result.run_id, explanations)
    await _cache_summary(redis, summary)
    await _publish_terminal(redis, result, summary)
    _write_bc_pairs(settings, pairs)
    logger.success(
        "Arena run {} finished with status={} decisions={} divergences={} pairs={}",
        result.run_id,
        result.status.value,
        sum(len(records) for records in decisions.values()),
        len(divergences),
        len(pairs),
    )
    return result


async def _build_env_factory(
    *,
    metadata: ArenaRunMetadata,
    timescale: TimescaleStore,
    settings: Settings,
) -> Callable[[int], LuminaTradingEnv]:
    start = _ensure_utc(metadata.start_date)
    end = _ensure_utc(metadata.end_date)
    rows = await timescale.get_historical_window_rows(metadata.ticker, start, end, freq="1d")
    base_episode = _episode_from_rows(metadata.ticker, rows)
    if base_episode is None:
        if not settings.ALLOW_SYNTHETIC_SIMULATION_FALLBACK:
            raise RuntimeError(
                f"Not enough historical OHLCV for arena run {metadata.run_id} "
                f"({metadata.ticker}, {start.date()} -> {end.date()}). Run a backfill first or set "
                "ALLOW_SYNTHETIC_SIMULATION_FALLBACK=true for a clearly synthetic smoke test."
            )
        logger.warning(
            "Arena run {} has insufficient historical OHLCV for {}; using synthetic episodes",
            metadata.run_id,
            metadata.ticker,
        )

    def factory(seed: int) -> LuminaTradingEnv:
        if base_episode is None:
            rng = np.random.default_rng(seed)
            episode = jump_diffusion_episode(
                max(3, settings.ARENA_SYNTHETIC_STEPS),
                rng=rng,
            )
            episode["ticker"] = metadata.ticker
        else:
            episode = _perturb_episode(base_episode, seed)
        return LuminaTradingEnv(_SingleEpisode(episode))

    return factory


def _episode_from_rows(ticker: str, rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    if len(rows) < 60:
        return None
    sorted_rows = sorted(rows, key=lambda row: row["time"])
    prices = np.asarray([float(row["close"]) for row in sorted_rows], dtype=np.float32)
    if np.any(prices <= 0):
        return None
    rng = np.random.default_rng(_seed_for("arena-historical", ticker))
    log_returns = np.diff(np.log(prices), prepend=np.log(prices[0]))
    return {
        "prices": prices,
        "market_states": rng.standard_normal((len(prices), NEXUS_OUTPUT_DIM)).astype(np.float32)
        * 0.1,
        "volatility": np.abs(log_returns).astype(np.float32),
        "uncertainties": np.full(len(prices), 0.2, dtype=np.float32),
        "open": np.asarray([float(row["open"]) for row in sorted_rows], dtype=np.float32),
        "high": np.asarray([float(row["high"]) for row in sorted_rows], dtype=np.float32),
        "low": np.asarray([float(row["low"]) for row in sorted_rows], dtype=np.float32),
        "close": prices,
        "volume": np.asarray([float(row["volume"] or 0) for row in sorted_rows], dtype=np.float32),
        "ticker": ticker,
        "synthetic": False,
    }


def _perturb_episode(base: dict[str, Any], seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    prices = np.asarray(base["prices"], dtype=np.float32)
    log_returns = np.diff(np.log(prices), prepend=np.log(prices[0]))
    scale = np.maximum(np.asarray(base["volatility"], dtype=np.float32), 1e-4)
    shocks = rng.normal(0.0, scale * 0.05, size=len(prices)).astype(np.float32)
    perturbed_returns = log_returns + shocks
    perturbed_prices = float(prices[0]) * np.exp(np.cumsum(perturbed_returns))
    episode = dict(base)
    episode["prices"] = perturbed_prices.astype(np.float32)
    episode["close"] = episode["prices"]
    episode["market_states"] = (
        np.asarray(base["market_states"], dtype=np.float32)
        + rng.standard_normal(np.asarray(base["market_states"]).shape).astype(np.float32) * 0.01
    )
    episode["synthetic"] = False
    return episode


async def _watch_cancel(redis: RedisCache, runner: ArenaRunner, run_id: UUID) -> None:
    cancel_key = _CANCEL_PREFIX + str(run_id)
    while True:
        if await redis.client.get(cancel_key):
            logger.warning("Arena run {} received cancel request", run_id)
            await runner.cancel()
            return
        await asyncio.sleep(0.5)


async def _load_metadata(timescale: TimescaleStore, run_id: UUID) -> ArenaRunMetadata:
    async with timescale._conn() as conn:
        row = await conn.fetchrow(
            """
            SELECT run_id, status, ticker, start_date, end_date,
                   n_trajectories, mc_seeds, playback_multiplier,
                   created_at, completed_at, failure_reason
            FROM arena_runs WHERE run_id = $1
            """,
            run_id,
        )
    if row is None:
        raise ValueError(f"arena run {run_id} not found")
    return ArenaRunMetadata(
        run_id=row["run_id"],
        status=ArenaRunStatus(row["status"]),
        ticker=row["ticker"],
        start_date=row["start_date"],
        end_date=row["end_date"],
        n_trajectories=row["n_trajectories"],
        mc_seeds=list(row["mc_seeds"]),
        playback_multiplier=float(row["playback_multiplier"]),
        created_at=row["created_at"],
        completed_at=row["completed_at"],
        failure_reason=row["failure_reason"],
    )


async def _mark_failed(timescale: TimescaleStore, run_id: UUID, reason: str) -> None:
    async with timescale._conn() as conn:
        await conn.execute(
            """
            UPDATE arena_runs
            SET status = $2, completed_at = $3, failure_reason = $4
            WHERE run_id = $1
            """,
            run_id,
            ArenaRunStatus.FAILED.value,
            datetime.now(UTC),
            reason,
        )


async def _persist_divergences(
    timescale: TimescaleStore,
    divergences: list[DivergencePoint],
) -> None:
    if not divergences:
        return
    records = [
        (
            d.run_id,
            d.step_index,
            d.sim_timestamp,
            d.best_trajectory_id,
            d.worst_trajectory_id,
            list(d.best_action_vector),
            list(d.worst_action_vector),
            float(d.action_l2_distance),
            float(d.best_subsequent_sharpe),
            float(d.worst_subsequent_sharpe),
            float(d.sharpe_delta),
        )
        for d in divergences
    ]
    async with timescale._conn() as conn:
        await conn.executemany(
            """
            INSERT INTO arena_divergence_points (
                run_id, step_index, sim_timestamp,
                best_trajectory_id, worst_trajectory_id,
                best_action_vector, worst_action_vector,
                action_l2_distance, best_subsequent_sharpe,
                worst_subsequent_sharpe, sharpe_delta
            ) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11)
            ON CONFLICT (run_id, step_index, sim_timestamp) DO NOTHING
            """,
            records,
        )


async def _persist_pairs(timescale: TimescaleStore, pairs: list[CounterfactualPair]) -> None:
    if not pairs:
        return
    records = [
        (
            p.pair_id,
            p.run_id,
            p.divergence_step_index,
            p.sim_timestamp,
            p.state_artifact_path,
            list(p.good_action_vector),
            list(p.bad_action_vector),
            float(p.good_outcome_sharpe),
            float(p.bad_outcome_sharpe),
            float(p.confidence_score),
        )
        for p in pairs
    ]
    async with timescale._conn() as conn:
        await conn.executemany(
            """
            INSERT INTO arena_counterfactual_pairs (
                pair_id, run_id, divergence_step_index, sim_timestamp,
                state_artifact_path, good_action_vector, bad_action_vector,
                good_outcome_sharpe, bad_outcome_sharpe, confidence_score
            ) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10)
            ON CONFLICT (pair_id) DO NOTHING
            """,
            records,
        )


async def _persist_explanations(
    timescale: TimescaleStore,
    run_id: UUID,
    explanations: list[StepExplanation],
) -> None:
    if not explanations:
        return
    records = [
        (
            e.record_id,
            run_id,
            e.text,
            list(e.tags),
        )
        for e in explanations
    ]
    async with timescale._conn() as conn:
        await conn.executemany(
            """
            INSERT INTO arena_step_explanations (record_id, run_id, text, tags)
            VALUES ($1,$2,$3,$4)
            ON CONFLICT (record_id) DO UPDATE SET
                text = EXCLUDED.text,
                tags = EXCLUDED.tags
            """,
            records,
        )


async def _cache_summary(redis: RedisCache, summary: RunSummary) -> None:
    await redis.client.set(
        f"arena:summary:{summary.run_id}",
        json.dumps(summary.model_dump(mode="json")),
        ex=7 * 24 * 3600,
    )


async def _publish_terminal(
    redis: RedisCache,
    metadata: ArenaRunMetadata,
    summary: RunSummary,
) -> None:
    await redis.client.publish(
        _PUBSUB_PREFIX + str(metadata.run_id),
        json.dumps(
            {
                "type": "summary",
                "run_id": str(metadata.run_id),
                "status": metadata.status.value,
                "summary": summary.model_dump(mode="json"),
            }
        ),
    )


def _write_bc_pairs(settings: Settings, pairs: list[CounterfactualPair]) -> None:
    if not pairs:
        return
    writer = BCDatasetWriter(settings.arena.artifact_dir)
    writer.append_pairs(pairs, settings.arena.artifact_dir)


def _seed_for(run_id: str, ticker: str) -> int:
    return zlib.crc32(f"{run_id}:{ticker}".encode()) & 0xFFFF_FFFF


def _ensure_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _decode(value: str | bytes) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else value


async def _amain() -> None:
    configure_logging()
    settings = get_settings()
    redis = RedisCache()
    timescale = TimescaleStore()
    await redis.connect()
    await timescale.connect()
    agent = load_agent(settings, pick_device())

    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()

    def request_stop() -> None:
        stop_event.set()

    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, request_stop)

    worker = ArenaWorker(
        redis=redis,
        timescale=timescale,
        agent=agent,
        settings=settings,
    )
    try:
        await worker.run(stop_event)
    finally:
        await redis.disconnect()
        await timescale.disconnect()


def main() -> int:
    try:
        asyncio.run(_amain())
    except Exception:
        logger.exception("Arena worker crashed")
        return 1
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
