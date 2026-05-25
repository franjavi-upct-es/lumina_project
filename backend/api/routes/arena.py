# backend/api/routes/arena.py
"""Spartan Arena REST + WebSocket endpoints.

The router intentionally mirrors the existing backtest route's pattern
(see ``backtest.py``): the API surface enqueues work and stores the run
request in Redis, but does not block on the run itself. A worker
(``scripts/run_arena.py`` for now; a Celery task in production) picks
up the request and produces records.

Endpoints
---------
    POST   /arena/run
    GET    /arena/runs
    GET    /arena/runs/{run_id}
    GET    /arena/runs/{run_id}/decisions
    GET    /arena/runs/{run_id}/divergences
    GET    /arena/runs/{run_id}/explanations
    GET    /arena/runs/{run_id}/pairs
    GET    /arena/runs/{run_id}/summary
    WS     /arena/runs/{run_id}/live
    POST   /arena/runs/{run_id}/cancel
"""

from __future__ import annotations

import contextlib
import json
import os
from datetime import UTC, datetime
from uuid import UUID, uuid4

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Query,
    WebSocket,
    WebSocketDisconnect,
    status,
)
from loguru import logger
from pydantic import BaseModel, Field

from backend.api.deps import get_redis, get_timescale, require_api_key
from backend.config.constants import (
    ARENA_DEFAULT_TRAJECTORIES,
    ARENA_MAX_TRAJECTORIES,
    ARENA_MIN_TRAJECTORIES,
)
from backend.data_engine.storage.redis_cache import RedisCache
from backend.data_engine.storage.timescale import TimescaleStore
from backend.simulation.arena.schemas import (
    ArenaRunMetadata,
    ArenaRunStatus,
    CounterfactualPair,
    DecisionRecord,
    DivergencePoint,
    RunSummary,
    StepExplanation,
)

router = APIRouter(tags=["Arena"])

_REDIS_REQUEST_PREFIX = "arena:request:"
_REDIS_CANCEL_PREFIX = "arena:cancel:"
_REDIS_PUBSUB_PREFIX = "arena:"


# ----------------------------------------------------------------------
# Request body
# ----------------------------------------------------------------------
class ArenaRunRequest(BaseModel):
    """POST /arena/run body — minimal control surface."""

    ticker: str = Field(..., pattern=r"^[A-Z]{1,8}$")
    start_date: datetime
    end_date: datetime
    n_trajectories: int = Field(
        default=ARENA_DEFAULT_TRAJECTORIES,
        ge=ARENA_MIN_TRAJECTORIES,
        le=ARENA_MAX_TRAJECTORIES,
    )
    mc_seeds: list[int] | None = None
    playback_multiplier: float = Field(default=1.0, ge=1.0, le=1000.0)


class ArenaRunCreatedResponse(BaseModel):
    run_id: UUID
    status: ArenaRunStatus = ArenaRunStatus.PENDING


# ----------------------------------------------------------------------
# POST /arena/run
# ----------------------------------------------------------------------
@router.post(
    "/run",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=ArenaRunCreatedResponse,
    dependencies=[Depends(require_api_key)],
)
async def start_run(
    req: ArenaRunRequest,
    redis: RedisCache = Depends(get_redis),
    timescale: TimescaleStore = Depends(get_timescale),
) -> ArenaRunCreatedResponse:
    """Enqueue an arena run; return immediately with the new run_id."""
    if req.end_date <= req.start_date:
        raise HTTPException(status_code=400, detail="end_date must be after start_date")

    seeds = req.mc_seeds
    if seeds is None:
        seeds = [int.from_bytes(os.urandom(4), "big") for _ in range(req.n_trajectories)]
    elif len(seeds) != req.n_trajectories:
        raise HTTPException(
            status_code=400,
            detail=(
                f"mc_seeds length ({len(seeds)}) does not match n_trajectories "
                f"({req.n_trajectories})"
            ),
        )

    metadata = ArenaRunMetadata(
        run_id=uuid4(),
        status=ArenaRunStatus.PENDING,
        ticker=req.ticker,
        start_date=req.start_date,
        end_date=req.end_date,
        n_trajectories=req.n_trajectories,
        mc_seeds=seeds,
        playback_multiplier=req.playback_multiplier,
    )

    # 1. Persist metadata to the DB so GET endpoints can find it.
    await _insert_run_metadata(timescale, metadata)
    # 2. Stash the original request in Redis so a worker can pick it up.
    await redis.client.set(
        _REDIS_REQUEST_PREFIX + str(metadata.run_id),
        json.dumps(req.model_dump(mode="json")),
        ex=24 * 3600,
    )
    logger.info(
        "Arena run {} accepted: ticker={} N={} playback={:.2f}",
        metadata.run_id,
        metadata.ticker,
        metadata.n_trajectories,
        metadata.playback_multiplier,
    )
    return ArenaRunCreatedResponse(run_id=metadata.run_id, status=ArenaRunStatus.PENDING)


# ----------------------------------------------------------------------
# Listings + lookup
# ----------------------------------------------------------------------
@router.get("/runs", response_model=list[ArenaRunMetadata])
async def list_runs(
    limit: int = Query(default=100, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
    timescale: TimescaleStore = Depends(get_timescale),
) -> list[ArenaRunMetadata]:
    async with timescale._conn() as conn:
        rows = await conn.fetch(
            """
            SELECT run_id, status, ticker, start_date, end_date,
                   n_trajectories, mc_seeds, playback_multiplier,
                   created_at, completed_at, failure_reason
            FROM arena_runs
            ORDER BY created_at DESC
            LIMIT $1 OFFSET $2
            """,
            limit,
            offset,
        )
    return [_row_to_run_metadata(r) for r in rows]


@router.get("/runs/{run_id}", response_model=ArenaRunMetadata)
async def get_run(
    run_id: UUID,
    timescale: TimescaleStore = Depends(get_timescale),
) -> ArenaRunMetadata:
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
        raise HTTPException(status_code=404, detail="run_id not found")
    return _row_to_run_metadata(row)


@router.get("/runs/{run_id}/decisions", response_model=list[DecisionRecord])
async def get_decisions(
    run_id: UUID,
    trajectory_id: int | None = Query(default=None, ge=0),
    limit: int = Query(default=100, ge=1, le=10_000),
    offset: int = Query(default=0, ge=0),
    timescale: TimescaleStore = Depends(get_timescale),
) -> list[DecisionRecord]:
    query = """
        SELECT record_id, run_id, trajectory_id, step_index,
               sim_timestamp, wall_timestamp, ticker, ohlcv,
               action_kind, action_vector, confidence, uncertainty,
               realized_reward, state_artifact_path, attribution, mc_seed
        FROM arena_decision_records
        WHERE run_id = $1
    """
    params: list = [run_id]
    if trajectory_id is not None:
        query += " AND trajectory_id = $2"
        params.append(trajectory_id)
    limit_pos = len(params) + 1
    offset_pos = len(params) + 2
    query += f" ORDER BY step_index ASC LIMIT ${limit_pos} OFFSET ${offset_pos}"
    params.extend([limit, offset])
    async with timescale._conn() as conn:
        rows = await conn.fetch(query, *params)
    return [_row_to_decision_record(r) for r in rows]


@router.get("/runs/{run_id}/divergences", response_model=list[DivergencePoint])
async def get_divergences(
    run_id: UUID,
    limit: int = Query(default=100, ge=1, le=10_000),
    offset: int = Query(default=0, ge=0),
    timescale: TimescaleStore = Depends(get_timescale),
) -> list[DivergencePoint]:
    async with timescale._conn() as conn:
        rows = await conn.fetch(
            """
            SELECT run_id, step_index, sim_timestamp,
                   best_trajectory_id, worst_trajectory_id,
                   best_action_vector, worst_action_vector,
                   action_l2_distance, best_subsequent_sharpe,
                   worst_subsequent_sharpe, sharpe_delta
            FROM arena_divergence_points
            WHERE run_id = $1
            ORDER BY step_index ASC LIMIT $2 OFFSET $3
            """,
            run_id,
            limit,
            offset,
        )
    return [
        DivergencePoint(
            run_id=r["run_id"],
            step_index=r["step_index"],
            sim_timestamp=r["sim_timestamp"],
            best_trajectory_id=r["best_trajectory_id"],
            worst_trajectory_id=r["worst_trajectory_id"],
            best_action_vector=list(r["best_action_vector"]),
            worst_action_vector=list(r["worst_action_vector"]),
            action_l2_distance=float(r["action_l2_distance"]),
            best_subsequent_sharpe=float(r["best_subsequent_sharpe"]),
            worst_subsequent_sharpe=float(r["worst_subsequent_sharpe"]),
            sharpe_delta=float(r["sharpe_delta"]),
        )
        for r in rows
    ]


@router.get("/runs/{run_id}/explanations", response_model=list[StepExplanation])
async def get_explanations(
    run_id: UUID,
    limit: int = Query(default=100, ge=1, le=10_000),
    offset: int = Query(default=0, ge=0),
    timescale: TimescaleStore = Depends(get_timescale),
) -> list[StepExplanation]:
    async with timescale._conn() as conn:
        rows = await conn.fetch(
            """
            SELECT record_id, text, tags
            FROM arena_step_explanations
            WHERE run_id = $1
            ORDER BY record_id LIMIT $2 OFFSET $3
            """,
            run_id,
            limit,
            offset,
        )
    return [
        StepExplanation(record_id=r["record_id"], text=r["text"], tags=list(r["tags"] or []))
        for r in rows
    ]


@router.get("/runs/{run_id}/pairs", response_model=list[CounterfactualPair])
async def get_pairs(
    run_id: UUID,
    limit: int = Query(default=100, ge=1, le=10_000),
    offset: int = Query(default=0, ge=0),
    timescale: TimescaleStore = Depends(get_timescale),
) -> list[CounterfactualPair]:
    async with timescale._conn() as conn:
        rows = await conn.fetch(
            """
            SELECT pair_id, run_id, divergence_step_index, sim_timestamp,
                   state_artifact_path, good_action_vector, bad_action_vector,
                   good_outcome_sharpe, bad_outcome_sharpe, confidence_score
            FROM arena_counterfactual_pairs
            WHERE run_id = $1
            ORDER BY confidence_score DESC LIMIT $2 OFFSET $3
            """,
            run_id,
            limit,
            offset,
        )
    return [
        CounterfactualPair(
            pair_id=r["pair_id"],
            run_id=r["run_id"],
            divergence_step_index=r["divergence_step_index"],
            sim_timestamp=r["sim_timestamp"],
            state_artifact_path=r["state_artifact_path"],
            good_action_vector=list(r["good_action_vector"]),
            bad_action_vector=list(r["bad_action_vector"]),
            good_outcome_sharpe=float(r["good_outcome_sharpe"]),
            bad_outcome_sharpe=float(r["bad_outcome_sharpe"]),
            confidence_score=float(r["confidence_score"]),
        )
        for r in rows
    ]


@router.get("/runs/{run_id}/summary", response_model=RunSummary)
async def get_summary(
    run_id: UUID,
    redis: RedisCache = Depends(get_redis),
) -> RunSummary:
    """Look up a cached :class:`RunSummary` written by the worker."""
    raw = await redis.client.get(f"arena:summary:{run_id}")
    if raw is None:
        raise HTTPException(status_code=404, detail="summary not yet available")
    return RunSummary(**json.loads(raw))


@router.post(
    "/runs/{run_id}/cancel",
    dependencies=[Depends(require_api_key)],
)
async def cancel_run(
    run_id: UUID,
    redis: RedisCache = Depends(get_redis),
) -> dict:
    """Set a cooperative-cancel flag in Redis. The worker polls it."""
    await redis.client.set(_REDIS_CANCEL_PREFIX + str(run_id), "1", ex=3600)
    return {"run_id": str(run_id), "cancel_requested": True}


# ----------------------------------------------------------------------
# WebSocket — live stream
# ----------------------------------------------------------------------
@router.websocket("/runs/{run_id}/live")
async def live_stream(websocket: WebSocket, run_id: UUID) -> None:
    """Stream DecisionRecord + DivergencePoint events for a running arena.

    The worker publishes JSON-serialised events to the Redis channel
    ``arena:<run_id>``. We subscribe, relay every message verbatim, and
    drop the connection on any error.
    """
    await websocket.accept()
    redis = await get_redis()
    channel_name = _REDIS_PUBSUB_PREFIX + str(run_id)
    pubsub = redis.client.pubsub()
    try:
        await pubsub.subscribe(channel_name)
        async for message in pubsub.listen():
            if message is None:
                continue
            if message.get("type") != "message":
                continue
            data = message.get("data")
            if isinstance(data, bytes):
                data = data.decode("utf-8", errors="replace")
            await websocket.send_text(data)
    except (WebSocketDisconnect, RuntimeError):
        pass
    except Exception:
        logger.exception("Arena WS for run {} crashed", run_id)
    finally:
        with contextlib.suppress(Exception):
            await pubsub.unsubscribe(channel_name)
        with contextlib.suppress(Exception):
            await pubsub.aclose()


# ----------------------------------------------------------------------
# Internal helpers
# ----------------------------------------------------------------------
async def _insert_run_metadata(timescale: TimescaleStore, metadata: ArenaRunMetadata) -> None:
    async with timescale._conn() as conn:
        await conn.execute(
            """
            INSERT INTO arena_runs (
                run_id, status, ticker, start_date, end_date,
                n_trajectories, mc_seeds, playback_multiplier,
                created_at, completed_at, failure_reason
            ) VALUES (
                $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11
            )
            """,
            metadata.run_id,
            metadata.status.value,
            metadata.ticker,
            metadata.start_date,
            metadata.end_date,
            metadata.n_trajectories,
            list(metadata.mc_seeds),
            float(metadata.playback_multiplier),
            metadata.created_at,
            metadata.completed_at,
            metadata.failure_reason,
        )


def _row_to_run_metadata(row) -> ArenaRunMetadata:
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


def _row_to_decision_record(row) -> DecisionRecord:
    raw_attribution = row["attribution"]
    if isinstance(raw_attribution, str):
        raw_attribution = json.loads(raw_attribution)
    return DecisionRecord(
        record_id=row["record_id"],
        run_id=row["run_id"],
        trajectory_id=row["trajectory_id"],
        step_index=row["step_index"],
        sim_timestamp=row["sim_timestamp"],
        wall_timestamp=row["wall_timestamp"],
        ticker=row["ticker"],
        ohlcv=row["ohlcv"] if isinstance(row["ohlcv"], dict) else json.loads(row["ohlcv"]),
        action_kind=row["action_kind"],
        action_vector=list(row["action_vector"]),
        confidence=float(row["confidence"]),
        uncertainty=float(row["uncertainty"]),
        realized_reward=None if row["realized_reward"] is None else float(row["realized_reward"]),
        state_artifact_path=row["state_artifact_path"],
        attribution=raw_attribution,
        mc_seed=int(row["mc_seed"]),
    )


# Convenience constant for callers that publish events.
def channel_for(run_id: UUID) -> str:
    """Return the canonical Redis pub/sub channel for a given run."""
    return _REDIS_PUBSUB_PREFIX + str(run_id)


# Convenience helper for the worker.
def cancel_flag_key(run_id: UUID) -> str:
    return _REDIS_CANCEL_PREFIX + str(run_id)


# Avoid unused-import warnings for the `datetime` UTC alias when not needed
# in the typed code paths above.
_ = UTC  # used by some payload constructors below.
