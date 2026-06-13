# backend/api/routes/backtest.py
"""Backtest endpoints — submission + result lookup.

Architecture
============
Backtests are long-running CPU-bound jobs; the API must not run them
inline. The composed runtime starts ``backend.simulation.backtest_worker``,
which polls Redis for ``backtest:request:<id>``, runs the frozen agent
through ``backend.simulation.environments.LuminaTradingEnv``, logs the run
to MLflow, and writes the result back under ``backtest:result:<id>``.

This decoupling is intentional even at the API level: it means the
worker can be deployed independently and scaled horizontally.

The Redis-key queue is deliberately simple for the local composed stack.
A production Celery/RQ worker can consume the same request/result contract
later without changing the dashboard API surface.
"""

from __future__ import annotations

import json
import uuid

from fastapi import APIRouter, Depends, HTTPException

from backend.api.deps import get_redis, get_timescale, require_api_key
from backend.api.schemas import BacktestRequest, BacktestResultResponse
from backend.data_engine.storage.redis_cache import RedisCache
from backend.data_engine.storage.timescale import TimescaleStore

router = APIRouter(prefix="/api/backtest", tags=["backtest"])


def _request_key(run_id: str) -> str:
    return f"backtest:request:{run_id}"


def _result_key(run_id: str) -> str:
    return f"backtest:result:{run_id}"


@router.post(
    "/run",
    response_model=BacktestResultResponse,
    dependencies=[Depends(require_api_key)],
)
async def run_backtest(
    req: BacktestRequest,
    redis: RedisCache = Depends(get_redis),
    ts: TimescaleStore = Depends(get_timescale),
) -> BacktestResultResponse:
    """Enqueue a backtest job.

    The endpoint always returns immediately with a freshly-minted
    ``run_id``. Callers must poll :func:`get_results` for completion.
    """
    run_id = f"bt_{uuid.uuid4().hex[:12]}"
    payload = req.model_dump(mode="json")
    await redis.client.set(_request_key(run_id), json.dumps(payload), ex=24 * 3600)
    await redis.client.set(
        _result_key(run_id),
        json.dumps({"status": "pending"}),
        ex=24 * 3600,
    )
    # Save to TimescaleDB for history
    await ts.upsert_backtest_run(run_id, "pending")
    return BacktestResultResponse(run_id=run_id, status="pending")


@router.get(
    "/runs",
    response_model=list[BacktestResultResponse],
    dependencies=[Depends(require_api_key)],
)
async def list_backtest_runs(
    ts: TimescaleStore = Depends(get_timescale),
) -> list[BacktestResultResponse]:
    """List historical backtest runs."""
    rows = await ts.get_backtest_runs()
    return [
        BacktestResultResponse(
            run_id=r["run_id"],
            status=r["status"],
            sharpe=r["sharpe"],
            max_drawdown=r["max_drawdown"],
            total_return=r["total_return"],
        )
        for r in rows
    ]


@router.get(
    "/results/{run_id}",
    response_model=BacktestResultResponse,
    dependencies=[Depends(require_api_key)],
)
async def get_results(
    run_id: str,
    redis: RedisCache = Depends(get_redis),
    ts: TimescaleStore = Depends(get_timescale),
) -> BacktestResultResponse:
    """Return the current status / results of a previously-submitted backtest."""
    raw = await redis.client.get(_result_key(run_id))
    if raw is None:
        raise HTTPException(status_code=404, detail=f"Unknown run_id: {run_id}")
    data = json.loads(raw)

    status = data.get("status", "pending")
    sharpe = data.get("sharpe")
    max_drawdown = data.get("max_drawdown")
    total_return = data.get("total_return")
    failure_reason = data.get("failure_reason")

    # Keep Timescale synced
    await ts.upsert_backtest_run(run_id, status, sharpe, max_drawdown, total_return)

    return BacktestResultResponse(
        run_id=run_id,
        status=status,
        sharpe=sharpe,
        max_drawdown=max_drawdown,
        total_return=total_return,
        failure_reason=failure_reason,
    )
