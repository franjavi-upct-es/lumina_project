# backend/api/routes/backtest.py
"""Backtest endpoints — submission + result lookup.

Architecture
============
Backtests are long-running CPU-bound jobs; the API must not run them
inline. The intended deployment uses Celery + Redis as the queue
broker (the same Redis instance that powers the Feature Store).

The current implementation is a *thin* in-memory placeholder that
accepts a request, assigns a run id, and stores the request in Redis
under ``backtest:request:<id>``. A worker (not yet implemented in this
module) is expected to pick the request up, run it through
``backend.simulation.environments.LuminaTradingEnv`` against a frozen
agent checkpoint, and write the result back under
``backtest:result:<id>``.

This decoupling is intentional even at the API level: it means the
worker can be deployed independently and scaled horizontally.

Future work
===========
The Celery worker side of the pipeline will be wired in a follow-up
commit; the API surface defined here is already stable.
"""

from __future__ import annotations

import json
import uuid

from fastapi import APIRouter, Depends, HTTPException

from backend.api.deps import get_redis, require_api_key
from backend.api.schemas import BacktestRequest, BacktestResultResponse
from backend.data_engine.storage.redis_cache import RedisCache

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
    return BacktestResultResponse(run_id=run_id, status="pending")


@router.get(
    "/results/{run_id}",
    response_model=BacktestResultResponse,
    dependencies=[Depends(require_api_key)],
)
async def get_results(
    run_id: str,
    redis: RedisCache = Depends(get_redis),
) -> BacktestResultResponse:
    """Return the current status / results of a previously-submitted backtest."""
    raw = await redis.client.get(_result_key(run_id))
    if raw is None:
        raise HTTPException(status_code=404, detail=f"Unknown run_id: {run_id}")
    data = json.loads(raw)
    return BacktestResultResponse(
        run_id=run_id,
        status=data.get("status", "pending"),
        sharpe=data.get("sharpe"),
        max_drawdown=data.get("max_drawdown"),
        total_return=data.get("total_return"),
    )
