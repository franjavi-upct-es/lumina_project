# backend/api/routes/monitoring.py
"""Health-check and Prometheus-metrics endpoints.

The ``/health`` endpoint is the *single* status panel the operator and
the frontend trust. It probes every subsystem we own — Redis, Timescale,
the configured broker, and the kill-switch state — and aggregates the
results into one of three top-level statuses (``ok`` / ``degraded`` /
``down``).

The ``/metrics`` endpoint is the Prometheus scraper target. All counters
and histograms from across the codebase are registered with the default
Prometheus registry on import, so ``generate_latest()`` produces the
complete time-series snapshot without any explicit list of metrics here.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Response
from loguru import logger
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from backend.api.deps import get_broker, get_kill_switch, get_redis, get_timescale
from backend.api.schemas import HealthResponse, HealthStatus
from backend.data_engine.storage.redis_cache import RedisCache
from backend.data_engine.storage.timescale import TimescaleStore
from backend.execution.broker.base import BaseBroker
from backend.execution.safety.kill_switch import KillSwitch, KillSwitchState

router = APIRouter(prefix="/api/monitoring", tags=["monitoring"])


def _aggregate_status(
    components_connected: bool,
    kill_state: KillSwitchState,
) -> HealthStatus:
    """Derive the top-level status from per-subsystem health and kill switch.

    Decision table:

        kill = LIQUIDATE_ALL  → "down"     (system has explicitly given up)
        kill = CLOSE_ONLY     → "degraded" (still running, but risk-constrained)
        any component down    → "degraded"
        otherwise             → "ok"
    """
    if kill_state == KillSwitchState.LIQUIDATE_ALL:
        return "down"
    if kill_state == KillSwitchState.CLOSE_ONLY or not components_connected:
        return "degraded"
    return "ok"


@router.get("/health", response_model=HealthResponse)
async def health(
    redis: RedisCache = Depends(get_redis),
    ts: TimescaleStore = Depends(get_timescale),
    broker: BaseBroker = Depends(get_broker),
    ks: KillSwitch = Depends(get_kill_switch),
) -> HealthResponse:
    """Aggregate health check across Redis, Timescale, broker, kill switch.

    Each subsystem's ``health_check`` is independent and may itself fail
    (e.g. Redis being unreachable). We catch those exceptions per
    subsystem so a single failure does not poison the others.
    """

    async def _probe(name: str, coro):
        try:
            return await coro
        except Exception as exc:
            logger.warning(f"Health probe '{name}' failed: {exc}")
            return {"connected": False, "error": str(exc)}

    redis_h = await _probe("redis", redis.health_check())
    ts_h = await _probe("timescale", ts.health_check())
    broker_h = await _probe("broker", broker.health_check())
    kill_state = await ks.get_state()

    components = {
        "redis": redis_h,
        "timescale": ts_h,
        "broker": broker_h,
        "kill_switch": {"state": kill_state.value},
    }

    all_connected = all(
        c.get("connected", True) for k, c in components.items() if k != "kill_switch"
    )
    return HealthResponse(
        status=_aggregate_status(all_connected, kill_state),
        components=components,
    )


@router.get("/metrics")
def metrics() -> Response:
    """Prometheus scrape endpoint.

    The default Prometheus registry is populated by every metric defined
    across ``backend/*`` simply via module import (counters and
    histograms register themselves on construction). ``generate_latest``
    serialises the entire registry in the text exposition format.
    """
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
