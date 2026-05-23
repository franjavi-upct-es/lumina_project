# backend/api/main.py
"""FastAPI application factory.

We assemble the app via a factory function rather than a top-level
``app = FastAPI(...)`` so that test code can construct fresh instances
with overridden dependencies. The module-level ``app`` at the bottom is
the production entry point that uvicorn (and the Dockerfile CMD) target.

Lifespan handling
=================
The async ``lifespan`` context manager wires up startup and shutdown:

* On startup we configure logging, pre-connect to Redis and TimescaleDB
  (catching misconfiguration before the first request rather than on
  it), and start the :class:`LocalKillSwitchListener` so the in-process
  kill-switch latch (audit finding 3.2) stays synchronized with the
  Redis canonical state.
* On shutdown we close both connections cleanly and stop the listener.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from backend.api.middleware import ObservabilityMiddleware
from backend.api.routes import agent, arena, backtest, data, monitoring, portfolio, risk
from backend.config.logging import configure_logging
from backend.config.settings import get_settings
from backend.data_engine.storage.redis_cache import get_redis_cache
from backend.data_engine.storage.timescale import get_timescale_store
from backend.execution.safety.kill_switch import LocalKillSwitchListener

# Module-level handle so the lifespan can stop the listener cleanly on
# shutdown. We avoid stashing it on ``app.state`` because that would
# couple the lifespan to FastAPI specifics; a module-level attribute
# keeps the listener observable from any unit test that imports the
# module.
_KILL_SWITCH_LISTENER: LocalKillSwitchListener | None = None


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """Startup → yield → shutdown."""
    global _KILL_SWITCH_LISTENER

    configure_logging()
    redis = get_redis_cache()
    ts = get_timescale_store()
    await redis.connect()
    await ts.connect()

    # Start the in-process kill-switch listener. This must happen AFTER
    # Redis is connected.
    _KILL_SWITCH_LISTENER = LocalKillSwitchListener(redis)
    await _KILL_SWITCH_LISTENER.start()

    logger.info("Lumina V3 API started")
    try:
        yield
    finally:
        if _KILL_SWITCH_LISTENER is not None:
            await _KILL_SWITCH_LISTENER.stop()
            _KILL_SWITCH_LISTENER = None
        await redis.disconnect()
        await ts.disconnect()
        logger.info("Lumina V3 API stopped")


def create_app() -> FastAPI:
    """Build a new FastAPI instance with all routers + middleware wired."""
    settings = get_settings()
    app = FastAPI(
        title="Lumina V3",
        description="Chimera deep-fusion algorithmic trading system",
        version="3.0.0",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(ObservabilityMiddleware)
    app.include_router(agent.router)
    app.include_router(arena.router, prefix="/arena", tags=["Arena"])
    app.include_router(backtest.router)
    app.include_router(data.router)
    app.include_router(monitoring.router)
    app.include_router(portfolio.router)
    app.include_router(risk.router)
    return app


app = create_app()
