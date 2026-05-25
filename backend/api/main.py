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

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from backend.api.deps import get_broker
from backend.api.middleware import ObservabilityMiddleware
from backend.api.routes import agent, arena, backtest, data, monitoring, portfolio, risk
from backend.config.constants import TARGET_TICKERS
from backend.config.logging import configure_logging
from backend.config.settings import get_settings
from backend.data_engine.collectors.yfinance_collector import YFinanceCollector
from backend.data_engine.storage.redis_cache import get_redis_cache
from backend.data_engine.storage.timescale import get_timescale_store
from backend.execution.safety.kill_switch import LocalKillSwitchListener

_KILL_SWITCH_LISTENER: LocalKillSwitchListener | None = None
_BG_TASKS: list[asyncio.Task] = []


async def _auto_backfill_task(ts):
    """Auto-download yfinance data to ensure the database is ready for users."""
    try:
        # Give the API a few seconds to start up
        await asyncio.sleep(5)
        # We just check SPY. If it's missing recent data, we backfill.
        end = datetime.now(UTC)
        start = end - timedelta(days=30)
        df = await ts.get_historical_window("SPY", start, end, freq="1d")
        if df.height < 5:
            logger.info("Database appears empty or sparse. Starting automatic yfinance backfill...")
            await YFinanceCollector.backfill_to_timescale(ts, list(TARGET_TICKERS), start.date(), end.date())
            logger.success("Automatic backfill complete.")
    except asyncio.CancelledError:
        pass
    except Exception as e:
        logger.error(f"Auto-backfill task failed: {e}")

async def _portfolio_logger_task(ts):
    """Continuously record portfolio equity to TimescaleDB for the Dashboard history."""
    try:
        broker = get_broker() # Get the singleton
        while True:
            account = await broker.get_account()
            await ts.insert_portfolio_record(datetime.now(UTC), account.equity, account.cash)
            # Log every minute
            await asyncio.sleep(60)
    except asyncio.CancelledError:
        pass
    except Exception as e:
        logger.error(f"Portfolio logger task failed: {e}")

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
    
    # Start background tasks
    _BG_TASKS.append(asyncio.create_task(_auto_backfill_task(ts)))
    _BG_TASKS.append(asyncio.create_task(_portfolio_logger_task(ts)))

    logger.info("Lumina V3 API started")
    try:
        yield
    finally:
        for t in _BG_TASKS:
            t.cancel()
        if _BG_TASKS:
            await asyncio.gather(*_BG_TASKS, return_exceptions=True)
            
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
    from backend.api.middleware import CongestionControlMiddleware
    app.add_middleware(CongestionControlMiddleware, max_concurrent_requests=200)
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
