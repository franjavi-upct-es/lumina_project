# backend/api/deps.py
"""FastAPI dependency providers and authentication helpers.

This module is the single source of truth for *how* a request handler
obtains its collaborators. FastAPI's ``Depends`` machinery wires these
into every route automatically.

There are three categories of providers:

* **Storage** (``get_redis``, ``get_timescale``). The underlying objects
  are process-wide singletons (see the ``get_*`` functions in the
  storage modules); our providers simply ensure the connection has been
  established before the first request handler tries to use it.

* **Cross-cutting services** (``get_kill_switch``, ``get_broker``).
  These are *constructed per request* but the resources they wrap
  (Redis client, broker SDK) are themselves singletons, so the cost is
  effectively zero.

* **Authentication** (``require_api_key``). A no-op when the configured
  API key is empty, which keeps local development frictionless while
  staying strict in production.
"""

from __future__ import annotations

from fastapi import Depends, Header, HTTPException, status
from loguru import logger

from backend.config.settings import Settings, get_settings
from backend.data_engine.storage.redis_cache import RedisCache, get_redis_cache
from backend.data_engine.storage.timescale import TimescaleStore, get_timescale_store
from backend.execution.broker.base import BaseBroker
from backend.execution.broker.paper_adapter import PaperBroker
from backend.execution.safety.kill_switch import KillSwitch

# Process-wide singleton — instantiated lazily on first call to
# :func:`get_broker`. We keep it here (rather than inside ``broker/``)
# because the *choice* of broker is an API-level concern: the broker
# implementation is selected from settings and any test fixture can
# override the dependency via FastAPI's ``app.dependency_overrides``.
_BROKER_SINGLETON: BaseBroker | None = None


async def get_redis() -> RedisCache:
    """Return the connected :class:`RedisCache` singleton.

    Calling ``connect()`` on an already-connected instance is a no-op,
    so the cost of this guard is one ``ping``-equivalent.
    """
    r = get_redis_cache()
    await r.connect()
    return r


async def get_timescale() -> TimescaleStore:
    """Return the connected :class:`TimescaleStore` singleton."""
    t = get_timescale_store()
    await t.connect()
    return t


def get_kill_switch(redis: RedisCache = Depends(get_redis)) -> KillSwitch:
    """Build a stateless :class:`KillSwitch` view over the shared Redis."""
    return KillSwitch(redis)


def get_broker(settings: Settings = Depends(get_settings)) -> BaseBroker:
    """Return the configured broker.

    Selection logic
    ---------------
    * ``BROKER_MODE=paper`` → :class:`PaperBroker` (default; no network).
    * ``BROKER_MODE=alpaca`` → :class:`AlpacaBroker` (imported lazily so
      tests that never hit Alpaca don't pay the heavy import cost).

    The instance is cached at module scope so subsequent calls within the
    same process return the *same* object — this is required for the
    :class:`PaperBroker` whose state (cash, positions) lives in memory.
    """
    global _BROKER_SINGLETON
    if _BROKER_SINGLETON is not None:
        return _BROKER_SINGLETON
    if settings.BROKER_MODE == "alpaca":
        # Lazy import: keeps the alpaca-py dependency optional in
        # environments that only exercise the paper broker.
        from backend.execution.broker.alpaca_adapter import AlpacaBroker

        _BROKER_SINGLETON = AlpacaBroker()
        logger.info("Broker singleton initialised: AlpacaBroker")
    else:
        _BROKER_SINGLETON = PaperBroker()
        logger.info("Broker singleton initialised: PaperBroker")
    return _BROKER_SINGLETON


def reset_broker_singleton() -> None:
    """Test helper — clear the cached broker so the next call rebuilds it.

    Only intended for use from pytest fixtures; never call this from
    production code.
    """
    global _BROKER_SINGLETON
    _BROKER_SINGLETON = None


def require_api_key(
    x_api_key: str | None = Header(default=None),
    settings: Settings = Depends(get_settings),
) -> None:
    """Reject the request unless the ``x-api-key`` header matches settings.

    When ``settings.API_KEY`` is empty (development default) the check is
    skipped entirely. This keeps the local developer experience friction-
    free while making it impossible to deploy without authentication —
    the production deployment is expected to set ``API_KEY`` via the
    environment.
    """
    if not settings.API_KEY:
        return
    if x_api_key != settings.API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
        )
