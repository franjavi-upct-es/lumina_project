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

import hmac
from typing import Literal

from fastapi import Depends, Header, HTTPException, WebSocket, status
from pydantic import BaseModel

from backend.config.settings import Settings, get_settings
from backend.data_engine.storage.redis_cache import RedisCache, get_redis_cache
from backend.data_engine.storage.timescale import TimescaleStore, get_timescale_store
from backend.execution.broker import factory as broker_factory
from backend.execution.broker.base import BaseBroker
from backend.execution.safety.kill_switch import KillSwitch


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


def get_broker(settings: Settings | None = None) -> BaseBroker:
    """Return the configured process-local broker."""
    return broker_factory.get_broker(settings)


def reset_broker_singleton() -> None:
    """Test helper — clear the cached broker so the next call rebuilds it."""
    broker_factory.reset_broker_singleton()


class UserContext(BaseModel):
    """Context object carrying multi-tenant identity and subscription tier."""

    user_id: str
    tier: Literal["free", "pro", "enterprise"]


def require_api_key(
    x_api_key: str | None = Header(default=None),
    settings: Settings = Depends(get_settings),
) -> UserContext:
    """Reject the request unless the ``x-api-key`` header matches settings.

    Fail-closed (audit F1): when no ``API_KEY`` is configured the request is
    allowed only in development; every other environment is rejected. The key
    comparison is constant-time to avoid a timing side-channel (audit F9).

    Returns a UserContext object to lay the foundation for multi-tenant SaaS
    routing and subscription-tier quotas.
    """
    if not settings.API_KEY:
        if settings.ENVIRONMENT == "development":
            return UserContext(user_id="dev_user", tier="enterprise")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Server API key is not configured.",
        )
    if x_api_key is None or not hmac.compare_digest(x_api_key, settings.API_KEY):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
        )
    return UserContext(user_id="admin_user", tier="enterprise")


def _origin_allowed(origin: str | None, settings: Settings) -> bool:
    """Allow same-origin browser clients and non-browser clients.

    Browsers always send ``Origin`` on the WS handshake; we require it to be in
    the configured CORS allow-list (CSWSH protection). Clients with no Origin
    (e.g. server-to-server) are permitted — they cannot be driven by a
    victim's browser.
    """
    if origin is None:
        return True
    return origin in settings.CORS_ORIGINS


async def authorize_websocket(
    websocket: WebSocket,
    settings: Settings | None = None,
) -> bool:
    """Authorize a WebSocket *before* ``accept()`` (audit F2).

    Validates the ``Origin`` header against the CORS allow-list and, when an
    ``API_KEY`` is configured, a ``?token=`` (or ``?api_key=``) query parameter
    — browsers cannot set custom headers on the WS handshake, so the token
    travels as a query param. On failure the socket is closed with
    policy-violation code 1008 and ``False`` is returned.
    """
    settings = settings or get_settings()
    if not _origin_allowed(websocket.headers.get("origin"), settings):
        await websocket.close(code=1008)
        return False
    if settings.API_KEY:
        token = websocket.query_params.get("token") or websocket.query_params.get("api_key")
        if token is None or not hmac.compare_digest(token, settings.API_KEY):
            await websocket.close(code=1008)
            return False
    elif settings.ENVIRONMENT != "development":
        await websocket.close(code=1008)
        return False
    return True
