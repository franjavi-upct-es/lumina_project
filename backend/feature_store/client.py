# backend/feature_store/client.py
"""Unified Feature Store client — the single entry point for cognition.

The client wraps either an :class:`OnlineFeatureStore` (Redis-backed,
sub-millisecond, used by the live agent) *or* an
:class:`OfflineFeatureStore` (Timescale-backed, used by training and
backtesting), depending on the ``mode`` chosen at construction. Mixing
the two would be a bug — the live decision loop has no place inside a
training batch and vice versa — so we enforce the choice with explicit
``_require`` guards on every method.

A common alternative is to expose two unrelated classes; we keep them
behind a single facade so cognition code can be written once and
parametrised by mode (e.g. ``FeatureStoreClient(mode=settings.MODE)``).
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

import numpy as np

from backend.data_engine.storage.redis_cache import RedisCache
from backend.data_engine.storage.timescale import TimescaleStore
from backend.feature_store.definitions import FEATURE_REGISTRY
from backend.feature_store.offline import OfflineFeatureStore
from backend.feature_store.online import OnlineFeatureStore

Mode = Literal["online", "offline"]


class FeatureStoreClient:
    """Facade over the online / offline feature stores.

    Parameters
    ----------
    mode
        Either ``"online"`` (live inference) or ``"offline"`` (training).
    redis
        Required if ``mode == "online"``; ignored otherwise.
    timescale
        Required if ``mode == "offline"``; ignored otherwise.

    Raises
    ------
    ValueError
        If the mode is unknown, or if the corresponding storage backend
        was not provided.
    """

    def __init__(
        self,
        mode: Mode,
        redis: RedisCache | None = None,
        timescale: TimescaleStore | None = None,
    ) -> None:
        self._mode: Mode = mode
        self._online: OnlineFeatureStore | None = None
        self._offline: OfflineFeatureStore | None = None
        if mode == "online":
            if redis is None:
                raise ValueError("online mode requires a RedisCache instance")
            self._online = OnlineFeatureStore(redis)
        elif mode == "offline":
            if timescale is None:
                raise ValueError("offline mode requires a TimescaleStore instance")
            self._offline = OfflineFeatureStore(timescale)
        else:
            raise ValueError(f"Unknown mode: {mode!r}; expected 'online' or 'offline'")

    # ----- properties -------------------------------------------------------
    @property
    def mode(self) -> Mode:
        return self._mode

    # ----- online API -------------------------------------------------------
    async def get_embedding(self, feature_name: str, ticker: str) -> np.ndarray | None:
        """Return a single embedding vector from Redis. ``None`` if absent."""
        self._require("online")
        assert self._online is not None
        return await self._online.get(feature_name, ticker)

    async def mget_embeddings(
        self,
        feature_name: str,
        tickers: list[str],
    ) -> dict[str, np.ndarray]:
        """Batched MGET of one embedding type across many tickers."""
        self._require("online")
        assert self._online is not None
        return await self._online.mget(feature_name, tickers)

    async def get_bundle(
        self,
        ticker: str,
        feature_names: list[str] | None = None,
    ) -> dict[str, np.ndarray]:
        """One-shot retrieval of all hot features for a single ticker.

        Used by the live agent loop; this is the hottest code path in
        the system (called once per inference cycle) and is implemented
        in :meth:`OnlineFeatureStore.get_bundle` as a single Redis MGET.
        """
        self._require("online")
        assert self._online is not None
        return await self._online.get_bundle(ticker, feature_names)

    # ----- offline API ------------------------------------------------------
    async def get_training_window(
        self,
        ticker: str,
        start: datetime,
        end: datetime,
        features: list[str] | None = None,
    ) -> dict[str, Any]:
        """Fetch one or more *cold* features for a single training sample.

        Returns a heterogeneous dict keyed by feature name. Concrete
        value types per feature are documented on
        :meth:`OfflineFeatureStore.get_training_window`.
        """
        self._require("offline")
        assert self._offline is not None
        return await self._offline.get_training_window(ticker, start, end, features)

    # ----- discovery --------------------------------------------------------
    @staticmethod
    def list_features() -> list[str]:
        """Names of every feature registered with the store, sorted."""
        return sorted(FEATURE_REGISTRY.keys())

    # ----- internal ---------------------------------------------------------
    def _require(self, mode: Mode) -> None:
        """Raise if the client is used outside its declared mode."""
        if self._mode != mode:
            raise RuntimeError(f"Method requires mode='{mode}' but client is '{self._mode}'.")
