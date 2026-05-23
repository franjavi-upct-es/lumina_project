# backend/feature_store/offline.py
"""Offline Feature Store: TimescaleDB-backed, training-grade feature retrieval.

The offline store is the *cold* side of the feature store. It serves
historical, fully-materialised features used during training and back-
testing. It is conceptually the opposite of the online store (which
serves the latest values from Redis at sub-millisecond latency).

Two cold features are currently supported:

* ``ohlcv_window`` — sliding OHLCV window (T × 5) returned as a Polars
  DataFrame, indexed by ``time``. Used to feed the TFT.
* ``news_window`` — concatenation of the last-24-hour headlines for a
  ticker, tokenised through the FinBERT tokenizer and padded/truncated
  to a fixed length of 512 tokens. Returned as a dict of two int64
  tensors (``input_ids``, ``attention_mask``). Used to feed the
  distilled semantic encoder during offline training.

Why two return types in the same orchestrator?
==============================================
We deliberately keep the two shapes distinct rather than forcing both
into a single tensor layout. Forcing fictitious shape uniformity is the
kind of choice that looks clean on paper and bites you six months later
when somebody adds a third modality that doesn't fit the chosen mould.
The orchestrator returns a heterogeneous dict keyed by feature name and
the caller knows what shape to expect for each feature.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import polars as pl
import torch
from loguru import logger
from torch.utils.data import Dataset

from backend.config.constants import (
    NEWS_WINDOW_HOURS,
    OHLCV_WINDOW_MINUTES,
)
from backend.data_engine.storage.timescale import NewsEvent, TimescaleStore
from backend.feature_store.definitions import FEATURE_REGISTRY, NEWS_WINDOW

# Fixed length of the tokenised news window. Matches NEWS_WINDOW.dim in
# definitions.py — kept as a module constant for readability of the
# tokenisation code below.
_NEWS_TOKEN_LENGTH: int = NEWS_WINDOW.dim

# Maximum number of news events to look at per ticker when assembling
# the news window. We hard-cap at 64 to bound the tokenisation work
# even for highly-covered tickers like AAPL.
_NEWS_EVENT_BUDGET: int = 64


class OfflineFeatureStore:
    """Cold-feature retriever sitting on top of TimescaleDB.

    Each public method corresponds to one declared feature in
    :mod:`backend.feature_store.definitions`. The aggregator method
    :meth:`get_training_window` dispatches to the per-feature methods
    and returns a heterogeneous dict keyed by feature name.

    The class is stateless beyond the held :class:`TimescaleStore`
    reference; concurrent calls from multiple training workers are safe
    as long as the underlying connection pool is sized appropriately
    (see ``settings.TIMESCALE_POOL_MAX``).
    """

    def __init__(self, timescale: TimescaleStore) -> None:
        self._ts = timescale

    # ------------------------------------------------------------------ ohlcv
    async def get_ohlcv_window(
        self,
        ticker: str,
        start: datetime,
        end: datetime,
    ) -> pl.DataFrame:
        """Return the raw 1-minute OHLCV bars in ``[start, end)`` for ``ticker``.

        The result is a Polars DataFrame with columns
        ``time, open, high, low, close, volume, vwap, trade_count``.
        Rows are ordered by ``time`` ascending. Missing bars (gaps, holiday
        sessions) are simply absent from the DataFrame — downstream code
        decides whether to forward-fill, mask, or skip them.
        """
        return await self._ts.get_historical_window(ticker, start, end, freq="1m")

    # ------------------------------------------------------------------ news
    async def get_news_window(
        self,
        ticker: str,
        end: datetime,
        hours_back: int = NEWS_WINDOW_HOURS,
        max_length: int = _NEWS_TOKEN_LENGTH,
    ) -> dict[str, torch.Tensor]:
        """Build the tokenised news window for ``ticker`` ending at ``end``.

        We pull every ``NewsEvent`` whose timestamp falls in
        ``[end - hours_back, end]``, sort chronologically, concatenate
        their headlines and bodies into a single string, and tokenise
        with the FinBERT tokenizer. The result is padded with zeros
        (the FinBERT pad-token id is 0) or truncated to ``max_length``.

        Parameters
        ----------
        ticker
            Symbol whose news to retrieve.
        end
            The right endpoint of the time window (typically the bar
            timestamp the model is being asked to predict for).
        hours_back
            Width of the look-back window in hours. Defaults to
            :data:`NEWS_WINDOW_HOURS` from constants (24).
        max_length
            Output sequence length. Defaults to :data:`_NEWS_TOKEN_LENGTH`
            (512), matching ``NEWS_WINDOW.dim`` in the registry.

        Returns
        -------
        dict
            ``{"input_ids": (1, max_length) int64,
               "attention_mask": (1, max_length) int64}``.

            When no news exists for the window the function still
            returns a fully-padded all-zero tensor pair so the consumer
            does not have to special-case missing data — the attention
            mask will be all zeros, signalling "no informative tokens".
        """
        # Import lazily so unit-tests that mock the offline store do not
        # have to load the 5 MB FinBERT vocab.
        from backend.perception.semantic.tokenizer import get_tokenizer

        since = end - timedelta(hours=hours_back)
        # ``query_news_by_ticker`` already returns newest-first; we reverse
        # so the *oldest* of the 24h window comes first — this matches the
        # natural causal order a human reader would follow.
        events: list[NewsEvent] = await self._ts.query_news_by_ticker(
            ticker,
            since,
            limit=_NEWS_EVENT_BUDGET,
        )
        events.sort(key=lambda e: e.time)

        if not events:
            logger.debug(f"News window empty for {ticker} at {end.isoformat()}")
            return self._empty_news_tensor(max_length)

        # Concatenate headline + body, separated by the tokenizer's [SEP]
        # marker so the encoder can see the boundary; multiple articles
        # are concatenated end-to-end. We let the tokenizer truncate.
        parts: list[str] = []
        for event in events:
            joined = event.headline.strip()
            if event.body:
                joined += " " + event.body.strip()
            parts.append(joined)
        combined = " [SEP] ".join(parts)

        tokenizer = get_tokenizer()
        enc = tokenizer(
            combined,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
        }

    @staticmethod
    def _empty_news_tensor(max_length: int) -> dict[str, torch.Tensor]:
        """Helper: return zero-padded ``input_ids`` / ``attention_mask``."""
        return {
            "input_ids": torch.zeros((1, max_length), dtype=torch.long),
            "attention_mask": torch.zeros((1, max_length), dtype=torch.long),
        }

    # ----------------------------------------------------------- orchestrator
    async def get_training_window(
        self,
        ticker: str,
        start: datetime,
        end: datetime,
        features: list[str] | None = None,
    ) -> dict[str, Any]:
        """Fetch multiple cold features simultaneously for the same window.

        Parameters
        ----------
        ticker
            Symbol.
        start, end
            Half-open time interval ``[start, end)``. ``start`` is
            ignored for ``news_window`` (which uses ``end`` and its own
            ``hours_back`` parameter); that's by design — the news
            look-back is *attached* to the right endpoint and the
            caller's training-window left endpoint is not a meaningful
            boundary for a 24-hour rolling news context.
        features
            List of feature names. Defaults to ``["ohlcv_window"]``.
            Every entry must correspond to a *cold* feature in the
            registry (``FeatureDef.is_cold == True``); otherwise a
            ``ValueError`` is raised.

        Returns
        -------
        dict
            Keyed by feature name. Value type depends on the feature:

            * ``ohlcv_window`` → :class:`polars.DataFrame`
            * ``news_window``  → dict[str, :class:`torch.Tensor`]
        """
        requested = features or ["ohlcv_window"]

        # Validate up-front so we fail fast on typos.
        for name in requested:
            fdef = FEATURE_REGISTRY.get(name)
            if fdef is None:
                raise ValueError(
                    f"Unknown feature '{name}'. Known features: {sorted(FEATURE_REGISTRY)}"
                )
            if not fdef.is_cold:
                raise ValueError(
                    f"Feature '{name}' is not cold (source={fdef.source}); "
                    "use the online store instead."
                )

        # Issue all queries concurrently. The TimescaleDB connection pool
        # handles the parallelism; for a 2-feature dispatch the savings
        # are modest but the pattern scales without code changes when we
        # add more cold features.
        async def _fetch(name: str) -> tuple[str, Any]:
            if name == "ohlcv_window":
                return name, await self.get_ohlcv_window(ticker, start, end)
            if name == "news_window":
                return name, await self.get_news_window(ticker, end)
            # We validated above, so this branch is unreachable —
            # keep it for defensive completeness.
            raise ValueError(f"No retrieval logic for feature '{name}'")

        coros = [_fetch(name) for name in requested]
        results = await asyncio.gather(*coros)
        return dict(results)


# ----------------------------------------------------------------------------
# torch.utils.data.Dataset wrapper (legacy entry point)
# ----------------------------------------------------------------------------
class TimeSeriesDataset(Dataset):
    """Map-style PyTorch dataset over sliding OHLCV windows.

    Kept primarily for compatibility with older training scripts that
    used the synchronous ``Dataset`` interface. New code should use
    :func:`backend.perception.temporal.dataset.build_tft_loaders`, which
    wraps the async :class:`OfflineFeatureStore` properly.

    Notes
    -----
    Because :class:`Dataset.__getitem__` is synchronous and
    :class:`TimescaleStore` is async, each access runs an event loop on
    the worker thread. This is inefficient for high-throughput training
    — prefer the async iterable variant for serious work.
    """

    def __init__(
        self,
        tickers: list[str],
        start: datetime,
        end: datetime,
        window_minutes: int = OHLCV_WINDOW_MINUTES,
        stride_minutes: int = 15,
        timescale_dsn: str | None = None,
    ) -> None:
        self._tickers = tickers
        self._window = window_minutes
        self._stride = stride_minutes
        self._dsn = timescale_dsn
        self._index = self._build_index(start, end)
        self._store: TimescaleStore | None = None

    def _build_index(
        self,
        start: datetime,
        end: datetime,
    ) -> list[tuple[str, datetime]]:
        """Pre-compute every (ticker, window_start) pair the dataset exposes."""
        index: list[tuple[str, datetime]] = []
        step = timedelta(minutes=self._stride)
        for ticker in self._tickers:
            cursor = start
            while cursor + timedelta(minutes=self._window) <= end:
                index.append((ticker, cursor))
                cursor += step
        logger.info(f"TimeSeriesDataset index built: {len(index)} windows")
        return index

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, i: int) -> dict[str, Any]:
        ticker, window_start = self._index[i]
        window_end = window_start + timedelta(minutes=self._window)
        df = _run_sync(self._fetch_window(ticker, window_start, window_end))
        tensor = self._df_to_tensor(df)
        return {
            "ticker": ticker,
            "window_start_unix": torch.tensor(window_start.timestamp(), dtype=torch.float64),
            "x": tensor,
        }

    async def _fetch_window(
        self,
        ticker: str,
        start: datetime,
        end: datetime,
    ) -> pl.DataFrame:
        if self._store is None:
            self._store = TimescaleStore()
            await self._store.connect()
        return await self._store.get_historical_window(ticker, start, end, freq="1m")

    def _df_to_tensor(self, df: pl.DataFrame) -> torch.Tensor:
        """Convert a (T, 5) Polars DataFrame to a fixed-length tensor.

        If the DataFrame is *shorter* than ``self._window`` (gaps, halts),
        we *front-pad* with zeros — the model sees the most recent bars at
        the end of the sequence, which is what its causal attention mask
        expects.
        """
        channels = ["open", "high", "low", "close", "volume"]
        if df.height == 0:
            return torch.zeros((self._window, len(channels)), dtype=torch.float32)
        arr = df.select(channels).to_numpy().astype(np.float32)
        if arr.shape[0] >= self._window:
            arr = arr[-self._window :]
        else:
            pad = np.zeros((self._window - arr.shape[0], len(channels)), dtype=np.float32)
            arr = np.vstack([pad, arr])
        return torch.from_numpy(arr)


def _run_sync(coro: Any) -> Any:
    """Run an async coroutine to completion from a sync context.

    Used by :class:`TimeSeriesDataset.__getitem__`. We try to reuse the
    current event loop if there is one; otherwise we create a new loop
    on the calling thread (the DataLoader worker).
    """
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # We are inside an active event loop (e.g. inside a Jupyter
            # cell). Mixing sync ``get_item`` with an active loop would
            # deadlock the dataset; the caller should use the async
            # iterable variant instead.
            raise RuntimeError(
                "TimeSeriesDataset.__getitem__ called from inside a running "
                "event loop. Use backend.perception.temporal.dataset.build_tft_loaders "
                "instead."
            )
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(coro)
