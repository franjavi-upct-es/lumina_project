# backend/perception/temporal/dataset.py
"""Builds train/val DataLoaders for self-supervised TFT pre-training.

The pre-training task is a regression on the *future return*:

    target_t = clip( (close_{t + H} - close_t) / close_t, -0.10, 0.10 )

where ``H = OHLCV_HORIZON_MINUTES`` (60 by default). Clipping to ±10%
protects against split events and bad ticks; without the clip a single
mis-adjusted bar can blow up the gradient.

We use a strict *chronological* train/val split (never shuffle across
the boundary) because temporal data has autocorrelation: shuffling
would leak validation information into training and inflate metrics.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import polars as pl
import torch
from torch.utils.data import DataLoader, IterableDataset

from backend.config.constants import OHLCV_HORIZON_MINUTES, OHLCV_WINDOW_MINUTES
from backend.config.settings import get_settings
from backend.data_engine.storage.timescale import TimescaleStore
from backend.perception.temporal.preprocessor import preprocess_ohlcv_window

# Tickers will typically be batched together; we cap the look-back so a
# single worker does not exhaust the connection pool.
_MAX_TICKERS_PER_WORKER = 8


@dataclass(slots=True)
class _Sample:
    """In-memory representation of one training sample."""

    x: torch.Tensor  # (T, 9)
    target_return: float
    ticker: str


def _clip_return(value: float, lim: float = 0.10) -> float:
    return float(max(-lim, min(lim, value)))


class _TFTIterableDataset(IterableDataset[dict[str, Any]]):
    """Iterable dataset because the universe is large enough that a
    map-style dataset would require precomputing the entire index, and
    we want to share connections between workers.

    Workers split the ticker list among themselves; within a worker we
    stream sliding windows. Worker 0 always processes the indices that
    are 0 mod n_workers, etc.
    """

    def __init__(
        self,
        store: TimescaleStore,
        tickers: list[str],
        start: datetime,
        end: datetime,
        window_minutes: int,
        horizon_minutes: int,
        stride_minutes: int,
    ):
        super().__init__()
        self.store = store
        self.tickers = tickers
        self.start = start
        self.end = end
        self.window = window_minutes
        self.horizon = horizon_minutes
        self.stride = stride_minutes

    async def _iterate_async(self, my_tickers: list[str]) -> list[_Sample]:
        """Pull all windows for the assigned tickers."""
        out: list[_Sample] = []
        window_delta = timedelta(minutes=self.window + self.horizon)
        stride_delta = timedelta(minutes=self.stride)
        await self.store.connect()
        for ticker in my_tickers[:_MAX_TICKERS_PER_WORKER]:
            cursor = self.start
            while cursor + window_delta <= self.end:
                df: pl.DataFrame = await self.store.get_historical_window(
                    ticker, cursor, cursor + window_delta, freq="1m"
                )
                # Skip windows with too many gaps. The "expected" row count
                # is ``window + horizon``; we accept windows with ≥ 90% bars.
                expected = self.window + self.horizon
                if df.height < int(expected * 0.9):
                    cursor += stride_delta
                    continue
                arr = df.select(["close"]).to_numpy().squeeze(-1)
                if arr.size < expected:
                    cursor += stride_delta
                    continue
                x_window = df.head(self.window)
                tensor = preprocess_ohlcv_window(x_window, ticker)
                close_t = float(arr[self.window - 1])
                close_th = float(arr[self.window + self.horizon - 1])
                target = _clip_return((close_th - close_t) / close_t) if close_t else 0.0
                out.append(_Sample(x=tensor, target_return=target, ticker=ticker))
                cursor += stride_delta
        return out

    def __iter__(self) -> Iterator[dict[str, Any]]:
        import asyncio

        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            my_tickers = list(self.tickers)
        else:
            n = worker_info.num_workers
            wid = worker_info.id
            my_tickers = [t for i, t in enumerate(self.tickers) if i % n == wid]

        # NB: a worker has its own event loop; we run the async producer
        # to completion before yielding (acceptable for moderately sized
        # universes; for huge universes a streaming async impl. is needed).
        samples = asyncio.run(self._iterate_async(my_tickers))
        for s in samples:
            yield {"x": s.x, "target_return": s.target_return, "ticker": s.ticker}


def _collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Custom collate function: stacks tensors + keeps a tuple of tickers."""
    xs = torch.stack([b["x"] for b in batch])
    targets = torch.tensor([b["target_return"] for b in batch], dtype=torch.float32)
    tickers = tuple(b["ticker"] for b in batch)
    return {"x": xs, "target_return": targets, "ticker": tickers}


def build_tft_loaders(
    store: TimescaleStore,
    start: datetime,
    end: datetime,
    tickers: list[str] | None = None,
    batch_size: int | None = None,
    window_minutes: int = OHLCV_WINDOW_MINUTES,
    horizon_minutes: int = OHLCV_HORIZON_MINUTES,
    stride_minutes: int = 15,
    val_fraction: float = 0.2,
) -> tuple[DataLoader[dict[str, Any]], DataLoader[dict[str, Any]]]:
    """Chronological train/val split returning two DataLoaders.

    The split is on TIME, not on tickers — both loaders see all tickers
    but the validation period is strictly after the training period.

    Parameters
    ----------
    store : TimescaleStore
        Persistent connection to TimescaleDB.
    start, end : datetime
        Date range to draw windows from.
    tickers : list[str] | None
        Defaults to the full ``TARGET_TICKERS`` universe.
    batch_size : int | None
        Defaults to ``settings.FEATURE_STORE_BATCH_SIZE``.
    val_fraction : float
        Fraction of the time range to reserve for validation. With the
        default 0.2, the last 20 % becomes the validation set.

    Returns
    -------
    (train_loader, val_loader)
    """
    from backend.config.constants import TARGET_TICKERS

    settings = get_settings()
    tickers = tickers or sorted(TARGET_TICKERS)
    batch_size = batch_size or settings.FEATURE_STORE_BATCH_SIZE

    split = start + timedelta(seconds=(end - start).total_seconds() * (1.0 - val_fraction))

    train_ds = _TFTIterableDataset(
        store=store,
        tickers=tickers,
        start=start,
        end=split,
        window_minutes=window_minutes,
        horizon_minutes=horizon_minutes,
        stride_minutes=stride_minutes,
    )
    val_ds = _TFTIterableDataset(
        store=store,
        tickers=tickers,
        start=split,
        end=end,
        window_minutes=window_minutes,
        horizon_minutes=horizon_minutes,
        stride_minutes=stride_minutes,
    )

    train_loader: DataLoader[dict[str, Any]] = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=min(settings.FEATURE_STORE_NUM_WORKERS, len(tickers)),
        pin_memory=torch.cuda.is_available(),
        collate_fn=_collate,
    )
    val_loader: DataLoader[dict[str, Any]] = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=min(2, len(tickers)),
        pin_memory=torch.cuda.is_available(),
        collate_fn=_collate,
    )
    return train_loader, val_loader


__all__ = ["build_tft_loaders", "preprocess_ohlcv_window"]


_used = (preprocess_ohlcv_window, np, _Sample)
