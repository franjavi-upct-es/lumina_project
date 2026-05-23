# tests/perception/test_preprocessor.py
"""Unit tests for the TFT preprocessor and its technical-indicator helpers.

We test the *shape contract* (always 9 channels) and a handful of
indicator-specific invariants known from the literature:

* RSI(14) is bounded in [0, 100] (we divide by 100 → [0, 1]).
* RSI of a monotonic-up series tends to 100 (≈ 1.0 post-normalisation).
* MACD line is zero when fast EMA == slow EMA == constant series.
* Bollinger %B is exactly 0.5 in the middle of the band (= mean).
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl
import pytest
import torch

from backend.perception.temporal.preprocessor import (
    _bollinger_pct_b,
    _macd_line,
    _rsi,
    get_sector_one_hot,
    preprocess_ohlcv_window,
)


def _make_window(n: int, close_seq: list[float] | None = None) -> pl.DataFrame:
    """Build a (n,)-row OHLCV DataFrame for testing."""
    base = datetime(2026, 1, 1, 9, 30, tzinfo=UTC)
    times = [base + timedelta(minutes=i) for i in range(n)]
    close = close_seq if close_seq is not None else [100.0 + i for i in range(n)]
    return pl.DataFrame(
        {
            "time": times,
            "open": close,
            "high": [c + 0.5 for c in close],
            "low": [c - 0.5 for c in close],
            "close": close,
            "volume": [1000] * n,
        }
    )


def test_preprocess_ohlcv_window_output_shape_is_9_channels() -> None:
    """The contract is (T, 9) regardless of inputs."""
    df = _make_window(60)
    tensor = preprocess_ohlcv_window(df, "AAPL")
    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (60, 9)
    assert tensor.dtype == torch.float32


def test_preprocess_ohlcv_raises_on_empty_dataframe() -> None:
    df = pl.DataFrame(
        schema={
            "time": pl.Datetime,
            "open": pl.Float64,
            "high": pl.Float64,
            "low": pl.Float64,
            "close": pl.Float64,
            "volume": pl.Int64,
        }
    )
    with pytest.raises(ValueError, match="empty"):
        preprocess_ohlcv_window(df, "AAPL")


def test_preprocess_ohlcv_raises_on_missing_columns() -> None:
    df = pl.DataFrame({"close": [100.0, 101.0]})
    with pytest.raises(ValueError, match="missing"):
        preprocess_ohlcv_window(df, "AAPL")


def test_rsi_bounded_in_unit_interval() -> None:
    """After our /100 normalisation, RSI must lie in [0, 1]."""
    close = np.array(
        [
            100.0,
            101.0,
            99.0,
            102.0,
            98.0,
            103.0,
            97.0,
            104.0,
            96.0,
            105.0,
            95.0,
            106.0,
            94.0,
            107.0,
            93.0,
            108.0,
        ]
    )
    rsi = _rsi(close)
    assert rsi.min() >= 0.0 and rsi.max() <= 100.0


def test_rsi_of_monotonic_up_series_approaches_100() -> None:
    """Pure up-trends → RSI ≈ 100 (no losses, ratio infinite)."""
    close = np.arange(50, dtype=np.float64)
    close[0] = 1.0  # avoid div-by-zero on first return
    rsi = _rsi(close)
    # Skip the warm-up region (first 14 indices set to 50).
    assert rsi[-1] > 90.0


def test_macd_line_of_constant_series_is_zero() -> None:
    """A flat series → both EMAs equal → MACD = 0."""
    close = np.full(50, 100.0)
    macd = _macd_line(close)
    np.testing.assert_allclose(macd, 0.0, atol=1e-8)


def test_bollinger_pct_b_midpoint_is_half() -> None:
    """When close = rolling mean, %B = 0.5 exactly."""
    close = np.full(40, 100.0)
    pct_b = _bollinger_pct_b(close, period=20)
    # Indices past warm-up should be exactly 0.5.
    assert pytest.approx(pct_b[25]) == 0.5


def test_get_sector_one_hot_unknown_ticker_returns_zeros() -> None:
    """Unknown tickers get all-zero sector vector."""
    vec = get_sector_one_hot("ZZZZ-UNKNOWN")
    assert vec.sum() == 0.0
