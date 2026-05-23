# tests/data_engine/collectors/test_yfinance_collector.py
"""Tests for the yfinance → Polars schema transformer.

We don't actually call the yfinance network — we hand-build a small
pandas DataFrame that matches the shape ``yf.download`` returns, and
verify the transformer produces our canonical Polars schema.
"""

from __future__ import annotations

from datetime import datetime

import pandas as pd
import polars as pl

from backend.data_engine.collectors.yfinance_collector import YFinanceCollector


def _make_fake_yf_dataframe(n: int = 3) -> pd.DataFrame:
    """Mimic the ``yf.download`` output for ``interval='1d'``."""
    return pd.DataFrame(
        {
            "Open": [100.0 + i for i in range(n)],
            "High": [101.0 + i for i in range(n)],
            "Low": [99.0 + i for i in range(n)],
            "Close": [100.5 + i for i in range(n)],
            "Volume": [1_000_000 + i * 100 for i in range(n)],
        },
        index=pd.DatetimeIndex(
            [datetime(2026, 1, 1 + i) for i in range(n)],
            name="Date",
        ),
    )


def test_yf_to_polars_empty_returns_correct_schema() -> None:
    df = YFinanceCollector._yf_to_polars(pd.DataFrame())
    assert isinstance(df, pl.DataFrame)
    assert df.height == 0
    assert set(df.columns) >= {"time", "open", "high", "low", "close", "volume"}


def test_yf_to_polars_renames_columns_and_preserves_values() -> None:
    raw = _make_fake_yf_dataframe(n=3)
    out = YFinanceCollector._yf_to_polars(raw)
    assert isinstance(out, pl.DataFrame)
    assert out.height == 3
    assert "time" in out.columns
    # Values preserved (compare via to_list which converts cleanly).
    assert out["open"].to_list() == [100.0, 101.0, 102.0]
    assert out["close"].to_list() == [100.5, 101.5, 102.5]
