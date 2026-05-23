# tests/data_engine/pipelines/test_cleaning.py
"""Unit tests for data cleaning functions."""

from __future__ import annotations

from datetime import UTC, datetime

import polars as pl

from backend.data_engine.pipelines.cleaning import (
    clean_news_text,
    clean_ohlcv,
    compute_content_hash,
    dedupe_news_events,
    detect_price_outliers,
    validate_supply_chain_edge,
)
from backend.data_engine.storage.timescale import NewsEvent


def test_compute_content_hash_is_stable() -> None:
    """Hashing the same (source, headline, body) twice → identical SHA-256."""
    a = compute_content_hash("reuters", "Apple earnings", "record revenue")
    b = compute_content_hash("reuters", "Apple earnings", "record revenue")
    assert a == b
    assert len(a) == 64


def test_compute_content_hash_differs_on_source() -> None:
    """Changing the publisher produces a different hash."""
    a = compute_content_hash("reuters", "Apple earnings", "x")
    b = compute_content_hash("bloomberg", "Apple earnings", "x")
    assert a != b


def test_clean_news_text_strips_html_and_whitespace() -> None:
    raw = "<p>Apple   <b>reports</b>\n\nrecord earnings</p>"
    cleaned = clean_news_text(raw)
    assert "<" not in cleaned and ">" not in cleaned
    assert "  " not in cleaned


def test_clean_news_text_handles_none() -> None:
    assert clean_news_text(None) == ""


def test_dedupe_news_events_removes_duplicates() -> None:
    """Events with the same content hash are deduplicated."""
    shared_hash = compute_content_hash("reuters", "h", "b")
    a = NewsEvent(
        time=datetime(2026, 1, 1, tzinfo=UTC),
        tickers=["AAPL"],
        source="reuters",
        headline="h",
        body="b",
        content_hash=shared_hash,
    )
    b = NewsEvent(
        time=datetime(2026, 1, 2, tzinfo=UTC),
        tickers=["AAPL"],
        source="reuters",
        headline="h",
        body="b",
        content_hash=shared_hash,
    )
    out = dedupe_news_events([a, b])
    assert len(out) == 1


def test_validate_supply_chain_edge_accepts_normal_edge() -> None:
    """Two tickers in TARGET_TICKERS (AAPL → NVDA) pass."""
    edge = {"source_ticker": "AAPL", "target_ticker": "NVDA", "relationship_type": "supplier"}
    assert validate_supply_chain_edge(edge) is True


def test_validate_supply_chain_edge_rejects_self_loop() -> None:
    edge = {"source_ticker": "AAPL", "target_ticker": "AAPL", "relationship_type": "x"}
    assert validate_supply_chain_edge(edge) is False


def test_validate_supply_chain_edge_rejects_unknown_ticker() -> None:
    """Unknown ticker → reject (not silently accept)."""
    edge = {"source_ticker": "AAPL", "target_ticker": "ZZZZ", "relationship_type": "x"}
    assert validate_supply_chain_edge(edge) is False


def test_detect_price_outliers_zscore_flags_extreme_returns() -> None:
    """A single giant spike in a low-noise series gets flagged at 5σ.

    With 50+ calm points before the spike, σ stays tight enough that the
    spike's z-score comfortably clears the 5σ threshold. The spike is
    1000× the calm-regime tick size to leave generous headroom.
    """
    # 50 ticks moving by 0.01 each → returns ~1e-4 (very calm regime).
    close = [100.0 + 0.01 * i for i in range(50)]
    # The spike: jump from 100.49 to 500.0 (return ≈ +4.0, i.e. 400%).
    close.append(500.0)
    # A few more calm ticks after the spike so the moving window stays
    # representative.
    close.extend([100.5 + 0.01 * i for i in range(5)])
    df = pl.DataFrame({"close": close})
    mask = detect_price_outliers(df, method="zscore")
    # The spike is at index 50.
    assert bool(mask.to_list()[50]) is True


def test_detect_price_outliers_iqr_method_runs() -> None:
    close = [100.0 + 0.1 * i for i in range(20)] + [500.0]
    df = pl.DataFrame({"close": close})
    mask = detect_price_outliers(df, method="iqr")
    assert mask.len() == df.height


def test_clean_ohlcv_removes_invalid_bars() -> None:
    """Bars where high < open are dropped as inconsistent.

    clean_ohlcv forward-fills the vwap column, so the input DF must
    include the full schema (time/open/high/low/close/volume/vwap).
    """
    df = pl.DataFrame(
        {
            "time": [
                datetime(2026, 1, 1, 9, 30, tzinfo=UTC),
                datetime(2026, 1, 1, 9, 31, tzinfo=UTC),
            ],
            "open": [100.0, 100.0],
            "high": [101.0, 99.0],  # row 1 invalid: high < open
            "low": [99.0, 98.0],
            "close": [100.5, 99.5],
            "volume": [1000, 1000],
            "vwap": [100.5, 99.5],
        }
    )
    cleaned = clean_ohlcv(df, "AAPL")
    assert cleaned.height == 1
