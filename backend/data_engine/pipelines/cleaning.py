# backend/data_engine/pipelines/cleaning.py
"""Pure data cleaning functions. No I/O, no state, no side effects."""

from __future__ import annotations

import hashlib
import re
import unicodedata
from typing import Literal

import polars as pl
from bs4 import BeautifulSoup
from loguru import logger

from backend.config.constants import TARGET_TICKERS
from backend.data_engine.storage.timescale import NewsEvent

OutlierMethod = Literal["iqr", "zscore"]
_MAX_FORWARD_FILL_MINUTES = 5
_MAX_NEWS_BODY_CHARS = 2048
_ZSCORE_THRESHOLD = 5.0
_IQR_MULTIPLIER = 3.0


def detect_price_outliers(df: pl.DataFrame, method: OutlierMethod = "zscore") -> pl.Series:
    if df.height < 3:
        return pl.Series("is_outlier", [False] * df.height)
    returns = df["close"].pct_change().fill_null(0.0)
    if method == "zscore":
        mean = returns.mean() or 0.0
        std = returns.std() or 1.0
        if std == 0:
            return pl.Series("is_outlier", [False] * df.height)
        z = (returns - mean).abs() / std
        return z > _ZSCORE_THRESHOLD
    q1 = returns.quantile(0.25) or 0.0
    q3 = returns.quantile(0.75) or 0.0
    iqr = q3 - q1
    return (returns < q1 - _IQR_MULTIPLIER * iqr) | (returns > q3 + _IQR_MULTIPLIER * iqr)


def _validate_ohlc_consistency(df: pl.DataFrame) -> pl.DataFrame:
    return df.filter(
        (pl.col("high") >= pl.max_horizontal("open", "close"))
        & (pl.col("low") <= pl.min_horizontal("open", "close"))
        & (pl.col("volume") >= 0)
    )


def clean_ohlcv(df: pl.DataFrame, ticker: str) -> pl.DataFrame:
    if df.height == 0:
        return df
    before = df.height
    df = _validate_ohlc_consistency(df)
    dropped = before - df.height
    if dropped > 0:
        logger.warning(f"[{ticker}] dropped {dropped} rows failing OHLC consistency")
    if df.height >= 3:
        mask = detect_price_outliers(df, method="zscore")
        n_out = int(mask.sum() or 0)
        if n_out > 0:
            df = df.filter(~mask)
            logger.info(f"[{ticker}] removed {n_out} price outliers")
    df = df.sort("time")
    df = df.with_columns(
        [
            pl.col(c).forward_fill(limit=_MAX_FORWARD_FILL_MINUTES)
            for c in ("open", "high", "low", "close", "vwap")
        ]
    )
    df = df.drop_nulls(subset=["open", "high", "low", "close"])
    return df


_WHITESPACE_RE = re.compile(r"\s+")


def clean_news_text(raw: str | None) -> str:
    if not raw:
        return ""
    soup = BeautifulSoup(raw, "html.parser")
    for tag in soup(["script", "style", "iframe"]):
        tag.decompose()
    text = soup.get_text(separator=" ")
    text = unicodedata.normalize("NFKC", text)
    text = _WHITESPACE_RE.sub(" ", text).strip()
    if len(text) > _MAX_NEWS_BODY_CHARS:
        text = text[:_MAX_NEWS_BODY_CHARS].rsplit(" ", 1)[0]
    return text


def compute_content_hash(source: str, headline: str, body: str | None) -> str:
    body_part = (body or "")[:200]
    material = f"{source}||{headline}||{body_part}".encode()
    return hashlib.sha256(material).hexdigest()


def dedupe_news_events(events: list[NewsEvent]) -> list[NewsEvent]:
    seen: set[str] = set()
    out: list[NewsEvent] = []
    for e in events:
        if e.content_hash in seen:
            continue
        seen.add(e.content_hash)
        out.append(e)
    return out


def validate_supply_chain_edge(edge: dict) -> bool:
    src = edge.get("source_ticker")
    tgt = edge.get("target_ticker")
    if not src or not tgt or src == tgt:
        return False
    return src in TARGET_TICKERS and tgt in TARGET_TICKERS
