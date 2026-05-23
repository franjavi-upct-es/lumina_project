# backend/perception/structural/graph_builder.py
"""Construct the financial graph fed to the GATv2 encoder.

Output: a :class:`torch_geometric.data.Data` object with

    * ``x``          : (N, 32) node features
    * ``edge_index`` : (2, E) edge list
    * ``edge_attr``  : (E, 4) edge features
    * ``ticker``     : list[str] of length N — keeps node ↔ ticker map

Two types of edges are merged:

1. **Supply-chain edges** — extracted from 10-K filings by
   :mod:`backend.data_engine.collectors.chain_scrapers`. These are
   *directional* but the GATv2 treats the graph as a digraph, which
   matches the semantics ("AAPL is a customer of TSM" ⇏ vice-versa).

2. **Correlation edges** — built from a rolling Pearson correlation of
   1-day log returns. We threshold at |r| > 0.5 because finance prices
   are noisy and unconditional pairs with weaker correlation usually
   reflect chance, not co-movement.

Node features (32-D, intentionally padded so the GATv2 input shape is
fixed regardless of how many sectors we eventually onboard):

    channels 0..NUM_SECTORS - 1 : one-hot sector
    channel  NUM_SECTORS        : log10(market_cap_usd) / 13   (≈ [0, 1])
    channel  NUM_SECTORS + 1    : annualised 60-day realised volatility
    channel  NUM_SECTORS + 2    : beta vs SPY
    remaining channels          : zero-padded to 32

Edge features (4-D):

    [r, |r|, is_supply_chain (0/1), weight]

where ``weight`` is the supply-chain edge weight when applicable, or
``max(r, 0)`` when the edge is correlation-only.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta

import numpy as np
import polars as pl
import torch
from loguru import logger
from torch_geometric.data import Data

from backend.config.constants import NUM_SECTORS, TARGET_TICKERS
from backend.data_engine.storage.redis_cache import (
    RedisCache,
)
from backend.data_engine.storage.timescale import TimescaleStore
from backend.perception.temporal.preprocessor import _MARKET_CAP_TIER, get_sector_one_hot

# Number of trading days to use in the correlation window.
_CORR_WINDOW_DAYS: int = 60
# Pearson correlation threshold above which we add a dynamic edge.
_CORR_THRESHOLD: float = 0.5
# Synthetic market caps used as a fallback when no external feed is
# available. Numbers are billions of USD as of late 2024 (rough).
_FALLBACK_MARKET_CAP_USD: dict[str, float] = {
    "AAPL": 3_500e9,
    "MSFT": 3_100e9,
    "NVDA": 3_300e9,
    "GOOGL": 2_100e9,
    "META": 1_300e9,
    "AMZN": 2_000e9,
    "TSLA": 800e9,
    "AVGO": 750e9,
    "ORCL": 460e9,
    "ADBE": 250e9,
    "NFLX": 290e9,
    "CRM": 290e9,
    "AMD": 270e9,
    "INTC": 120e9,
    "JPM": 600e9,
    "BAC": 320e9,
    "GS": 160e9,
    "MS": 170e9,
    "V": 580e9,
    "MA": 460e9,
    "XOM": 510e9,
    "CVX": 290e9,
    "COP": 130e9,
    "JNJ": 380e9,
    "PFE": 160e9,
    "UNH": 540e9,
    "ABBV": 320e9,
    "WMT": 700e9,
    "COST": 410e9,
    "HD": 410e9,
    "SPY": 500e9,
    "QQQ": 200e9,
}


async def _fetch_close_matrix(
    store: TimescaleStore,
    tickers: list[str],
    as_of: datetime,
) -> tuple[np.ndarray, list[str]]:
    """Return an (T, N) matrix of close prices.

    Tickers without data in the look-back window are dropped from the
    returned list; this keeps the matrix dense and avoids NaN handling
    inside the correlation computation.
    """
    end = as_of
    start = as_of - timedelta(days=_CORR_WINDOW_DAYS * 2)  # generous to allow weekends/holidays
    per_ticker: dict[str, np.ndarray] = {}
    for ticker in tickers:
        df: pl.DataFrame = await store.get_historical_window(ticker, start, end, freq="1d")
        if df.height < _CORR_WINDOW_DAYS // 2:
            continue
        closes = df.select("close").to_numpy().squeeze(-1).astype(np.float64)
        per_ticker[ticker] = closes

    if not per_ticker:
        logger.warning("graph_builder: no price data fetched at all")
        return np.zeros((0, 0), dtype=np.float64), []

    min_len = min(len(v) for v in per_ticker.values())
    kept_tickers = list(per_ticker.keys())
    matrix = np.stack([per_ticker[t][-min_len:] for t in kept_tickers], axis=1)
    return matrix, kept_tickers


def _build_node_features(
    tickers: list[str],
    returns: np.ndarray,
    spy_returns: np.ndarray | None,
) -> np.ndarray:
    """Compute the (N, 32) node-feature matrix."""
    n = len(tickers)
    features = np.zeros((n, 32), dtype=np.float32)
    for i, ticker in enumerate(tickers):
        features[i, :NUM_SECTORS] = get_sector_one_hot(ticker)
        market_cap = _FALLBACK_MARKET_CAP_USD.get(ticker.upper(), 1e9)
        features[i, NUM_SECTORS] = math.log10(market_cap) / 13.0  # ~[0, 1]
        ticker_ret = returns[:, i]
        features[i, NUM_SECTORS + 1] = float(np.std(ticker_ret) * math.sqrt(252))
        if spy_returns is not None and len(spy_returns) == len(ticker_ret):
            var_spy = float(np.var(spy_returns))
            cov = float(np.cov(ticker_ret, spy_returns, ddof=0)[0, 1])
            features[i, NUM_SECTORS + 2] = cov / var_spy if var_spy > 1e-12 else 0.0
        # Channel NUM_SECTORS + 3: market-cap tier mirror (redundant with above
        # but kept for backward compatibility with the preprocessor signature).
        features[i, NUM_SECTORS + 3] = _MARKET_CAP_TIER.get(ticker.upper(), 1) / 4.0
    return features


def _build_edges(
    tickers: list[str],
    returns: np.ndarray,
    supply_chain_edges: list[tuple[str, str, float]],
) -> tuple[np.ndarray, np.ndarray]:
    """Combine supply-chain + correlation edges. Returns (edge_index, edge_attr)."""
    ticker_to_idx = {t: i for i, t in enumerate(tickers)}
    n = len(tickers)
    corr = np.corrcoef(returns.T)  # (N, N)

    edges_dict: dict[tuple[int, int], list[float]] = {}

    # 1. Supply-chain edges first (so they "win" the merge).
    for src, tgt, weight in supply_chain_edges:
        if src == tgt or src not in ticker_to_idx or tgt not in ticker_to_idx:
            continue
        i, j = ticker_to_idx[src], ticker_to_idx[tgt]
        r = float(corr[i, j]) if n > 1 else 0.0
        edges_dict[(i, j)] = [r, abs(r), 1.0, float(weight)]

    # 2. Correlation edges, only add where the supply-chain edge does not exist.
    for i in range(n):
        for j in range(n):
            if i == j or (i, j) in edges_dict:
                continue
            r = float(corr[i, j])
            if abs(r) > _CORR_THRESHOLD:
                edges_dict[(i, j)] = [r, abs(r), 0.0, max(r, 0.0)]

    if not edges_dict:
        return np.zeros((2, 0), dtype=np.int64), np.zeros((0, 4), dtype=np.float32)

    edge_pairs = np.array(list(edges_dict.keys()), dtype=np.int64).T  # (2, E)
    edge_attrs = np.array(list(edges_dict.values()), dtype=np.float32)  # (E, 4)
    return edge_pairs, edge_attrs


async def build_graph_data(
    as_of: datetime,
    store: TimescaleStore,
    redis: RedisCache | None = None,
    universe: list[str] | None = None,
) -> Data:
    """Build the graph snapshot for a given point in time.

    The ``redis`` argument is accepted for forward compatibility with a
    caching layer; the current implementation does not use it.
    """
    universe = universe or sorted(TARGET_TICKERS)
    closes, kept = await _fetch_close_matrix(store, universe, as_of)
    if closes.size == 0:
        # Empty graph: still returns a valid Data object so downstream
        # code doesn't break — but the GATv2 will have no nodes.
        return Data(
            x=torch.zeros((0, 32), dtype=torch.float32),
            edge_index=torch.zeros((2, 0), dtype=torch.long),
            edge_attr=torch.zeros((0, 4), dtype=torch.float32),
            ticker=[],
        )

    # log returns
    log_returns = np.diff(np.log(closes), axis=0)

    spy_idx = kept.index("SPY") if "SPY" in kept else None
    spy_returns = log_returns[:, spy_idx] if spy_idx is not None else None

    features = _build_node_features(kept, log_returns, spy_returns)
    _sc_nodes, sc_edges = await store.get_supply_chain_graph(as_of)
    edge_index, edge_attr = _build_edges(kept, log_returns, sc_edges)

    # Sanity checks (defensive: these would indicate upstream bugs).
    assert features.shape == (len(kept), 32), features.shape
    assert edge_index.ndim == 2 and edge_index.shape[0] == 2
    assert edge_attr.shape[1] == 4

    return Data(
        x=torch.from_numpy(features),
        edge_index=torch.from_numpy(edge_index),
        edge_attr=torch.from_numpy(edge_attr),
        ticker=list(kept),
    )
