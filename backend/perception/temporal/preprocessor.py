# backend/perception/temporal/preprocessor.py
"""Per-window OHLCV normalisation and technical-indicator computation.

This is the layer that prepares raw OHLCV windows for the
:class:`backend.perception.temporal.tft_model.TemporalFusionTransformer`.

Design choices
==============

1. **Per-window normalisation, never global.**
   We MinMax-scale each window independently. Global normalisation
   would leak the test-set distribution into the training set (a future
   maximum becomes available to the model when normalising a past
   window). Per-window scaling makes every input live in the same
   bounded range *without* peeking ahead.

2. **Indicators computed on the *unnormalised* close series.**
   RSI, MACD and Bollinger %B are scale-invariant when computed
   correctly, but doing it on the normalised series adds numerical
   noise. We compute them first, then attach.

3. **Final tensor layout: (T, 9) per window.**
   * channels 0..4: normalised OHLCV (open, high, low, close, volume)
   * channel  5   : RSI(14) divided by 100 → range [0, 1]
   * channel  6   : MACD line (already small) divided by ATR for scale
   * channel  7   : Bollinger %B (already in [0, 1] under normal regimes)
   * channel  8   : market-cap tier ∈ {0, 1, 2, 3, 4} divided by 4

The number 9 is then handed to the TFT's
:class:`~backend.perception.temporal.tft_model.VariableSelectionNetwork`
as ``num_features``.

References
----------
* Wilder (1978), *New Concepts in Technical Trading Systems* — RSI.
* Appel (2005), *Technical Analysis: Power Tools for Active Investors* — MACD.
* Bollinger (1980), *Bollinger on Bollinger Bands*.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import torch

from backend.config.constants import NUM_SECTORS, TICKER_TO_SECTOR

# Default market-cap tier when the ticker is missing from our static
# covariate file. Conservative: "small-cap unknown".
_DEFAULT_CAP_TIER: int = 1

# Static market-cap tier per ticker. In production this would live in a
# YAML file under ``backend/config/static_covariates.yml`` and be reloaded
# at process start; for now we hard-code the universe tiers.
#   0 = micro (< $300M)
#   1 = small ($300M – $2B)
#   2 = mid ($2B – $10B)
#   3 = large ($10B – $200B)
#   4 = mega (> $200B)
_MARKET_CAP_TIER: dict[str, int] = {
    "AAPL": 4,
    "MSFT": 4,
    "NVDA": 4,
    "GOOGL": 4,
    "META": 4,
    "AMZN": 4,
    "TSLA": 4,
    "AVGO": 4,
    "ORCL": 3,
    "ADBE": 3,
    "NFLX": 3,
    "CRM": 3,
    "AMD": 3,
    "INTC": 3,
    "JPM": 4,
    "BAC": 3,
    "GS": 3,
    "MS": 3,
    "V": 4,
    "MA": 4,
    "XOM": 4,
    "CVX": 3,
    "COP": 3,
    "JNJ": 4,
    "PFE": 3,
    "UNH": 4,
    "ABBV": 3,
    "WMT": 4,
    "COST": 4,
    "HD": 4,
    "SPY": 4,
    "QQQ": 4,
}


# -------------------------------------------------------------------- helpers
def _rsi(close: np.ndarray, period: int = 14) -> np.ndarray:
    """Wilder's Relative Strength Index.

    The original Wilder formulation uses an exponential smoothing with
    α = 1/period (rather than 2/(period+1)). We replicate the original
    so our values match TA-Lib / TradingView defaults.
    """
    if close.size < 2:
        return np.zeros_like(close)
    delta = np.diff(close, prepend=close[0])
    gains = np.where(delta > 0, delta, 0.0)
    losses = np.where(delta < 0, -delta, 0.0)

    avg_gain = np.zeros_like(close, dtype=np.float64)
    avg_loss = np.zeros_like(close, dtype=np.float64)
    avg_gain[period] = gains[1 : period + 1].mean()
    avg_loss[period] = losses[1 : period + 1].mean()
    alpha = 1.0 / period
    for t in range(period + 1, close.size):
        avg_gain[t] = (1 - alpha) * avg_gain[t - 1] + alpha * gains[t]
        avg_loss[t] = (1 - alpha) * avg_loss[t - 1] + alpha * losses[t]

    # RSI = 100 - 100 / (1 + RS) where RS = avg_gain / avg_loss.
    #
    # Three edge cases to handle correctly:
    #   * avg_loss > 0  → standard formula
    #   * avg_loss = 0 and avg_gain > 0 → no losses at all → RSI = 100
    #   * avg_loss = 0 and avg_gain = 0 → no movement at all → RSI = 50
    rsi = np.full_like(avg_gain, 50.0)
    no_loss = avg_loss == 0
    has_gain = avg_gain > 0
    # Default branch (avg_loss > 0): standard formula.
    safe = ~no_loss
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=safe)
    rsi = np.where(safe, 100.0 - 100.0 / (1.0 + rs), rsi)
    # Pure-gain branch: RSI saturates at 100.
    rsi = np.where(no_loss & has_gain, 100.0, rsi)
    # Warm-up region (before `period`) is undefined → fill with neutral 50.
    rsi[:period] = 50.0
    return rsi.astype(np.float32)


def _macd_line(close: np.ndarray, fast: int = 12, slow: int = 26) -> np.ndarray:
    """MACD line = EMA(fast) - EMA(slow).

    We return only the line; the signal and histogram are downstream of
    the same EMAs and add little information given the TFT will compute
    its own temporal aggregates internally.
    """
    if close.size == 0:
        return np.zeros(0, dtype=np.float32)

    def _ema(x: np.ndarray, span: int) -> np.ndarray:
        alpha = 2.0 / (span + 1.0)
        out = np.empty_like(x, dtype=np.float64)
        out[0] = x[0]
        for t in range(1, x.size):
            out[t] = alpha * x[t] + (1 - alpha) * out[t - 1]
        return out

    return (_ema(close, fast) - _ema(close, slow)).astype(np.float32)


def _bollinger_pct_b(close: np.ndarray, period: int = 20, n_std: float = 2.0) -> np.ndarray:
    """Bollinger %B = (close − lower) / (upper − lower).

    Bands are computed on the rolling mean ± n_std × rolling stdev.
    %B is dimensionless and usually sits in [0, 1] (or briefly outside
    during breakouts). We clip to [-0.5, 1.5] to bound the input range.
    """
    n = close.size
    if n < period:
        return np.full(n, 0.5, dtype=np.float32)
    out = np.full(n, 0.5, dtype=np.float64)
    for t in range(period - 1, n):
        window = close[t - period + 1 : t + 1]
        m = window.mean()
        s = window.std(ddof=0)
        upper = m + n_std * s
        lower = m - n_std * s
        if upper - lower > 1e-9:
            out[t] = (close[t] - lower) / (upper - lower)
    return np.clip(out, -0.5, 1.5).astype(np.float32)


def _minmax(window: np.ndarray) -> np.ndarray:
    """MinMax scale a (T, F) array per-feature. Constant columns → zeros."""
    mn = window.min(axis=0, keepdims=True)
    mx = window.max(axis=0, keepdims=True)
    rng = mx - mn
    rng = np.where(rng > 1e-9, rng, 1.0)
    return (window - mn) / rng


# -------------------------------------------------------------------- public
def preprocess_ohlcv_window(df: pl.DataFrame, ticker: str) -> torch.Tensor:
    """Turn a Polars OHLCV window into a (T, 9) torch.Tensor[float32].

    Parameters
    ----------
    df : pl.DataFrame
        Must contain columns ``open, high, low, close, volume`` and have
        at least 2 rows. Other columns (``vwap``, ``trade_count``) are
        ignored.
    ticker : str
        Used to look up the market-cap tier static covariate.

    Returns
    -------
    torch.Tensor of shape (T, 9), dtype float32.

    Raises
    ------
    ValueError
        If ``df`` does not have the required columns or is empty.
    """
    required = {"open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"preprocess_ohlcv_window: missing columns {missing}")
    if df.height == 0:
        raise ValueError("preprocess_ohlcv_window: empty DataFrame")

    arr = df.select(["open", "high", "low", "close", "volume"]).to_numpy().astype(np.float64)
    close = arr[:, 3]

    # --- Technical indicators on the raw close series ---------------------
    rsi = _rsi(close) / 100.0  # in [0, 1]

    macd = _macd_line(close)
    # MACD scale ≈ a few units of price; normalise by ATR-like rolling range
    range_est = arr[:, 1] - arr[:, 2]  # high - low
    scale = float(np.mean(range_est)) if range_est.size else 1.0
    macd_norm = (macd / (scale + 1e-9)).astype(np.float32)

    pct_b = _bollinger_pct_b(close)

    # --- MinMax-scaled OHLCV ----------------------------------------------
    ohlcv_norm = _minmax(arr).astype(np.float32)  # (T, 5)

    # --- Static covariate -------------------------------------------------
    cap_tier = _MARKET_CAP_TIER.get(ticker.upper(), _DEFAULT_CAP_TIER) / 4.0
    cap_col = np.full((arr.shape[0], 1), cap_tier, dtype=np.float32)

    out = np.concatenate(
        [
            ohlcv_norm,
            rsi[:, None].astype(np.float32),
            macd_norm[:, None],
            pct_b[:, None],
            cap_col,
        ],
        axis=1,
    )
    assert out.shape[1] == 9, f"Unexpected feature count: {out.shape[1]}"
    return torch.from_numpy(out)


def get_sector_one_hot(ticker: str) -> np.ndarray:
    """Return the sector one-hot vector for a ticker.

    Used by the GATv2 graph builder as a node feature. Living here keeps
    every "static covariate" function in one module.
    """
    out = np.zeros(NUM_SECTORS, dtype=np.float32)
    sector = TICKER_TO_SECTOR.get(ticker.upper())
    if sector is None:
        return out
    from backend.config.constants import SECTORS

    if sector in SECTORS:
        out[SECTORS.index(sector)] = 1.0
    return out
