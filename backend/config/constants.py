"""Immutable project constants. Single source of truth for dimensions, tickers, etc."""

from __future__ import annotations

# ---------- Embedding dimensions ----------
DIM_PRICE = 128  # TFT output
DIM_SEMANTIC = 64  # Distilled LLM output
DIM_GRAPH = 32  # GATv2 output

# ---------- Temporal windows ----------
OHLCV_WINDOW_MINUTES = 240  # 4 hours of context for TFT

# ---------- Universe ----------
# TODO: expand to fill 50-ticker universe post-Phase 1
TARGET_TICKERS: frozenset[str] = frozenset(
    {
        "AAPL",
        "MSFT",
        "NVDA",
        "GOOGL",
        "META",
        "AMZN",
        "TSLA",
        "AMD",
        "INTC",
        "AVGO",
        "CRM",
        "ORCL",
        "ADBE",
        "NFLX",
        "JPM",
        "BAC",
        "GS",
        "MS",
        "V",
        "MA",
        "XOM",
        "CVX",
        "COP",
        "JNJ",
        "PFE",
        "UNH",
        "ABBV",
        "WMT",
        "COST",
        "HD",
        "SPY",
        "QQQ",
    }
)

# ---------- Market hours (NYSE, UTC) ----------
MARKET_OPEN_HOUR_UTC = 13  # 9:30 EST = 13:30 UTC (EDT in summer: 12:30; simplified)
MARKET_OPEN_MINUTE = 30
MARKET_CLOSE_HOUR_UTC = 20  # 4:00 PM EST
MARKET_CLOSE_MINUTE = 0

# ---------- Sector map (simplified; expand in production) ----------
TICKER_TO_SECTOR: dict[str, str] = {
    "AAPL": "tech",
    "MSFT": "tech",
    "NVDA": "tech",
    "GOOGL": "tech",
    "META": "tech",
    "AMZN": "tech",
    "TSLA": "consumer_discretionary",
    "AMD": "tech",
    "INTC": "tech",
    "AVGO": "tech",
    "CRM": "tech",
    "ORCL": "tech",
    "ADBE": "tech",
    "NFLX": "tech",
    "JPM": "financials",
    "BAC": "financials",
    "GS": "financials",
    "MS": "financials",
    "V": "financials",
    "MA": "financials",
    "XOM": "energy",
    "CVX": "energy",
    "COP": "energy",
    "JNJ": "healthcare",
    "PFE": "healthcare",
    "UNH": "healthcare",
    "ABBV": "healthcare",
    "WMT": "consumer_staples",
    "COST": "consumer_staples",
    "HD": "consumer_discretionary",
    "SPY": "index",
    "QQQ": "index",
}
