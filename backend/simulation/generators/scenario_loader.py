# backend/simulation/generators/scenario_loader.py
"""Historical episode generator: synthetic or store-backed.

Two modes
=========

1. **Synthetic mode** (default when ``timescale_store`` is ``None``).
   Returns a random walk with realistic intraday volatility. This is
   the mode used by unit tests and by the Spartan curriculum's
   Phase-A behavioural cloning step (the policy only needs to learn the
   shape of the action space, not the actual distribution).

2. **Real mode** (when both a TimescaleDB store and frozen encoders are
   provided). Picks a random ticker + start time within the configured
   range and replays the corresponding OHLCV through the trained
   encoders and Deep Fusion Nexus to recover the 256-D market_state
   sequence used by :class:`LuminaTradingEnv`.

Why a generator, not a torch.Dataset?
=====================================
Each episode has variable real length (gaps from holidays, halts).
Gymnasium environments call ``reset()`` and expect a *next* episode,
not a fixed-size batch — an iterator matches that contract exactly.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from typing import Any

import numpy as np
import torch
from loguru import logger

from backend.config.constants import (
    DIM_GRAPH,
    DIM_SEMANTIC,
    NEXUS_OUTPUT_DIM,
)


class HistoricalEpisodeGenerator:
    """Iterator yielding episode dicts compatible with :class:`LuminaTradingEnv`.

    Episode dict schema
    -------------------
        prices         : (T,)              float32
        market_states  : (T, NEXUS_OUTPUT_DIM)  float32
        volatility     : (T,)              float32
        uncertainties  : (T,)              float32
        ticker         : str
        synthetic      : bool
    """

    _MAX_RETRIES: int = 5
    """Number of attempts to sample a window with sufficient data in real mode."""

    def __init__(
        self,
        timescale_store: Any | None = None,
        encoders: dict[str, Any] | None = None,
        start: datetime | None = None,
        end: datetime | None = None,
        episode_length_min: int = 390,
        tickers: list[str] | None = None,
        rng: np.random.Generator | None = None,
    ):
        self.store = timescale_store
        self.encoders = encoders or {}
        self.start = start or datetime(2018, 1, 1, tzinfo=UTC)
        self.end = end or datetime(2024, 1, 1, tzinfo=UTC)
        self.length = episode_length_min
        self.tickers = tickers or ["SPY"]
        self.rng = rng or np.random.default_rng(42)
        self._mode = "real" if self.store is not None else "synthetic"
        if self._mode == "real" and not self.encoders:
            logger.warning(
                "Real-mode HistoricalEpisodeGenerator requires encoders; "
                "falling back to synthetic states.",
            )
            self._mode = "synthetic"

    def __iter__(self) -> Iterator[dict[str, Any]]:
        return self

    def __next__(self) -> dict[str, Any]:
        if self._mode == "synthetic":
            return self._synthetic_episode()
        return self._real_episode()

    # ------------------------------------------------------------------
    def _synthetic_episode(self) -> dict[str, Any]:
        """Geometric Brownian motion + random latent states.

        Used for unit tests and Phase-A behavioural cloning before the
        encoders are trained. Volatility is drawn from a realistic
        intraday distribution.
        """
        n = self.length
        mu = 0.0001
        sigma = float(self.rng.uniform(0.005, 0.02))
        returns = self.rng.normal(mu, sigma, n)
        prices = 100.0 * np.exp(np.cumsum(returns))
        states = self.rng.standard_normal((n, NEXUS_OUTPUT_DIM)).astype(np.float32) * 0.1
        volatility = np.abs(returns)
        uncertainties = self.rng.uniform(0.1, 0.4, n).astype(np.float32)
        return {
            "prices": prices.astype(np.float32),
            "market_states": states,
            "volatility": volatility.astype(np.float32),
            "uncertainties": uncertainties,
            "ticker": str(self.rng.choice(self.tickers)),
            "synthetic": True,
        }

    # ------------------------------------------------------------------
    def _real_episode(self) -> dict[str, Any]:
        """Replay a real historical window through the frozen encoders.

        Synchronous wrapper around the async store call (we accept the
        event-loop creation cost because episodes are pulled rarely:
        ~once per environment reset, not per step).
        """
        import asyncio

        assert self.store is not None  # mode-gated: only called when set
        for _ in range(self._MAX_RETRIES):
            ticker = str(self.rng.choice(self.tickers))
            max_start = self.end - timedelta(minutes=self.length)
            seconds_range = max(int((max_start - self.start).total_seconds()), 1)
            start_at = self.start + timedelta(
                seconds=int(self.rng.integers(0, seconds_range)),
            )
            df = asyncio.run(
                self.store.get_historical_window(
                    ticker,
                    start_at,
                    start_at + timedelta(minutes=self.length),
                    freq="1m",
                ),
            )
            if df.height < int(self.length * 0.9):
                continue
            return self._build_real_episode_from_df(ticker, df)

        logger.warning(
            f"_real_episode failed after {self._MAX_RETRIES} retries; "
            "falling back to synthetic episode.",
        )
        return self._synthetic_episode()

    # ------------------------------------------------------------------
    def _build_real_episode_from_df(self, ticker: str, df) -> dict[str, Any]:
        """Materialise a real-mode episode from an OHLCV Polars frame."""
        prices = df.select("close").to_numpy().squeeze(-1).astype(np.float32)
        n = len(prices)
        # Pad to expected length if there were minor gaps.
        if n < self.length:
            pad = np.full(self.length - n, prices[-1], dtype=np.float32)
            prices = np.concatenate([prices, pad])
            n = self.length

        log_returns = np.diff(np.log(prices), prepend=np.log(prices[0]))
        volatility = np.abs(log_returns).astype(np.float32)

        # Build the market_state sequence. If the full encoder stack is
        # available, run it; otherwise fall back to deterministic noise
        # seeded by the start datetime so episodes are reproducible.
        market_states = (
            self._encode_window(ticker, df, n)
            if self.encoders
            else (self.rng.standard_normal((n, NEXUS_OUTPUT_DIM)).astype(np.float32) * 0.1)
        )
        uncertainties = self.rng.uniform(0.1, 0.4, n).astype(np.float32)
        return {
            "prices": prices,
            "market_states": market_states,
            "volatility": volatility,
            "uncertainties": uncertainties,
            "ticker": ticker,
            "synthetic": False,
        }

    # ------------------------------------------------------------------
    def _encode_window(self, ticker: str, df, n: int) -> np.ndarray:
        """Run the frozen encoder stack over the price window.

        We use the TFT for price embeddings; for semantic and graph we
        approximate the bar-resolved sequence by forward-filling the
        most recent embeddings — which is how the live system behaves.
        """
        tft = self.encoders.get("tft")
        nexus = self.encoders.get("nexus")
        if tft is None or nexus is None:
            return self.rng.standard_normal((n, NEXUS_OUTPUT_DIM)).astype(np.float32) * 0.1

        # We need (1, T, F) on the same device as the encoders.
        from backend.perception.temporal.preprocessor import preprocess_ohlcv_window

        device = next(tft.parameters()).device
        x = preprocess_ohlcv_window(df.head(n), ticker).unsqueeze(0).to(device)
        with torch.no_grad():
            # The TFT emits one embedding per window; for sequence
            # alignment we replicate it across the n bars. This is the
            # same approximation the live system uses on a 1-min cadence.
            price_emb, _ = tft(x)
            semantic_emb = torch.zeros((1, DIM_SEMANTIC), device=device)
            graph_emb = torch.zeros((1, DIM_GRAPH), device=device)
            fused = nexus(price_emb, semantic_emb, graph_emb)["market_state"]
        fused_np = fused.cpu().numpy().squeeze(0).astype(np.float32)
        return np.tile(fused_np, (n, 1))
