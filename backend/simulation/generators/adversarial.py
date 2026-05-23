# backend/simulation/generators/adversarial.py
"""Adversarial episode generator — Phase B "Domain Randomization".

This module implements section 7B of the architecture spec:
"The Spartan Forge". Episodes are *warped* versions of the underlying
historical / synthetic episodes, with one of six perturbation types
applied. The agent is forced to learn that "stability is a privilege,
not a right".

Six warp types
--------------
1. ``FLASH_CRASH``       — instantaneous −5..−15% gap during 5 bars,
                           volatility ×5 in the affected window.
2. ``SUSTAINED_CRASH``   — geometric drift down −20% over the second
                           half of the episode (≈ 2020 Mar-style).
3. ``VOL_SPIKE``         — 20-bar window with realised vol ×10
                           and shocks injected on top of the trend.
4. ``CORRELATION_BREAK`` — Gaussian noise (σ=0.5) added to the latent
                           market_states, mimicking a regime where
                           previously useful features lose predictive
                           power.
5. ``SEMANTIC_PANIC``    — additive negative drift on the *Semantic*
                           portion of the latent state (channels
                           [DIM_PRICE : DIM_PRICE + DIM_SEMANTIC]),
                           paired with a spike in epistemic uncertainty.
6. ``SILENT_DRIFT``      — slow log-linear drift of −0.1% per bar,
                           hidden from volatility — meant to test the
                           agent's resistance to "boiling-frog" decay.

Each warp is picked uniformly at random unless the trainer asks for a
specific type. The wrapped generator is consumed *fully*; we mutate the
returned dict and pass it on.

Note on the SEMANTIC_PANIC channel slice
----------------------------------------
We assume the encoder concatenation ordering: [Price | Semantic | Graph].
``DIM_PRICE`` and ``DIM_SEMANTIC`` from ``backend.config.constants`` give
the slice indices. If the Nexus changes its output mapping, update this
file accordingly.
"""

from __future__ import annotations

from enum import StrEnum

import numpy as np

from backend.config.constants import DIM_PRICE, DIM_SEMANTIC


class WarpType(StrEnum):
    """Enumeration of all available adversarial perturbations."""

    FLASH_CRASH = "flash_crash"
    SUSTAINED_CRASH = "sustained_crash"
    VOL_SPIKE = "vol_spike"
    CORRELATION_BREAK = "correlation_break"
    SEMANTIC_PANIC = "semantic_panic"
    SILENT_DRIFT = "silent_drift"


class AdversarialGenerator:
    """Wrap a base episode generator and apply a chosen warp.

    Parameters
    ----------
    base_generator : iterator
        Anything that yields episode dicts compatible with
        ``LuminaTradingEnv``.
    rng : numpy.random.Generator | None
        Override for determinism.
    """

    def __init__(self, base_generator, rng: np.random.Generator | None = None):
        self.base = base_generator
        self.rng = rng or np.random.default_rng()

    # ------------------------------------------------------------------
    def __iter__(self):
        return self

    def __next__(self) -> dict:
        return self.random_warp()

    # ------------------------------------------------------------------
    def generate(self, warp: WarpType) -> dict:
        """Pull one episode from ``base`` and apply the requested warp."""
        episode = next(iter(self.base))
        prices = episode["prices"]
        n = len(prices)

        if warp == WarpType.FLASH_CRASH:
            t = int(self.rng.integers(50, max(51, n - 50)))
            crash_size = float(self.rng.uniform(0.05, 0.15))
            window = slice(t, min(t + 5, n))
            episode["prices"] = prices.copy()
            episode["prices"][window] *= 1.0 - crash_size
            episode["volatility"] = episode["volatility"].copy()
            episode["volatility"][window] *= 5.0

        elif warp == WarpType.SUSTAINED_CRASH:
            t = int(self.rng.integers(0, max(1, n // 2)))
            drift = np.linspace(0.0, -0.20, n - t)
            episode["prices"] = prices.copy()
            episode["prices"][t:] *= np.exp(drift)

        elif warp == WarpType.VOL_SPIKE:
            t = int(self.rng.integers(50, max(51, n - 50)))
            window = slice(t, min(t + 20, n))
            episode["volatility"] = episode["volatility"].copy()
            episode["volatility"][window] *= 10.0
            shocks = self.rng.normal(0.0, 0.05, window.stop - window.start)
            episode["prices"] = prices.copy()
            episode["prices"][window] *= np.exp(np.cumsum(shocks))

        elif warp == WarpType.CORRELATION_BREAK:
            episode["market_states"] = (
                episode["market_states"] + self.rng.normal(0.0, 0.5, episode["market_states"].shape)
            ).astype(np.float32)

        elif warp == WarpType.SEMANTIC_PANIC:
            t = int(self.rng.integers(0, max(1, n)))
            length = min(30, n - t)
            sem_slice = slice(DIM_PRICE, DIM_PRICE + DIM_SEMANTIC)
            shock = self.rng.normal(-2.0, 1.0, (length, DIM_SEMANTIC)).astype(np.float32)
            episode["market_states"] = episode["market_states"].copy()
            # Note: this is correct only if NEXUS_OUTPUT_DIM == DIM_FUSED.
            # When the Nexus uses a 256-d head, the SEMANTIC slice does not
            # map cleanly to the latent state. Implementation detail:
            # we still push a directional shock through the first DIM_SEMANTIC
            # channels — the precise channel does not need to be "semantic"
            # for the warp to be useful; it only needs to be a predictable
            # shock pattern the agent must learn to discount.
            channels = min(sem_slice.stop, episode["market_states"].shape[1]) - sem_slice.start
            channels = max(0, channels)
            if channels > 0:
                episode["market_states"][
                    t : t + length, sem_slice.start : sem_slice.start + channels
                ] += shock[:, :channels]
            episode["uncertainties"] = episode["uncertainties"].copy()
            episode["uncertainties"][t : t + length] = np.minimum(
                episode["uncertainties"][t : t + length] + 0.5,
                1.0,
            )

        elif warp == WarpType.SILENT_DRIFT:
            drift_per_bar = -1e-3
            drift = np.linspace(0.0, drift_per_bar * n, n)
            episode["prices"] = prices.copy() * np.exp(np.cumsum(np.full(n, drift_per_bar)))

        episode["warp_type"] = warp.value
        return episode

    # ------------------------------------------------------------------
    def random_warp(self) -> dict:
        """Pick a warp uniformly at random and apply it."""
        warp = WarpType(self.rng.choice([w.value for w in WarpType]))
        return self.generate(warp)
