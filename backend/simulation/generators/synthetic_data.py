# backend/simulation/generators/synthetic_data.py
"""Synthetic price generators based on stochastic process models.

Two processes are implemented, both standard in quantitative finance:

1. **Geometric Brownian Motion (GBM)** — Black-Scholes-like:
       dS_t = μ S_t dt + σ S_t dW_t
   In discrete form:
       S_{t+1} = S_t · exp((μ − σ²/2) Δt + σ √Δt · ε_t),  ε_t ~ N(0, 1)
   Pure GBM lacks fat tails; it is fine for sanity checks but the agent
   trained on GBM alone will be over-confident in calm regimes.

2. **Merton Jump-Diffusion** — GBM + compound Poisson jumps:
       dS_t = μ S_t dt + σ S_t dW_t + S_{t−} (J − 1) dN_t
   where N_t ~ Poisson(λ t) and log J ~ N(μ_J, σ_J²). This adds
   discontinuities and gives realistic kurtosis. Use it as the
   *minimum* synthetic process before letting the agent see real data.

Both functions return arrays compatible with ``LuminaTradingEnv``
episodes (see ``HistoricalEpisodeGenerator`` for the schema).
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from backend.config.constants import NEXUS_OUTPUT_DIM


def gbm_episode(
    n_steps: int,
    mu: float = 5e-5,  # ≈ 5% annualised at 1-min cadence
    sigma: float = 0.012,  # ≈ realistic intraday vol
    s0: float = 100.0,
    rng: np.random.Generator | None = None,
) -> dict:
    """Generate a single Geometric Brownian Motion episode."""
    rng = rng or np.random.default_rng()
    dt = 1.0
    eps = rng.standard_normal(n_steps)
    log_returns = (mu - 0.5 * sigma * sigma) * dt + sigma * np.sqrt(dt) * eps
    prices = s0 * np.exp(np.cumsum(log_returns))
    return _build_episode(prices, log_returns, rng)


def jump_diffusion_episode(
    n_steps: int,
    mu: float = 5e-5,
    sigma: float = 0.012,
    lambda_jump: float = 1e-3,  # one jump every ~1000 bars
    mu_jump: float = -0.02,
    sigma_jump: float = 0.03,
    s0: float = 100.0,
    rng: np.random.Generator | None = None,
) -> dict:
    """Merton jump-diffusion episode with negative-mean jumps."""
    rng = rng or np.random.default_rng()
    dt = 1.0
    eps = rng.standard_normal(n_steps)
    diffusion = (mu - 0.5 * sigma * sigma) * dt + sigma * np.sqrt(dt) * eps
    n_jumps = rng.poisson(lambda_jump * n_steps * dt)
    jump_total = np.zeros(n_steps)
    for _ in range(n_jumps):
        idx = rng.integers(n_steps)
        jump_total[idx] += rng.normal(mu_jump, sigma_jump)
    log_returns = diffusion + jump_total
    prices = s0 * np.exp(np.cumsum(log_returns))
    return _build_episode(prices, log_returns, rng)


def _build_episode(prices: np.ndarray, log_returns: np.ndarray, rng: np.random.Generator) -> dict:
    n = len(prices)
    return {
        "prices": prices.astype(np.float32),
        "market_states": rng.standard_normal((n, NEXUS_OUTPUT_DIM)).astype(np.float32) * 0.1,
        "volatility": np.abs(log_returns).astype(np.float32),
        "uncertainties": rng.uniform(0.1, 0.4, n).astype(np.float32),
        "synthetic": True,
    }


class SyntheticEpisodeGenerator:
    """Iterator wrapper around the two functions above."""

    def __init__(
        self,
        n_steps: int = 390,
        process: str = "jump_diffusion",
        rng: np.random.Generator | None = None,
        **kwargs: float,
    ):
        self.n_steps = n_steps
        self.kwargs = kwargs
        self.rng = rng or np.random.default_rng()
        self._fn: Callable[..., dict]
        if process == "gbm":
            self._fn = gbm_episode
        elif process == "jump_diffusion":
            self._fn = jump_diffusion_episode
        else:
            raise ValueError(f"Unknown process: {process}")

    def __iter__(self):
        return self

    def __next__(self) -> dict:
        return self._fn(self.n_steps, rng=self.rng, **self.kwargs)
