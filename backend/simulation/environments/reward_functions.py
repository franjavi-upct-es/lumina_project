# backend/simulation/environments/reward_functions.py
"""Reward-shaping functions for the trading environment.

The ``LuminaTradingEnv.step`` method has a hard-coded basic reward; this
module provides plug-in alternatives used during the *Spartan Phase C*
(Sharpe-ratio fine-tuning).

All functions take the same signature:

    fn(pnl, equity, vol, position, recent_returns) -> float

so the environment can swap them without code changes.

Three reward families
---------------------
1. ``pnl_reward``      — pure relative PnL. Maximises return regardless
                         of volatility. Useful for Phase A (BC).
2. ``sharpe_reward``   — incremental Sharpe ratio (Moody & Saffell, 2001):
                            Δ Sharpe_t ≈ (r_t − ŕ̂_t) / σ̂_t
                         where ŕ̂_t and σ̂_t are EWMA estimators.
3. ``sortino_reward``  — same idea but only penalises *downside* deviation
                         (Sortino, 1994).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def pnl_reward(pnl: float, equity: float, **_) -> float:
    """Simple normalised PnL. Used during Phase A behavioural cloning."""
    if equity <= 0:
        return 0.0
    return 100.0 * pnl / equity


@dataclass
class _EWMAState:
    mean: float = 0.0
    var: float = 0.0
    n: int = 0


class IncrementalSharpe:
    """Online Sharpe ratio with EWMA mean and variance.

    EWMA mean update :     μ_t = (1−η) μ_{t−1} + η r_t
    EWMA variance update:  v_t = (1−η) (v_{t−1} + η (r_t − μ_{t−1})²)
    Sharpe approximation:  S_t ≈ μ_t / sqrt(v_t + ε)
    """

    def __init__(self, eta: float = 0.05, eps: float = 1e-6):
        self.eta = eta
        self.eps = eps
        self.state = _EWMAState()

    def update(self, r: float) -> float:
        s = self.state
        s.n += 1
        if s.n == 1:
            s.mean = r
            s.var = 0.0
            return 0.0
        prev_mean = s.mean
        s.mean = (1.0 - self.eta) * s.mean + self.eta * r
        s.var = (1.0 - self.eta) * (s.var + self.eta * (r - prev_mean) ** 2)
        return s.mean / np.sqrt(s.var + self.eps)


class IncrementalSortino:
    """Online Sortino ratio: like Sharpe but downside-only variance."""

    def __init__(self, eta: float = 0.05, eps: float = 1e-6):
        self.eta = eta
        self.eps = eps
        self._mean = 0.0
        self._down_var = 0.0
        self._n = 0

    def update(self, r: float) -> float:
        self._n += 1
        if self._n == 1:
            self._mean = r
            return 0.0
        prev_mean = self._mean
        self._mean = (1.0 - self.eta) * self._mean + self.eta * r
        if r < prev_mean:
            self._down_var = (1.0 - self.eta) * (self._down_var + self.eta * (r - prev_mean) ** 2)
        return self._mean / np.sqrt(self._down_var + self.eps)
