# backend/cognition/agent/uncertainty_gate.py
"""The Uncertainty Gate — Lumina V3's "Amygdala".

This is the safety override system described in section 8 of
Lumina_V3_Deep_Fusion_Architecture.md.

Mathematical formulation
------------------------
For each decision step we collect ``N`` Monte-Carlo Dropout samples from
the policy. Each sample is a 4-D action vector

    a^{(i)} = [direction^{(i)}, urgency^{(i)}, sizing^{(i)}, stop^{(i)}]

The per-dimension epistemic uncertainty is the sample std-dev:

    σ_d = std_i( a_d^{(i)} )       for d ∈ {0..3}

We aggregate the four std-devs into a single scalar by taking the *mean*
(not the max, because a single noisy dimension should not dominate):

    u_t = (1/4) Σ_d σ_d

We then apply a rolling-mean smoother of length W = 10:

    ū_t = (1/W) Σ_{k=t-W+1}^{t} u_k

The gate uses **hysteresis** on ū_t:
* if not currently vetoing AND ū_t > τ_high → engage veto
* if currently  vetoing  AND ū_t < τ_low  → release veto

The hysteresis band ``τ_high − τ_low`` prevents oscillation around the
threshold (a phenomenon equivalent to chattering in sliding-mode control
theory). Default values match the architecture spec: τ_high = 0.85,
τ_low = 0.50, W = 10.

Bayesian justification
----------------------
Per Gal & Ghahramani (2016, "Dropout as a Bayesian Approximation"), running
the network with dropout enabled at inference time approximates sampling
from the variational posterior over network weights. The variance across
samples therefore approximates the *epistemic* uncertainty — the model's
ignorance — as opposed to *aleatoric* uncertainty (irreducible noise),
which is captured by the network's predicted log-std.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
from loguru import logger
from prometheus_client import Counter, Gauge

from backend.config.constants import (
    MC_DROPOUT_SAMPLES,
    UNCERTAINTY_CRITICAL_THRESHOLD,
)

# Prometheus instrumentation — exported via /api/monitoring/metrics
GATE_VETOES = Counter(
    "uncertainty_gate_vetoes_total",
    "Number of agent actions vetoed by the gate",
    labelnames=("reason",),
)
GATE_ACTIVE = Gauge(
    "uncertainty_gate_active",
    "1 if the gate is currently in 'veto' state, 0 otherwise",
)
GATE_CURRENT_UNCERTAINTY = Gauge(
    "uncertainty_gate_current",
    "Most recent rolling-mean uncertainty value",
)


@dataclass
class UncertaintyGateConfig:
    """Hyper-parameters for the gate. Values match Lumina_V3 §8."""

    threshold_high: float = UNCERTAINTY_CRITICAL_THRESHOLD  # τ_high (engage)
    threshold_low: float = 0.50  # τ_low  (release)
    rolling_window: int = MC_DROPOUT_SAMPLES  # W
    warmup_steps: int = 100
    """Number of initial steps during which the gate never vetoes.

    Reason: at agent-initialisation time the policy has not yet been
    queried, the rolling buffer is empty, and the variance estimate is
    not statistically meaningful. We disable the gate until enough
    samples have accumulated.
    """
    max_consecutive_vetoes: int = 50
    """Operational alarm: more than this many consecutive vetoes likely
    indicates a degraded data feed or distributional shift. The gate
    keeps vetoing but emits a CRITICAL log so the human operator
    intervenes."""


class UncertaintyGate:
    """Stateful gate object — one instance per running agent.

    The gate is *not* thread-safe; it must be called sequentially from the
    agent's decision loop. State persists for the lifetime of the process
    and is intentionally NOT serialised to Redis: the gate must reset to
    a clean state on restart so it cannot inherit a stale veto.
    """

    def __init__(self, config: UncertaintyGateConfig | None = None):
        self.config = config or UncertaintyGateConfig()
        self._history: deque[float] = deque(maxlen=self.config.rolling_window)
        self._step: int = 0
        self._currently_vetoing: bool = False
        self._consecutive_vetoes: int = 0

    # ------------------------------------------------------------------
    @staticmethod
    def aggregate_action_samples(samples: np.ndarray) -> float:
        """Reduce N action-vector samples to a scalar uncertainty value.

        Parameters
        ----------
        samples : (N, action_dim) array of MC-Dropout action samples.

        Returns
        -------
        u : float — mean across dimensions of the per-dim std-dev.
        """
        if samples.ndim != 2:
            raise ValueError(f"Expected (N, action_dim), got shape {samples.shape}")
        if samples.shape[0] < 2:
            return 0.0
        return float(samples.std(axis=0, ddof=1).mean())

    # ------------------------------------------------------------------
    def should_veto(self, uncertainty: float) -> bool:
        """Update internal state and return whether to veto the current action.

        Parameters
        ----------
        uncertainty : float
            Output of ``aggregate_action_samples`` for the current step.
        """
        self._step += 1
        self._history.append(uncertainty)
        GATE_CURRENT_UNCERTAINTY.set(uncertainty)

        if self._step < self.config.warmup_steps or len(self._history) < self.config.rolling_window:
            return False

        rolling = float(np.mean(self._history))

        if self._currently_vetoing:
            if rolling < self.config.threshold_low:
                self._currently_vetoing = False
                self._consecutive_vetoes = 0
                GATE_ACTIVE.set(0)
                logger.info(f"Uncertainty gate RELEASED (rolling u={rolling:.3f})")
        else:
            if rolling > self.config.threshold_high:
                self._currently_vetoing = True
                GATE_ACTIVE.set(1)
                GATE_VETOES.labels(reason="epistemic").inc()
                logger.warning(f"Uncertainty gate ENGAGED (rolling u={rolling:.3f})")

        if self._currently_vetoing:
            self._consecutive_vetoes += 1
            if self._consecutive_vetoes == self.config.max_consecutive_vetoes:
                logger.critical(
                    f"Uncertainty gate has vetoed {self._consecutive_vetoes} consecutive steps. "
                    "This usually indicates a data-feed problem or distributional shift. "
                    "Human review recommended.",
                )
            return True

        return False

    # ------------------------------------------------------------------
    @staticmethod
    def defensive_action(action_dim: int = 4) -> np.ndarray:
        """The fallback action emitted whenever the gate vetoes.

        Per architecture spec §8: when the model is "confused", we move to
        cash. In our 4-D action vocabulary that means
            direction = 0  (flat)
            urgency   = -1 (limit/passive)  — irrelevant if direction=0
            sizing    = -1 (zero risk)
            stop_distance = 0
        We return the simplest representation: all zeros. The execution
        layer interprets ``direction=0, sizing=0`` as "close any open
        position and stand down".
        """
        return np.zeros(action_dim, dtype=np.float32)

    # ------------------------------------------------------------------
    def stats(self) -> dict:
        """Snapshot of current gate state — surfaced via /api/agent/status."""
        return {
            "step": self._step,
            "currently_vetoing": self._currently_vetoing,
            "consecutive_vetoes": self._consecutive_vetoes,
            "rolling_uncertainty": (float(np.mean(self._history)) if self._history else 0.0),
            "threshold_high": self.config.threshold_high,
            "threshold_low": self.config.threshold_low,
        }
