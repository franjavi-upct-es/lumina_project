# backend/execution/safety/arbitrator.py
"""SafetyArbitrator — composes the atomic rules in ``rules.py`` and produces
a single approve/reject decision plus the *post-veto* target action.

Two policies for handling vetoes
--------------------------------
1. ``force_close_on_veto = True``  (default, conservative)
        If any rule fires AND the agent currently holds a position,
        the post-veto target is **0** (close the position). If the
        agent is flat, the target is unchanged from current_position
        (i.e. "do nothing").

2. ``force_close_on_veto = False`` (preserve-position mode)
        Same logic but the target is left at ``current_position`` —
        useful during DR training where we want the model to keep
        learning the consequences of holding through volatility.

In all cases ``ARBITRATOR_VETOES`` Prometheus counter is incremented
for every fired rule, providing per-rule visibility on the dashboard.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from loguru import logger
from prometheus_client import Counter

from backend.execution.safety.rules import ALL_RULES, SafetyContext, SafetyRule

ARBITRATOR_VETOES = Counter(
    "arbitrator_vetoes_total",
    "Number of times each safety rule has fired",
    labelnames=("rule",),
)
ARBITRATOR_APPROVALS = Counter(
    "arbitrator_approvals_total",
    "Actions approved by the arbitrator",
)


@dataclass
class SafetyDecision:
    """Outcome of a single arbitrator evaluation."""

    approved: bool
    final_action: float
    """Post-veto target portfolio fraction in [-1, 1].
    Note: a scalar, NOT the full 4-D vector. The orchestrator turns this
    into a (qty, side) broker order."""
    vetoes: list[str] = field(default_factory=list)
    reason: str = ""


@dataclass
class SafetyConfig:
    rules: list[SafetyRule] = field(default_factory=lambda: list(ALL_RULES))
    force_close_on_veto: bool = True


class SafetyArbitrator:
    def __init__(self, config: SafetyConfig | None = None):
        self.config = config or SafetyConfig()

    # ------------------------------------------------------------------
    @staticmethod
    def _action_to_target(action) -> float:
        """Convert a 4-D action vector into the target portfolio fraction."""
        import numpy as np

        a = np.asarray(action, dtype=float)
        direction = float(np.clip(a[0], -1.0, 1.0))
        size_factor = (float(np.clip(a[2], -1.0, 1.0)) + 1.0) * 0.5
        return direction * size_factor

    # ------------------------------------------------------------------
    def evaluate(self, ctx: SafetyContext) -> SafetyDecision:
        vetoes: list[str] = []
        for rule in self.config.rules:
            allow, reason = rule(ctx)
            if not allow:
                vetoes.append(f"{rule.__name__}: {reason}")
                ARBITRATOR_VETOES.labels(rule=rule.__name__).inc()

        if not vetoes:
            ARBITRATOR_APPROVALS.inc()
            return SafetyDecision(
                approved=True,
                final_action=self._action_to_target(ctx.proposed_action),
            )

        # A rule fired → choose the post-veto target.
        if self.config.force_close_on_veto and abs(ctx.current_position) > 1e-6:
            final = 0.0
            logger.warning(f"Arbitrator VETO → force close. Vetoes: {vetoes}")
        else:
            final = ctx.current_position
            logger.warning(f"Arbitrator VETO → hold. Vetoes: {vetoes}")

        return SafetyDecision(
            approved=False,
            final_action=final,
            vetoes=vetoes,
            reason="; ".join(vetoes),
        )
