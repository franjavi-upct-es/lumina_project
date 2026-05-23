# backend/execution/safety/rules.py
"""Atomic safety rules. Each is a pure function returning (allow, reason).

Section 8 of Lumina_V3_Deep_Fusion_Architecture.md describes the Risk
Manager's responsibility: even when the model is confident (low epistemic
uncertainty), *physical* constraints (drawdown limits, position caps,
leverage rules) can veto an action. These rules are the implementation.

Design principle
----------------
We compose individual rules instead of writing a giant `if-else`. Each rule
is a unary function of the ``SafetyContext`` so we can:

  * Unit-test rules independently.
  * Reorder, enable, or disable them at runtime.
  * Audit a vetoed action by listing exactly which rule fired.

Each rule's name maps 1:1 to a Prometheus counter label, so dashboards
can show "rule firing rate" over time.

The rules are written defensively against the 4-D action vector. Where a
rule needs to inspect a single component (e.g. ``direction``), it reads
``ctx.proposed_action[0]``. The full vector is passed in case a rule
needs to examine sizing/urgency too.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np


@dataclass
class SafetyContext:
    """All state a rule may need.

    The execution orchestrator builds one of these per decision step.
    """

    proposed_action: np.ndarray  # shape (4,)
    """The agent's raw 4-D continuous action vector."""

    current_position: float
    """Current portfolio fraction in [-1, 1]."""

    equity: float
    peak_equity: float
    uncertainty: float
    """Latest gate-level uncertainty (NOT used by the gate itself; rules
    can apply *additional* uncertainty-based vetoes if needed)."""

    kill_switch_state: str
    """One of {"NORMAL", "CLOSE_ONLY", "LIQUIDATE_ALL"}."""

    consecutive_losses: int = 0
    max_drawdown_pct: float = 0.20
    max_position: float = 1.0
    """Hard cap on |portfolio fraction|."""
    max_size_per_trade: float = 0.5
    """Cap on |action[2]| (sizing) per single trade decision."""
    uncertainty_threshold: float = 0.85


SafetyRule = Callable[[SafetyContext], tuple[bool, str]]


# ----------------------------------------------------------------------
# Rules
# ----------------------------------------------------------------------
def rule_kill_switch(ctx: SafetyContext) -> tuple[bool, str]:
    """Veto if the kill switch is armed.

    LIQUIDATE_ALL → veto everything.
    CLOSE_ONLY    → veto unless the proposed action *reduces* the position.
    """
    if ctx.kill_switch_state == "LIQUIDATE_ALL":
        return False, "Kill switch armed: LIQUIDATE_ALL"
    if ctx.kill_switch_state == "CLOSE_ONLY":
        # Allow only if |new position| < |current position|
        new_dir = float(ctx.proposed_action[0])
        if abs(new_dir) > abs(ctx.current_position):
            return False, "Kill switch CLOSE_ONLY: position increases not allowed"
    return True, ""


def rule_max_drawdown(ctx: SafetyContext) -> tuple[bool, str]:
    """Veto when drawdown from peak exceeds the configured limit."""
    if ctx.peak_equity <= 0:
        return True, ""
    dd = 1.0 - ctx.equity / ctx.peak_equity
    if dd > ctx.max_drawdown_pct:
        return False, f"Drawdown {dd:.2%} > limit {ctx.max_drawdown_pct:.2%}"
    return True, ""


def rule_uncertainty(ctx: SafetyContext) -> tuple[bool, str]:
    """Belt-and-suspenders uncertainty veto on top of the gate."""
    if ctx.uncertainty > ctx.uncertainty_threshold:
        return False, f"Uncertainty {ctx.uncertainty:.3f} > {ctx.uncertainty_threshold}"
    return True, ""


def rule_position_limit(ctx: SafetyContext) -> tuple[bool, str]:
    """Veto if |direction × sizing| would exceed the position cap."""
    direction = float(ctx.proposed_action[0])
    size_factor = (float(ctx.proposed_action[2]) + 1.0) * 0.5  # [0, 1]
    target = direction * size_factor
    if abs(target) > ctx.max_position + 1e-6:
        return False, f"Target |{target:.3f}| > max {ctx.max_position}"
    return True, ""


def rule_no_flip_high_uncertainty(ctx: SafetyContext) -> tuple[bool, str]:
    """Reject mid-uncertainty position flips (long ↔ short).

    A flip from +0.5 to -0.5 doubles transaction cost and is risky
    when the model is not fully confident. We block it above 0.7.
    """
    if ctx.uncertainty <= 0.7:
        return True, ""
    new_dir = float(ctx.proposed_action[0])
    if ctx.current_position * new_dir < 0:
        return False, "Flip blocked at uncertainty > 0.7"
    return True, ""


def rule_loss_streak(ctx: SafetyContext) -> tuple[bool, str]:
    """Cool-off after 5 consecutive losing trades."""
    if ctx.consecutive_losses >= 5:
        return False, f"Loss streak = {ctx.consecutive_losses}; cooling off"
    return True, ""


def rule_size_cap(ctx: SafetyContext) -> tuple[bool, str]:
    """Per-trade size cap on action[2]."""
    size_raw = float(ctx.proposed_action[2])
    size_factor = (size_raw + 1.0) * 0.5
    if size_factor > ctx.max_size_per_trade + 1e-6:
        return False, f"Size factor {size_factor:.3f} > {ctx.max_size_per_trade}"
    return True, ""


# ----------------------------------------------------------------------
# Default ordered list of rules
# ----------------------------------------------------------------------
ALL_RULES: list[SafetyRule] = [
    rule_kill_switch,
    rule_max_drawdown,
    rule_uncertainty,
    rule_position_limit,
    rule_size_cap,
    rule_no_flip_high_uncertainty,
    rule_loss_streak,
]
