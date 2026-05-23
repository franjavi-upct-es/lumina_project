# backend/execution/orchestrator.py
"""Execution Orchestrator — single call path from agent action to broker order.

The orchestrator is the *only* component allowed to talk to the broker.
It owns three responsibilities:

1. Build a ``SafetyContext`` and pass it to the ``SafetyArbitrator``.
2. Translate the agent's 4-D action vector into a (ticker, qty, side)
   broker order. The four action components are interpreted as follows
   (see ``backend.simulation.environments.base_env`` for the canonical
   decoder):

       direction     = action[0]   (target portfolio fraction)
       urgency       = action[1]   (limit ↔ market order)
       sizing        = action[2]   (Kelly-fraction multiplier)
       stop_distance = action[3]   (ATR multiplier; not handled here —
                                    the broker layer manages stops)

3. Compose the final action by combining ``direction × size_factor``
   and applying the arbitrator's veto if it fired. The arbitrator's
   ``final_action`` over-rides the agent's request.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass

import numpy as np
from loguru import logger
from prometheus_client import Counter, Histogram

from backend.execution.broker.base import BaseBroker, Order, OrderSide
from backend.execution.safety.arbitrator import SafetyArbitrator, SafetyDecision
from backend.execution.safety.kill_switch import KillSwitch, KillSwitchState
from backend.execution.safety.rules import SafetyContext

ORDERS_SUBMITTED = Counter(
    "execution_orders_submitted_total",
    "Orders sent to broker",
    labelnames=("side",),
)
ORDERS_REJECTED = Counter(
    "execution_orders_rejected_total",
    "Orders blocked by arbitrator",
)
EXECUTION_LATENCY = Histogram(
    "execution_latency_seconds",
    "Action → order acknowledgement",
    buckets=(0.001, 0.005, 0.01, 0.05, 0.1, 0.5),
)


@dataclass
class ExecutionResult:
    """What ``execute()`` returns. ``final_action`` is the *post-veto*
    target portfolio fraction (a scalar in [-1, 1])."""

    decision: SafetyDecision
    orders: list[Order]
    final_action: float


class ExecutionOrchestrator:
    """Glue between the cognitive layer and the broker."""

    def __init__(
        self,
        broker: BaseBroker,
        arbitrator: SafetyArbitrator,
        kill_switch: KillSwitch,
    ):
        self.broker = broker
        self.arb = arbitrator
        self.ks = kill_switch

    # ------------------------------------------------------------------
    @staticmethod
    def _action_to_target(action: np.ndarray) -> float:
        """Decode the 4-D action vector into the target portfolio fraction.

        Mirrors ``LuminaTradingEnv._decode_action`` so the live behaviour
        matches the training environment exactly.
        """
        direction = float(np.clip(action[0], -1.0, 1.0))
        size_raw = float(np.clip(action[2], -1.0, 1.0))
        size_factor = (size_raw + 1.0) * 0.5  # [-1, 1] → [0, 1]
        return direction * size_factor

    # ------------------------------------------------------------------
    async def execute(
        self,
        ticker: str,
        proposed_action: np.ndarray,
        uncertainty: float,
        consecutive_losses: int = 0,
    ) -> ExecutionResult:
        """Run one safety-arbitrated execution cycle for ``ticker``."""
        action_arr = np.asarray(proposed_action, dtype=np.float32)

        with EXECUTION_LATENCY.time():
            acct = await self.broker.get_account()
            positions = acct.positions
            pos_qty = positions[ticker].qty if ticker in positions else 0.0
            current_position = pos_qty / (acct.equity / 100.0) if acct.equity > 0 else 0.0
            ks_state = await self.ks.get_state()

            # Hard fast-path: kill switch armed → liquidate everything.
            if ks_state == KillSwitchState.LIQUIDATE_ALL:
                orders = await self.broker.liquidate_all()
                return ExecutionResult(
                    decision=SafetyDecision(
                        approved=False,
                        final_action=0.0,
                        vetoes=["kill_switch_liquidate"],
                        reason="Kill switch armed: LIQUIDATE_ALL",
                    ),
                    orders=orders,
                    final_action=0.0,
                )

            # Build the safety context from the *raw* 4-D action so the
            # rules can inspect direction / sizing independently.
            ctx = SafetyContext(
                proposed_action=action_arr,
                current_position=current_position,
                equity=acct.equity,
                peak_equity=acct.equity,
                uncertainty=uncertainty,
                kill_switch_state=ks_state.value,
                consecutive_losses=consecutive_losses,
            )
            decision = self.arb.evaluate(ctx)

            # If vetoed but the resulting target is identical to the
            # current position (i.e. "hold"), no order is generated.
            target_position = (
                self._action_to_target(action_arr)
                if decision.approved
                else float(decision.final_action)
            )
            if abs(target_position - current_position) < 1e-4:
                if not decision.approved:
                    ORDERS_REJECTED.inc()
                return ExecutionResult(
                    decision=decision,
                    orders=[],
                    final_action=current_position,
                )

            # Translate the position delta into a broker order.
            delta = target_position - current_position
            side: OrderSide = "buy" if delta > 0 else "sell"
            qty = abs(delta) * (acct.equity / 100.0)
            order = await self.broker.submit_order(
                ticker=ticker,
                qty=qty,
                side=side,
                client_order_id=f"lum_{uuid.uuid4().hex[:12]}",
            )
            ORDERS_SUBMITTED.labels(side=side).inc()
            logger.info(
                f"Executed {side} {qty:.2f} {ticker} "
                f"(target={target_position:.3f}, vetoed={not decision.approved})",
            )
            return ExecutionResult(
                decision=decision,
                orders=[order],
                final_action=target_position,
            )
