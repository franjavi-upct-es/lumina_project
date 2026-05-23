# backend/execution/safety/risk_monitor.py
"""Continuous risk monitor: escalates kill switch when thresholds are breached."""

from __future__ import annotations

import asyncio

from loguru import logger

from backend.execution.broker.base import BaseBroker
from backend.execution.safety.kill_switch import KillSwitch, KillSwitchState


class RiskMonitor:
    def __init__(
        self,
        broker: BaseBroker,
        kill_switch: KillSwitch,
        max_drawdown_soft: float = 0.15,
        max_drawdown_hard: float = 0.20,
        check_interval_s: float = 5.0,
    ):
        self.broker = broker
        self.ks = kill_switch
        self.soft = max_drawdown_soft
        self.hard = max_drawdown_hard
        self.interval = check_interval_s
        self._peak_equity: float | None = None
        self._running = False

    async def run(self) -> None:
        self._running = True
        while self._running:
            try:
                await self._check_once()
            except Exception as exc:
                logger.error(f"RiskMonitor error: {exc}")
            await asyncio.sleep(self.interval)

    async def stop(self) -> None:
        self._running = False

    async def _check_once(self) -> None:
        acct = await self.broker.get_account()
        if self._peak_equity is None or acct.equity > self._peak_equity:
            self._peak_equity = acct.equity
            return
        dd = 1.0 - (acct.equity / self._peak_equity)
        state = await self.ks.get_state()
        if dd > self.hard and state != KillSwitchState.LIQUIDATE_ALL:
            await self.ks.set_state(
                KillSwitchState.LIQUIDATE_ALL,
                f"Hard drawdown breach: {dd:.2%}",
            )
        elif dd > self.soft and state == KillSwitchState.NORMAL:
            await self.ks.set_state(
                KillSwitchState.CLOSE_ONLY,
                f"Soft drawdown breach: {dd:.2%}",
            )
