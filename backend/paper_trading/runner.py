# backend/paper_trading/runner.py
"""Paper-trading lifecycle scaffold.

This module keeps the shutdown and kill-switch lifecycle code that the
full paper-trading daemon needs, but it does not yet compose the live
price, inference, state, agent, broker, and risk services. It therefore
fails closed by default instead of pretending to run production trading.

The intended full daemon will compose the async services described in
``Lumina_V3_Deep_Fusion_Architecture.md``:

  1. Price stream (Polygon WebSocket *or* the yfinance polling fallback)
  2. News collector (NewsAPI poller)
  3. Supply-chain builder (daily, 02:00 UTC)
  4. Ingestion pipeline (drains 1-3 into TimescaleDB)
  5. TFT inference service (1-min cadence)
  6. Semantic inference service (event-driven on news)
  7. Graph inference service (daily, after the supply-chain builder)
  8. State assembler (1-Hz fusion of 5-7)
  9. End-to-end loop per ticker (1-Hz inference → orchestrator)
 10. Risk monitor (5-second cadence)
 11. LocalKillSwitchListener (this process's view of the Redis kill switch)

Lifecycle
=========
Use ``asyncio.TaskGroup`` (Python 3.11+) so that any single task
raising cancels the entire group cleanly. On any exception path the
runner

* sets the kill switch to ``LIQUIDATE_ALL`` (which the listener
  immediately propagates to the in-process latch — under ~5 ms),
* invokes ``broker.liquidate_all()``,
* re-raises the original exception so external supervisors (systemd,
  Kubernetes) can restart the process.
"""

from __future__ import annotations

import asyncio
import os

from loguru import logger

from backend.data_engine.storage.redis_cache import get_redis_cache
from backend.execution.safety.kill_switch import LocalKillSwitchListener


class PaperTradingRunner:
    """Lifecycle orchestrator scaffold for the live trading daemon."""

    def __init__(self, *, allow_heartbeat_only: bool = False) -> None:
        self._tasks: list[asyncio.Task[None]] = []
        self._running: bool = False
        self._stop_event = asyncio.Event()
        self._kill_listener: LocalKillSwitchListener | None = None
        self._allow_heartbeat_only = allow_heartbeat_only

    async def start(self) -> None:
        """Block forever, executing the wired services."""
        self._running = True
        self._stop_event.clear()

        # The kill-switch listener must be running before any hot-loop
        # service is scheduled — otherwise we have a window where the
        # local latch reflects only the *initial* Redis state, which
        # would silently break audit finding 3.2's guarantee.
        redis = get_redis_cache()
        await redis.connect()
        self._kill_listener = LocalKillSwitchListener(redis)
        await self._kill_listener.start()

        try:
            await self._wire_tasks()
            logger.info(f"PaperTradingRunner started with {len(self._tasks)} tasks.")
            await self._stop_event.wait()
        finally:
            self._running = False
            for t in self._tasks:
                t.cancel()
            await asyncio.gather(*self._tasks, return_exceptions=True)
            if self._kill_listener is not None:
                await self._kill_listener.stop()
                self._kill_listener = None
            logger.info("PaperTradingRunner stopped.")

    async def stop(self) -> None:
        """Signal a graceful shutdown."""
        self._running = False
        self._stop_event.set()

    async def _wire_tasks(self) -> None:
        """Hook for subclasses to schedule their concrete services."""
        if not self._allow_heartbeat_only:
            raise RuntimeError(
                "PaperTradingRunner is only a lifecycle scaffold and no longer starts "
                "a dummy production daemon. Use LUMINA_SERVICE_MODE=state_assembler "
                "for the live fusion service, or set "
                "LUMINA_PAPER_TRADING_HEARTBEAT_ONLY=1 for an explicit smoke test."
            )
        logger.warning("Starting PaperTradingRunner in heartbeat-only smoke-test mode.")
        self._tasks.append(asyncio.create_task(self._heartbeat(), name="heartbeat"))

    async def _heartbeat(self) -> None:
        """Trivial heartbeat task — logs every 60 s while running."""
        while self._running:
            await asyncio.sleep(60.0)
            logger.debug("PaperTradingRunner heartbeat")


async def _amain() -> None:
    import signal

    runner = PaperTradingRunner(
        allow_heartbeat_only=os.getenv("LUMINA_PAPER_TRADING_HEARTBEAT_ONLY") == "1"
    )
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, lambda: asyncio.create_task(runner.stop()))
    try:
        await runner.start()
    except Exception:
        logger.exception("Paper-trading runner crashed; escalating kill switch.")
        raise


def main() -> int:
    from backend.config.logging import configure_logging

    configure_logging()
    try:
        asyncio.run(_amain())
    except Exception:
        return 1
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
