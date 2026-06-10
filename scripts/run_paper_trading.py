# scripts/run_paper_trading.py
"""CLI: launch the paper-trading lifecycle scaffold.

This script instantiates :class:`backend.paper_trading.runner.PaperTradingRunner`,
registers signal handlers so SIGTERM / SIGINT trigger a graceful shutdown, and
blocks until the runner exits.

The runner is a scaffold, not a production daemon. It fails closed unless
``LUMINA_PAPER_TRADING_HEARTBEAT_ONLY=1`` is set for an explicit smoke test.

Exit codes
----------
0 — clean shutdown after SIGTERM/SIGINT
1 — unhandled exception from the runner (after the kill switch has
    been escalated to ``LIQUIDATE_ALL``)
"""

from __future__ import annotations

import asyncio
import os
import signal
import sys

from loguru import logger

from backend.config.logging import configure_logging
from backend.paper_trading.runner import PaperTradingRunner


async def _amain() -> None:
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
    configure_logging()
    try:
        asyncio.run(_amain())
    except Exception:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
