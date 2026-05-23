# scripts/run_paper_trading.py
"""CLI: launch the paper-trading daemon.

This is the production entry point for Phase 8. It instantiates a
:class:`backend.paper_trading.runner.PaperTradingRunner`, registers
signal handlers so SIGTERM / SIGINT trigger a graceful shutdown, and
blocks until the runner exits.

The runner itself is responsible for managing the nine async services
that make up the live stack (see its docstring for the full list).

Exit codes
----------
0 — clean shutdown after SIGTERM/SIGINT
1 — unhandled exception from the runner (after the kill switch has
    been escalated to ``LIQUIDATE_ALL``)
"""

from __future__ import annotations

import asyncio
import signal
import sys

from loguru import logger

from backend.config.logging import configure_logging
from backend.paper_trading.runner import PaperTradingRunner


async def _amain() -> None:
    runner = PaperTradingRunner()
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
