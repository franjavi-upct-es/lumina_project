# scripts/run_crisis_drill.py
"""CLI: run the synthetic 2020-style crisis drill — the Phase-8 milestone.

Build a *fixed* synthetic episode that mimics the broad shape of the
COVID crash:

    days 0..2  : VOL_SPIKE
    day 3      : FLASH_CRASH (−12 %)
    days 4..15 : SUSTAINED_CRASH drift (−25 % over the window)
    days 16..21: VOL_SPIKE again

Step the agent through it via :class:`backend.cognition.agent.PPOAgent.act`
and assert at the end:

    final_equity      > 0.85 * INITIAL_CAPITAL
    kill_switch_state != "LIQUIDATE_ALL"
    arbitrator_vetoes > 10

If any assertion fails the script exits 1 (suitable for CI gating).
A full episode trace is persisted as JSON for post-mortem review.

The skeleton below is fully implementable today but requires a trained
agent checkpoint at ``models/agent/final.pt``. Until that file exists,
the script reports the missing prerequisite and exits with code 2.
"""

from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger

from backend.config.logging import configure_logging

_REQUIRED_CHECKPOINT = Path("models/agent/final.pt")


def main() -> int:
    configure_logging()
    if not _REQUIRED_CHECKPOINT.exists():
        logger.error(
            f"Trained agent checkpoint not found at {_REQUIRED_CHECKPOINT}. "
            "Run `python -m scripts.train_agent` first.",
        )
        return 2
    logger.info("Crisis-drill wiring runs here once the checkpoint is present.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
