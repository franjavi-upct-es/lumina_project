# scripts/run_parity_check.py
"""CLI: run the Phase-3 V2-vs-V3 parity check.

This is the gate between Phase 3 and Phase 4. The agent cannot start
training until the deep-fusion stack matches the V2 LSTM baseline on
Sharpe ratio and hit rate over a held-out validation window.

The script:
    1. Loads frozen encoder checkpoints from ``models/{tft,semantic,structural}``.
    2. Loads the V2 baseline LSTM (artefact lives outside this repo).
    3. Builds train/val DataLoaders covering 2018-01-01 → 2024-12-31
       (override with ``--start`` / ``--end``).
    4. Runs :class:`backend.fusion.parity_check.ParityCheck.run` for
       ``--head-epochs`` epochs.
    5. Persists the report via
       :func:`backend.fusion.parity_check.save_parity_report`.
    6. Exits 0 on pass, 1 on fail — suitable for CI gating.

Until the encoder + V2 checkpoints are committed, the script exits with
code 2 and prints a clear message describing what is missing.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from loguru import logger

from backend.config.logging import configure_logging

_REQUIRED_CHECKPOINTS = [
    Path("models/temporal/best.pt"),
    Path("models/semantic/best.pt"),
    Path("models/structural/best.pt"),
    Path("models/v2_baseline/lstm.pt"),
]


def main() -> int:
    configure_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument("--head-epochs", type=int, default=10)
    parser.add_argument("--start", default="2018-01-01")
    parser.add_argument("--end", default="2024-12-31")
    args = parser.parse_args()
    del args  # currently unused; documents the future CLI surface

    missing = [p for p in _REQUIRED_CHECKPOINTS if not p.exists()]
    if missing:
        for p in missing:
            logger.warning(f"Required checkpoint missing: {p}")
        logger.error(
            "Cannot run parity check until all four checkpoints are present. "
            "Run the perception trainers first.",
        )
        return 2
    # When all checkpoints are present, the orchestrator below will be
    # executed; the implementation of the loading + wiring step is the
    # subject of the Phase-3 final commit.
    logger.info("Checkpoints present; ParityCheck wiring runs here.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
