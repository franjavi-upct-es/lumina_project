# scripts/train_agent.py
"""CLI: run the full Spartan curriculum end-to-end.

Thin wrapper around :func:`backend.cognition.training.trainer.train_full_curriculum`.
Exists so the Docker ``brain`` image can launch training via
``LUMINA_SERVICE_MODE=train_agent`` without the entrypoint needing to
know any Python-level details.

Exit codes
----------
0 — curriculum completed; the final checkpoint is at
    ``models/agent/final.pt`` and ``models/agent/manifest.json``
    records the run metadata.
1 — a curriculum phase failed its acceptance gate.
2 — environment error (e.g. missing checkpoint, missing config).

Usage
-----
    python -m scripts.train_agent
    python -m scripts.train_agent --episodes-dr 200 --episodes-sharpe 500
"""

from __future__ import annotations

import argparse
import sys

from loguru import logger

from backend.cognition.training.trainer import train_full_curriculum
from backend.config.logging import configure_logging


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the full Spartan curriculum (BC → DR → Sharpe).",
    )
    parser.add_argument(
        "--bc-epochs",
        type=int,
        default=20,
        help="Number of epochs for the Behavioural Cloning warm-start.",
    )
    parser.add_argument(
        "--episodes-dr",
        type=int,
        default=500,
        help="Number of episodes for the Domain-Randomisation phase.",
    )
    parser.add_argument(
        "--episodes-sharpe",
        type=int,
        default=1000,
        help="Number of episodes for the Sharpe-optimisation phase.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Force device ('cuda' or 'cpu'). Defaults to auto-detect.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    configure_logging()
    args = _parse_args(argv)
    try:
        final_path = train_full_curriculum(
            bc_epochs=args.bc_epochs,
            episodes_dr=args.episodes_dr,
            episodes_sharpe=args.episodes_sharpe,
            device=args.device,
        )
    except RuntimeError as exc:
        # A curriculum phase failed its gate; this is the documented
        # exit path for "the agent did not learn well enough yet".
        logger.error(f"Curriculum failed: {exc}")
        return 1
    except FileNotFoundError as exc:
        logger.error(f"Missing prerequisite: {exc}")
        return 2
    logger.success(f"Curriculum complete. Final model at {final_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
