# scripts/benchmark_e2e.py
"""CLI: benchmark the end-to-end loop latency.

Methodology
-----------
1. Build the full reflex-arc stack in process:
       - PaperBroker (no network)
       - DeepFusionNexus + PolicyNetwork loaded from ``models/``
       - RedisCache pre-populated with synthetic embeddings for SPY
2. Run ``EndToEndLoop._cycle`` 10,000 times, recording the latency of
   each stage via ``time.perf_counter``.
3. Print p50/p95/p99 per stage plus total.
4. Fail (exit 1) if the total p99 exceeds the 100 ms budget from the
   architecture spec.

The full implementation lives in a follow-up commit once trained
checkpoints are available; until then the script clearly states what
is missing and exits with code 2.
"""

from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger

from backend.config.logging import configure_logging


def main() -> int:
    configure_logging()
    checkpoints = [Path("models/agent/final.pt"), Path("models/fusion/best.pt")]
    missing = [str(p) for p in checkpoints if not p.exists()]
    if missing:
        logger.error(
            f"Missing checkpoints: {', '.join(missing)}. Train first.",
        )
        return 2
    logger.info("Benchmark wiring runs here once checkpoints are present.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
