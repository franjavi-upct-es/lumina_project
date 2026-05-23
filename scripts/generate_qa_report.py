# scripts/generate_qa_report.py
"""CLI: aggregate Phase-8 QA artefacts into a single Markdown report.

The QA artefacts are produced by other Phase-8 scripts:

    reports/parity_phase3.md          (scripts/run_parity_check.py)
    reports/crisis_drill_*.json       (scripts/run_crisis_drill.py)
    reports/benchmark_e2e_*.csv       (scripts/benchmark_e2e.py)
    reports/06_calibration.png        (notebooks/06_uncertainty_calibration.py)
    reports/daily/*.md                (backend/paper_trading/reports.py)

This script reads the latest of each, composes a single executive
report and writes it to ``reports/qa_report_<YYYY-MM-DD>.md``.

The skeleton is fully fleshed out; consumer scripts populate the
artefacts as they run. When some artefacts are missing the script emits
a clearly-labelled "Not yet available" placeholder in the corresponding
section so the operator can see at a glance what is and is not ready.
"""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime
from pathlib import Path

from loguru import logger

from backend.config.logging import configure_logging


def _latest(pattern: str, root: Path) -> Path | None:
    matches = sorted(root.glob(pattern))
    return matches[-1] if matches else None


def _read_or_placeholder(path: Path | None, label: str) -> str:
    if path is None or not path.exists():
        return f"_{label} not yet available._\n"
    return path.read_text()


def _build_report(reports_dir: Path) -> str:
    today = datetime.now(UTC).date().isoformat()
    parity = _read_or_placeholder(reports_dir / "parity_phase3.md", "Parity report")
    crisis = _latest("crisis_drill_*.json", reports_dir)
    benchmark = _latest("benchmark_e2e_*.csv", reports_dir)
    daily = _latest("*.md", reports_dir / "daily")

    pieces = [
        f"# Lumina V3 — QA Report {today}\n",
        "## 1. Phase-3 parity\n",
        parity,
        "\n## 2. Crisis drill\n",
        f"Latest artefact: `{crisis}`\n"
        if crisis
        else "_Crisis drill artefact not yet available._\n",
        "\n## 3. Latency benchmarks\n",
        f"Latest CSV: `{benchmark}`\n" if benchmark else "_Benchmark CSV not yet available._\n",
        "\n## 4. Most recent daily report\n",
        daily.read_text() if daily and daily.exists() else "_No daily reports yet._\n",
    ]
    return "\n".join(pieces)


def main() -> int:
    configure_logging()
    parser = argparse.ArgumentParser(description="Aggregate Phase-8 QA artefacts.")
    parser.add_argument("--reports-dir", default="reports", type=Path)
    parser.add_argument("--out", default=None, type=Path)
    args = parser.parse_args()

    args.reports_dir.mkdir(parents=True, exist_ok=True)
    output = args.out or args.reports_dir / f"qa_report_{datetime.now(UTC).date().isoformat()}.md"
    output.write_text(_build_report(args.reports_dir))
    logger.success(f"QA report written to {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
