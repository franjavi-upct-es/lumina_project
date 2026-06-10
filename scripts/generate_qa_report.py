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
import csv
import json
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
    today = datetime.now(UTC).date().isoformat()
    output = args.out or args.reports_dir / f"qa_report_{today}.md"

    output.write_text(_build_report(args.reports_dir))
    logger.success(f"QA report written to {output}")

    # Generate Master KPI CSV
    _generate_master_csv(args.reports_dir)
    return 0


def _generate_master_csv(reports_dir: Path) -> None:
    kpis = []

    # 1. Latency KPIs
    benchmark_csv = _latest("benchmark_e2e_*.csv", reports_dir)
    if benchmark_csv:
        with open(benchmark_csv) as f:
            reader = csv.DictReader(f)
            for row in reader:
                kpis.append(
                    {
                        "category": "latency",
                        "metric": row["metric"],
                        "value": row["value_ms"],
                        "unit": "ms",
                    }
                )

    # 2. Crisis Drill KPIs
    crisis_json = _latest("crisis_drill_*.json", reports_dir)
    if crisis_json:
        with open(crisis_json) as f:
            trace = json.load(f)
            if trace:
                final_equity = trace[-1]["equity"]
                vetoes = sum(1 for t in trace if t["vetoed"])
                kpis.append(
                    {
                        "category": "safety",
                        "metric": "survival_equity",
                        "value": final_equity,
                        "unit": "$",
                    }
                )
                kpis.append(
                    {
                        "category": "safety",
                        "metric": "total_vetoes",
                        "value": vetoes,
                        "unit": "count",
                    }
                )

    # 3. Article Simulation KPIs (Historical Harvest)
    article_root = Path("artifacts/article_sims")
    if article_root.exists():
        sim_dirs = sorted(article_root.glob("20*"), key=lambda p: p.name)
        for sim_dir in sim_dirs:
            metrics_csv = sim_dir / "arena_metrics.csv"
            if metrics_csv.exists():
                with open(metrics_csv) as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        # Focus on the most challenging test scenario
                        if row["scenario"] == "test_drawdown_stress":
                            run_label = f"Run_{sim_dir.name[:8]}"
                            kpis.append(
                                {
                                    "category": "history",
                                    "metric": f"{run_label}_{row['phase']}_sharpe",
                                    "value": row["mean_sharpe"],
                                    "unit": "ratio",
                                }
                            )
                            kpis.append(
                                {
                                    "category": "history",
                                    "metric": f"{run_label}_{row['phase']}_repeat_rate",
                                    "value": row["bad_action_repeat_rate"],
                                    "unit": "rate",
                                }
                            )

    # 4. Write to CSV
    csv_path = reports_dir / "master_performance_kpis.csv"
    keys = ["category", "metric", "value", "unit"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(kpis)

    logger.success(f"Master KPIs aggregated to {csv_path}")

    _save_summary_plot(kpis, reports_dir)


def _save_summary_plot(kpis: list[dict], reports_dir: Path) -> None:
    import matplotlib.pyplot as plt

    if not kpis:
        return

    # Filter for latency only for a bar chart
    latency_kpis = [k for k in kpis if k["category"] == "latency"]
    if not latency_kpis:
        return

    metrics = [k["metric"] for k in latency_kpis]
    values = [float(k["value"]) for k in latency_kpis]

    plt.figure(figsize=(10, 6))
    plt.bar(metrics, values, color="skyblue")
    plt.axhline(y=100.0, color="red", linestyle="--", label="Budget (100ms)")
    plt.ylabel("Latency (ms)")
    plt.title("Lumina V3 Reflex Arc Latency Profile")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)

    plot_path = reports_dir / "reflex_arc_latency.png"
    plt.savefig(plot_path, dpi=300)
    logger.success(f"Summary visualization saved to {plot_path}")


if __name__ == "__main__":
    sys.exit(main())
