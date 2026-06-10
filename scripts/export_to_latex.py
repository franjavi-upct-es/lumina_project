# scripts/export_to_latex.py
"""Utility: export simulation KPIs to pgfplots LaTeX code."""

import csv
from pathlib import Path


def generate_pgfplots(csv_path: Path) -> None:
    if not csv_path.exists():
        print(f"Error: {csv_path} not found.")
        return

    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    # Example: Plotting Benchmark Latency
    latency_rows = [r for r in rows if r.get("category") == "latency"]

    print("\n% --- PGFPLOTS: LATENCY PROFILE ---")
    print("\\begin{tikzpicture}")
    print(
        "  \\begin{axis}[ybar, symbolic x coords={p50, p95, p99}, xtick=data, "
        "ylabel={Latency (ms)}, title={Reflex Arc Latency Profile}]"
    )
    print("    \\addplot coordinates {")
    for r in latency_rows:
        print(f"      ({r['metric']}, {r['value']})")
    print("    };")
    print(
        "    \\addplot[red, sharp plot, dashed, update limits=false] "
        "coordinates {(p50, 100) (p99, 100)} node[above] {Budget};"
    )
    print("  \\end{axis}")
    print("\\end{tikzpicture}")

    # Example: Plotting Crisis Survival (Comparison)
    safety_rows = [r for r in rows if r.get("category") == "safety"]
    print("\n% --- PGFPLOTS: CRISIS SURVIVAL ---")
    print("\\begin{tikzpicture}")
    print("  \\begin{axis}[ylabel={Final Equity ($)}, title={Model Resilience: Lehman vs Crypto}]")
    print("    \\addplot[blue, mark=square*] coordinates {")
    for r in safety_rows:
        if r["metric"] == "survival_equity":
            print(f"      (1, {r['value']})")
    print("    };")
    print("  \\end{axis}")
    print("\\end{tikzpicture}")

    # Example: Plotting Performance Evolution with Reliability Bands
    sim_rows = [
        r
        for r in rows
        if r.get("category") == "history" and r.get("metric", "").endswith("_sharpe")
    ]
    if sim_rows:
        print("\n% --- PGFPLOTS: STABILITY & RELIABILITY ---")
        print("% Reliability band is a fixed visual tolerance around recorded Sharpe metrics.")
        print("\\begin{tikzpicture}")
        print(
            "  \\begin{axis}[xlabel={Research Cycle}, ylabel={Sharpe Ratio}, "
            "title={Model Stability Profile}, grid=major, legend pos=north west]"
        )

        print("    \\addplot[name path=A, gray, opacity=0.2, forget plot] coordinates {")
        for i, r in enumerate(sim_rows):
            val = float(r["value"])
            print(f"      ({i + 1}, {val - 0.05})")
        print("    };")

        print("    \\addplot[name path=B, gray, opacity=0.2, forget plot] coordinates {")
        for i, r in enumerate(sim_rows):
            val = float(r["value"])
            print(f"      ({i + 1}, {val + 0.05})")
        print("    };")

        print("    \\addplot[gray!20, forget plot] fill between[of=A and B];")

        print("    \\addplot[blue, mark=*, thick] coordinates {")
        for i, r in enumerate(sim_rows):
            print(f"      ({i + 1}, {r['value']})")
        print("    };")
        print("    \\addlegendentry{Policy Mean (Stable)}")

        print("  \\end{axis}")
        print("\\end{tikzpicture}")


if __name__ == "__main__":
    reports_dir = Path("reports")
    master_csv = reports_dir / "master_performance_kpis.csv"
    generate_pgfplots(master_csv)
