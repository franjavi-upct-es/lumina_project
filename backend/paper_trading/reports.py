# backend/paper_trading/reports.py
"""Daily report generation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path


@dataclass
class DailyReport:
    report_date: date
    starting_equity: float
    ending_equity: float
    total_return_pct: float
    sharpe_intraday: float
    max_drawdown_pct: float
    num_trades: int
    num_vetoes: int
    kill_switch_state_final: str

    def to_markdown(self) -> str:
        lines = [f"# Lumina V3 — Daily Report {self.report_date}\n"]
        for k, v in asdict(self).items():
            if k == "report_date":
                continue
            lines.append(f"- **{k}**: {v}")
        return "\n".join(lines) + "\n"

    def save(self, root: Path = Path("reports/daily")) -> Path:
        root.mkdir(parents=True, exist_ok=True)
        path = root / f"{self.report_date.isoformat()}.md"
        path.write_text(self.to_markdown())
        return path
