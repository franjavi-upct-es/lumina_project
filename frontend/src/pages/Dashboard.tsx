// frontend/src/pages/Dashboard.tsx
//
// Operator dashboard. Top strip holds the live KPI tiles; the body
// stacks the equity curve + cross-modal attention heatmap, the risk
// table and the agent direction panel.

import { useEffect, useState } from "react";
import { AgentStatusPanel } from "../components/dashboard/AgentStatusPanel";
import { AttentionHeatmap } from "../components/dashboard/AttentionHeatmap";
import { EquityCurve } from "../components/dashboard/EquityCurve";
import { KillSwitchButton } from "../components/dashboard/KillSwitchButton";
import { RiskPanel } from "../components/dashboard/RiskPanel";
import { usePortfolio, usePortfolioHistory } from "../hooks/usePortfolio";
import { agentApi } from "../api/agent";
import type { AgentStatus } from "../types/agent.types";

interface KpiTileProps {
  label: string;
  value: string;
  sub?: string;
  subTone?: "pos" | "neg" | "muted";
}

function KpiTile({ label, value, sub, subTone = "muted" }: KpiTileProps) {
  const toneClass = subTone === "pos" ? "lx-pos" : subTone === "neg" ? "lx-neg" : "";
  return (
    <div className="lx-kpi">
      <div className="lx-kpi-label">{label}</div>
      <div className="lx-kpi-value" style={subTone === "pos" ? { color: "var(--green)" } : subTone === "neg" ? { color: "var(--red)" } : {}}>
        {value}
      </div>
      <div style={{ display: "flex", alignItems: "center" }}>
        {sub && <span className={`lx-kpi-sub ${toneClass}`}>{sub}</span>}
      </div>
    </div>
  );
}

const EQUITY_RANGES: Array<{ id: "1D" | "7D" | "30D" | "90D" | "YTD" | "ALL"; label: string }> = [
  { id: "1D",  label: "1D" },
  { id: "7D",  label: "7D" },
  { id: "30D", label: "30D" },
  { id: "90D", label: "90D" },
  { id: "YTD", label: "YTD" },
  { id: "ALL", label: "ALL" },
];

export function Dashboard() {
  const [range, setRange] = useState<typeof EQUITY_RANGES[number]["id"]>("90D");
  const { portfolio } = usePortfolio();
  const [realAttn, setRealAttn] = useState<number[][] | undefined>();
  const [agentStatus, setAgentStatus] = useState<AgentStatus | null>(null);

  // Fetch real data from backend
  const { history: equity } = usePortfolioHistory(range);

  useEffect(() => {
    const tick = async () => {
      try {
        const status = await agentApi.getStatus();
        setAgentStatus(status);
        if (status.attention_weights) {
           const w = status.attention_weights;
           setRealAttn([
             [w[0], 0.1, 0.1],
             [0.1, w[1], 0.1],
             [0.1, 0.1, w[2]]
           ]);
        }
      } catch {}
    };
    tick();
    const id = setInterval(tick, 5000);
    return () => clearInterval(id);
  }, []);

  const equityStats = summarizeEquity(equity);
  const hasAgentAction = Boolean(agentStatus?.has_action);

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
      <section
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(7, minmax(0, 1fr))",
          gap: 12,
        }}
      >
        <KpiTile label="Equity" value={formatUsd(portfolio?.equity)} />
        <KpiTile label="P&L · Range" value={formatUsdDelta(equityStats.pnl)} sub={formatPct(equityStats.ret)} subTone={(equityStats.pnl ?? 0) >= 0 ? "pos" : "neg"} />
        <KpiTile label="Drawdown" value={formatPct(portfolio?.drawdown_pct)} sub={`peak ${formatUsd(portfolio?.peak_equity)}`} subTone="neg" />
        <KpiTile label="Sharpe · 30D" value="—" sub="not exposed by API" />
        <KpiTile label="Position" value={hasAgentAction ? `${(agentStatus!.current_action * 100).toFixed(1)}%` : "—"} sub={hasAgentAction ? "latest agent action" : "no action yet"} />
        <KpiTile label="Uncertainty" value={hasAgentAction ? agentStatus!.uncertainty.toFixed(3) : "—"} sub={agentStatus?.gate_active ? "gate active" : hasAgentAction ? "gate open" : "no action yet"} subTone={agentStatus?.gate_active ? "neg" : "muted"} />
        <KpiTile label="Latency" value="—" sub="not exposed by API" />
      </section>

      <section style={{ display: "grid", gridTemplateColumns: "2fr 1fr", gap: 16 }}>
        <div className="lx-panel">
          <header
            style={{
              display: "flex",
              justifyContent: "space-between",
              alignItems: "center",
              marginBottom: 10,
            }}
          >
            <div>
              <div className="lx-label">Equity</div>
              <h2 style={{ margin: "2px 0 0 0", fontSize: 14, fontWeight: 600 }}>Curve</h2>
            </div>
            <div style={{ display: "flex", gap: 4 }}>
              {EQUITY_RANGES.map((r) => (
                <button
                  key={r.id}
                  className={`lx-btn ghost${range === r.id ? " active" : ""}`}
                  style={{ padding: "4px 10px" }}
                  onClick={() => setRange(r.id)}
                >
                  {r.label}
                </button>
              ))}
            </div>
          </header>

          <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 4, alignItems: "baseline" }}>
            <div>
              <div className="lx-label">NAV</div>
              <div className="lx-mono" style={{ fontSize: 20, fontWeight: 600 }}>{formatUsd(portfolio?.equity)}</div>
              <div className="lx-mono" style={{ fontSize: 11, color: "var(--text-muted)" }}>
                <span className={(equityStats.pnl ?? 0) >= 0 ? "lx-pos" : "lx-neg"}>{formatUsdDelta(equityStats.pnl)}</span>{" "}
                <span className={(equityStats.ret ?? 0) >= 0 ? "lx-pos" : "lx-neg"}>{formatPct(equityStats.ret)}</span>{" "}
                <span className="lx-dim">selected range</span>
              </div>
            </div>
            <div
              style={{
                display: "flex",
                gap: 14,
                fontFamily: "var(--font-mono)",
                fontSize: 10,
                color: "var(--text-secondary)",
              }}
            >
              <LegendDot color="var(--accent-bright)" label="Equity" />
            </div>
          </div>

          <EquityCurve data={equity} height={300} />

          <div style={{ display: "grid", gridTemplateColumns: "repeat(3, minmax(0, 1fr))", gap: 16, marginTop: 14 }}>
            <MiniMetric label="P&L cumulative" value={formatUsdDelta(equityStats.pnl)} tone={(equityStats.pnl ?? 0) >= 0 ? "pos" : "neg"} />
            <MiniMetric label="Drawdown %" value={formatPct(portfolio?.drawdown_pct)} tone="neg" />
            <MiniMetric label="Daily returns σ" value={formatPct(equityStats.sigma)} tone="muted" />
          </div>
        </div>

        <AttentionHeatmap weights={realAttn} />
      </section>

      <section style={{ display: "grid", gridTemplateColumns: "2fr 1fr", gap: 16 }}>
        <RiskPanel />
        <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
          <AgentStatusPanel />
          <KillSwitchButton />
        </div>
      </section>
    </div>
  );
}

function LegendDot({ color, label }: { color: string; label: string }) {
  return (
    <span style={{ display: "inline-flex", alignItems: "center", gap: 5 }}>
      <span style={{ width: 8, height: 2, background: color, display: "inline-block" }} />
      {label}
    </span>
  );
}

function MiniMetric({
  label,
  value,
  tone,
}: {
  label: string;
  value: string;
  tone: "pos" | "neg" | "muted";
}) {
  const valueColor =
    tone === "pos" ? "var(--green)" : tone === "neg" ? "var(--red)" : "var(--text-primary)";
  return (
    <div>
      <div className="lx-label" style={{ marginBottom: 4 }}>{label}</div>
      <div className="lx-mono" style={{ fontSize: 16, fontWeight: 600, color: valueColor }}>{value}</div>
    </div>
  );
}

function summarizeEquity(history: Array<{ equity: number }>): {
  pnl?: number;
  ret?: number;
  sigma?: number;
} {
  if (history.length < 2) return {};
  const first = history[0].equity;
  const last = history[history.length - 1].equity;
  const returns: number[] = [];
  for (let i = 1; i < history.length; i += 1) {
    const prior = history[i - 1].equity;
    if (prior > 0) returns.push(history[i].equity / prior - 1);
  }
  return {
    pnl: last - first,
    ret: first > 0 ? last / first - 1 : undefined,
    sigma: stddev(returns),
  };
}

function stddev(values: number[]): number | undefined {
  if (values.length < 2) return undefined;
  const mean = values.reduce((sum, value) => sum + value, 0) / values.length;
  const variance = values.reduce((sum, value) => sum + (value - mean) ** 2, 0) / values.length;
  return Math.sqrt(variance);
}

function formatUsd(value: number | undefined): string {
  if (value === undefined || value === null || Number.isNaN(value)) return "—";
  return value.toLocaleString("en-US", {
    style: "currency",
    currency: "USD",
    maximumFractionDigits: 0,
  });
}

function formatUsdDelta(value: number | undefined): string {
  if (value === undefined || value === null || Number.isNaN(value)) return "—";
  const abs = Math.abs(value).toLocaleString("en-US", {
    style: "currency",
    currency: "USD",
    maximumFractionDigits: 0,
  });
  return `${value >= 0 ? "+" : "-"}${abs}`;
}

function formatPct(value: number | undefined): string {
  if (value === undefined || value === null || Number.isNaN(value)) return "—";
  return `${(value * 100).toFixed(2)}%`;
}
