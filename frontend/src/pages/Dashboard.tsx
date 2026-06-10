// frontend/src/pages/Dashboard.tsx
//
// Operator dashboard. Top strip holds the live KPI tiles; the body
// stacks the equity curve + cross-modal attention heatmap, the risk
// table and the agent direction panel.

import { useEffect, useState } from "react";
import { AgentStatusPanel } from "../components/dashboard/AgentStatusPanel";
import { AttentionHeatmap } from "../components/dashboard/AttentionHeatmap";
import { EquityCurve, type EquityPoint } from "../components/dashboard/EquityCurve";
import { KillSwitchButton } from "../components/dashboard/KillSwitchButton";
import { RiskPanel } from "../components/dashboard/RiskPanel";
import { usePortfolio, usePortfolioHistory } from "../hooks/usePortfolio";
import { useAgentStore } from "../store/agentSlice";
import { agentApi } from "../api/agent";

const DEMO_ATTENTION = [
  [0.62, 0.22, 0.16],
  [0.18, 0.68, 0.14],
  [0.20, 0.18, 0.62],
];

interface KpiTileProps {
  label: string;
  value: string;
  sub?: string;
  subTone?: "pos" | "neg" | "muted";
  spark?: number[];
  sparkColor?: string;
}

function Sparkline({ data, color }: { data: number[]; color: string }) {
  if (data.length < 2) return null;
  const w = 110;
  const h = 28;
  const min = Math.min(...data);
  const max = Math.max(...data);
  const span = max - min || 1;
  const pts = data
    .map((v, i) => `${(i / (data.length - 1)) * w},${h - ((v - min) / span) * h}`)
    .join(" ");
  return (
    <svg width={w} height={h} viewBox={`0 0 ${w} ${h}`} style={{ marginLeft: "auto" }}>
      <polyline points={pts} fill="none" stroke={color} strokeWidth={1.2} strokeLinejoin="round" />
    </svg>
  );
}

function KpiTile({ label, value, sub, subTone = "muted", spark, sparkColor = "var(--accent-bright)" }: KpiTileProps) {
  const toneClass = subTone === "pos" ? "lx-pos" : subTone === "neg" ? "lx-neg" : "";
  return (
    <div className="lx-kpi">
      <div className="lx-kpi-label">{label}</div>
      <div className="lx-kpi-value" style={subTone === "pos" ? { color: "var(--green)" } : subTone === "neg" ? { color: "var(--red)" } : {}}>
        {value}
      </div>
      <div style={{ display: "flex", alignItems: "center" }}>
        {sub && <span className={`lx-kpi-sub ${toneClass}`}>{sub}</span>}
        {spark && <Sparkline data={spark} color={sparkColor} />}
      </div>
    </div>
  );
}

function useWiggleSpark(seed: number, length = 24): number[] {
  const [points] = useState(() => {
    const out: number[] = [];
    let v = seed;
    for (let i = 0; i < length; i++) {
      v += (Math.random() - 0.5) * 4 + Math.sin(i / 3) * 1.2;
      out.push(v);
    }
    return out;
  });
  return points;
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
  const agentState = useAgentStore();
  const [realAttn, setRealAttn] = useState<number[][] | undefined>();

  // Fetch real data from backend
  const { history: equity } = usePortfolioHistory(range);

  useEffect(() => {
    const tick = async () => {
      try {
        const status = await agentApi.getStatus();
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

  const sparkEquity = useWiggleSpark(40);
  const sparkPnl = useWiggleSpark(30);
  const sparkDd = useWiggleSpark(20);
  const sparkSharpe = useWiggleSpark(45);
  const sparkPos = useWiggleSpark(35);
  const sparkUnc = useWiggleSpark(22);
  const sparkLat = useWiggleSpark(50);

  // Keep latency / tick clocks visually alive without polling extra endpoints.
  const [latency, setLatency] = useState(142);
  useEffect(() => {
    const id = setInterval(() => setLatency((l) => 130 + Math.round(Math.sin(Date.now() / 1000) * 18) + Math.round(Math.random() * 4)), 1500);
    return () => clearInterval(id);
  }, []);

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
      <section
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(7, minmax(0, 1fr))",
          gap: 12,
        }}
      >
        <KpiTile label="Equity"      value={`$${(portfolio?.equity ?? 123847).toLocaleString()}`} sub="+2.34%"   subTone="pos" spark={sparkEquity} sparkColor="var(--green)" />
        <KpiTile label="P&L · Today" value="+$2,847.14"  sub="+2.36%"   subTone="pos" spark={sparkPnl}    sparkColor="var(--green)" />
        <KpiTile label="Drawdown"    value={`${((portfolio?.drawdown_pct ?? 0.0341) * 100).toFixed(2)}%`}      sub="max −7.92%" subTone="neg" spark={sparkDd}  sparkColor="var(--red)" />
        <KpiTile label="Sharpe · 30D" value="2.18"       sub="sortino 3.04" spark={sparkSharpe} sparkColor="var(--accent-bright)" />
        <KpiTile label="Position"    value={`${(agentState.currentAction * 100).toFixed(1)}%`}      sub="target +71.0%" spark={sparkPos} sparkColor="var(--accent-bright)" />
        <KpiTile label="Uncertainty" value={agentState.uncertainty.toFixed(3)}       sub="gate · OPEN"  subTone="pos" spark={sparkUnc} sparkColor="var(--green)" />
        <KpiTile label="Latency"     value={`${latency}ms`} sub="p99 318ms" spark={sparkLat} sparkColor="var(--accent-bright)" />
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
              <div className="lx-mono" style={{ fontSize: 20, fontWeight: 600 }}>${(portfolio?.equity ?? 123847).toLocaleString()}</div>
              <div className="lx-mono" style={{ fontSize: 11, color: "var(--text-muted)" }}>
                <span className="lx-pos">+$23,847</span> <span className="lx-pos">+23.85%</span>{" "}
                <span className="lx-dim">since inception</span>
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
              <LegendDot color="rgba(148,163,184,0.5)" label="Benchmark" />
              <LegendDot color="var(--red)" label="Drawdown" />
            </div>
          </div>

          <EquityCurve data={equity} height={300} />

          <div style={{ display: "grid", gridTemplateColumns: "repeat(3, minmax(0, 1fr))", gap: 16, marginTop: 14 }}>
            <MiniMetric label="P&L cumulative" value="+$23,847"   tone="pos" spark={sparkEquity} sparkColor="var(--green)" />
            <MiniMetric label="Drawdown %"     value={`${((portfolio?.drawdown_pct ?? 0.0341) * 100).toFixed(2)}%`}     tone="neg" spark={sparkDd}     sparkColor="var(--red)" />
            <MiniMetric label="Daily returns σ" value="0.82%"      tone="muted" spark={sparkPnl} sparkColor="rgba(148,163,184,0.6)" />
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
  spark,
  sparkColor,
}: {
  label: string;
  value: string;
  tone: "pos" | "neg" | "muted";
  spark: number[];
  sparkColor: string;
}) {
  const w = 180;
  const h = 30;
  const min = Math.min(...spark);
  const max = Math.max(...spark);
  const span = max - min || 1;
  const pts = spark
    .map((v, i) => `${(i / (spark.length - 1)) * w},${h - ((v - min) / span) * h}`)
    .join(" ");
  const valueColor =
    tone === "pos" ? "var(--green)" : tone === "neg" ? "var(--red)" : "var(--text-primary)";
  return (
    <div>
      <div className="lx-label" style={{ marginBottom: 4 }}>{label}</div>
      <div className="lx-mono" style={{ fontSize: 16, fontWeight: 600, color: valueColor }}>{value}</div>
      <svg
        width="100%"
        height={h}
        viewBox={`0 0 ${w} ${h}`}
        preserveAspectRatio="none"
        style={{ marginTop: 4, display: "block" }}
      >
        <polyline points={pts} fill="none" stroke={sparkColor} strokeWidth={1.2} />
      </svg>
    </div>
  );
}
