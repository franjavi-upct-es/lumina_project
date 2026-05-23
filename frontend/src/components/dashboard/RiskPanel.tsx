// frontend/src/components/dashboard/RiskPanel.tsx
//
// Portfolio-level risk surface for the dashboard.
//
// Reads from the `usePortfolio` hook, which polls /api/portfolio every
// 3 s. The endpoint returns the live broker snapshot plus the running
// peak equity (so drawdown is computed API-side, see backend/api/routes/
// portfolio.py for the rationale).
//
// Layout
// ------
//   ┌──────────────────────────────────────────────────────┐
//   │ Risk                                                 │
//   ├──────────────────────────────────────────────────────┤
//   │ Equity:    $100,234     Cash: $20,001                │
//   │ Peak:      $103,580     Buy-power: $40,002           │
//   │ Drawdown:  ──────────────────────  3.24%             │
//   ├──────────────────────────────────────────────────────┤
//   │ Positions                                            │
//   │   AAPL   100   @ 180.20   unr. PnL  +$245   1.2% eq │
//   │   MSFT    50   @ 400.10   unr. PnL  -$ 15   0.7% eq │
//   └──────────────────────────────────────────────────────┘

import { usePortfolio } from "../../hooks/usePortfolio";
import type { Portfolio, Position } from "../../types/market.types";

// Drawdown thresholds. Mirrors MAX_DRAWDOWN_LIMIT in backend settings
// (default 0.20). The visualisation uses three bands so operators see
// the progression toward the limit, not just "are we there yet".
const DRAWDOWN_WARN = 0.05;
const DRAWDOWN_DANGER = 0.10;
const DRAWDOWN_LIMIT = 0.20;

function formatUsd(value: number): string {
  return value.toLocaleString("en-US", {
    style: "currency",
    currency: "USD",
    maximumFractionDigits: 0,
  });
}

function formatPct(value: number, digits = 2): string {
  return `${(value * 100).toFixed(digits)}%`;
}

function drawdownColor(dd: number): string {
  if (dd >= DRAWDOWN_DANGER) return "#cf222e";
  if (dd >= DRAWDOWN_WARN) return "#ffb020";
  return "#28a745";
}

function DrawdownBar({ value }: { value: number }) {
  const widthPct = Math.min(100, (value / DRAWDOWN_LIMIT) * 100);
  const color = drawdownColor(value);
  return (
    <div
      style={{
        position: "relative",
        height: 12,
        background: "#f6f8fa",
        borderRadius: 4,
        overflow: "hidden",
        border: "1px solid #d0d7de",
      }}
    >
      <div
        style={{
          width: `${widthPct}%`,
          height: "100%",
          background: color,
          transition: "width 250ms ease-out, background 250ms ease-out",
        }}
      />
      {/* Warning and danger ticks */}
      {[DRAWDOWN_WARN, DRAWDOWN_DANGER].map((threshold) => (
        <div
          key={threshold}
          style={{
            position: "absolute",
            left: `${(threshold / DRAWDOWN_LIMIT) * 100}%`,
            top: 0,
            bottom: 0,
            width: 1,
            background: "#8c959f",
          }}
        />
      ))}
    </div>
  );
}

function StatRow({ label, value, hint }: { label: string; value: string; hint?: string }) {
  return (
    <div style={{ display: "flex", justifyContent: "space-between", padding: "4px 0", fontSize: 13 }}>
      <span style={{ color: "#586069" }}>{label}</span>
      <span style={{ fontFamily: "monospace", fontWeight: 600 }}>
        {value}
        {hint && <span style={{ marginLeft: 6, color: "#8c959f", fontWeight: 400 }}>{hint}</span>}
      </span>
    </div>
  );
}

function PositionsTable({ positions, equity }: { positions: Position[]; equity: number }) {
  if (positions.length === 0) {
    return (
      <div style={{ padding: 12, textAlign: "center", color: "#8c959f", fontSize: 13 }}>
        No open positions.
      </div>
    );
  }
  return (
    <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 12 }}>
      <thead>
        <tr style={{ borderBottom: "1px solid #d0d7de", color: "#586069" }}>
          <th style={{ textAlign: "left", padding: "6px 8px" }}>Ticker</th>
          <th style={{ textAlign: "right", padding: "6px 8px" }}>Qty</th>
          <th style={{ textAlign: "right", padding: "6px 8px" }}>Avg Px</th>
          <th style={{ textAlign: "right", padding: "6px 8px" }}>Unrealised P&L</th>
          <th style={{ textAlign: "right", padding: "6px 8px" }}>Exposure</th>
        </tr>
      </thead>
      <tbody>
        {positions.map((p) => {
          const exposurePct = equity > 0 ? p.market_value / equity : 0;
          const pnlColor = p.unrealized_pnl >= 0 ? "#1a7f37" : "#cf222e";
          return (
            <tr key={p.ticker} style={{ borderBottom: "1px solid #eaecef" }}>
              <td style={{ padding: "6px 8px", fontWeight: 600 }}>{p.ticker}</td>
              <td style={{ padding: "6px 8px", textAlign: "right", fontFamily: "monospace" }}>
                {p.qty.toLocaleString()}
              </td>
              <td style={{ padding: "6px 8px", textAlign: "right", fontFamily: "monospace" }}>
                ${p.avg_entry_price.toFixed(2)}
              </td>
              <td
                style={{
                  padding: "6px 8px",
                  textAlign: "right",
                  fontFamily: "monospace",
                  color: pnlColor,
                  fontWeight: 600,
                }}
              >
                {p.unrealized_pnl >= 0 ? "+" : ""}
                {formatUsd(p.unrealized_pnl)}
              </td>
              <td style={{ padding: "6px 8px", textAlign: "right", fontFamily: "monospace" }}>
                {formatPct(exposurePct, 1)}
              </td>
            </tr>
          );
        })}
      </tbody>
    </table>
  );
}

function RiskPanelBody({ portfolio }: { portfolio: Portfolio }) {
  return (
    <>
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8, marginBottom: 12 }}>
        <StatRow label="Equity" value={formatUsd(portfolio.equity)} />
        <StatRow label="Cash" value={formatUsd(portfolio.cash)} />
        <StatRow label="Peak" value={formatUsd(portfolio.peak_equity)} />
        <StatRow label="Buying power" value={formatUsd(portfolio.buying_power)} />
      </div>
      <div style={{ marginBottom: 12 }}>
        <div
          style={{
            display: "flex",
            justifyContent: "space-between",
            fontSize: 12,
            color: "#586069",
            marginBottom: 4,
          }}
        >
          <span>Drawdown</span>
          <span style={{ color: drawdownColor(portfolio.drawdown_pct), fontWeight: 600 }}>
            {formatPct(portfolio.drawdown_pct, 2)} / {formatPct(DRAWDOWN_LIMIT, 0)} limit
          </span>
        </div>
        <DrawdownBar value={portfolio.drawdown_pct} />
      </div>
      <PositionsTable positions={portfolio.positions} equity={portfolio.equity} />
    </>
  );
}

export function RiskPanel() {
  const { portfolio, error } = usePortfolio();
  return (
    <section
      style={{
        border: "1px solid #d0d7de",
        borderRadius: 6,
        padding: 16,
        marginBottom: 16,
        background: "#ffffff",
      }}
    >
      <header style={{ marginBottom: 12 }}>
        <h2 style={{ margin: 0, fontSize: 16, fontWeight: 600 }}>Risk</h2>
      </header>
      {error && (
        <div
          style={{
            padding: 8,
            background: "#ffeef0",
            color: "#cf222e",
            borderRadius: 4,
            fontSize: 12,
            marginBottom: 8,
          }}
        >
          Failed to load portfolio: {error.message}
        </div>
      )}
      {portfolio ? (
        <RiskPanelBody portfolio={portfolio} />
      ) : (
        <div style={{ padding: 12, color: "#8c959f", fontSize: 13, textAlign: "center" }}>
          Loading portfolio…
        </div>
      )}
    </section>
  );
}
