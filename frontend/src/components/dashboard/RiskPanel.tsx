// frontend/src/components/dashboard/RiskPanel.tsx
//
// Risk · Positions — dark-theme table styled after the operator console
// mockup. Header row carries portfolio aggregates; the table lists each
// open position with quantity, mark, market value, weight, beta, VaR,
// P&L and an exposure bar.

import { usePortfolio } from "../../hooks/usePortfolio";
import type { Portfolio, Position } from "../../types/market.types";

function formatUsd(value: number): string {
  return value.toLocaleString("en-US", {
    style: "currency",
    currency: "USD",
    maximumFractionDigits: 0,
  });
}

function formatPct(value: number, digits = 1): string {
  return `${(value * 100).toFixed(digits)}%`;
}

function ExposureBar({ pct, side }: { pct: number; side: "long" | "short" }) {
  const w = Math.max(2, Math.min(100, pct * 100));
  return (
    <div className={`lx-bar ${side === "short" ? "red" : ""}`} style={{ width: 96, marginLeft: "auto" }}>
      <div className="lx-bar-fill" style={{ width: `${w}%` }} />
    </div>
  );
}

interface SyntheticRow {
  weight: number;
  beta: number;
  var95: number;
}

function syntheticRow(p: Position, equity: number): SyntheticRow {
  return {
    weight: equity > 0 ? p.market_value / equity : 0,
    beta: 1.0, // Default to 1.0 for paper trading until live beta is streaming
    var95: Math.round(Math.abs(p.market_value) * 0.012),
  };
}

function RiskPanelBody({ portfolio }: { portfolio: Portfolio }) {
  return (
    <>
      <div
        style={{
          display: "flex",
          flexWrap: "wrap",
          gap: 24,
          padding: "4px 0 14px 0",
          color: "var(--text-secondary)",
          fontSize: 12,
        }}
      >
        <Stat label="Gross"      value={`$${(portfolio.equity / 1000).toFixed(1)}k`} />
        <Stat label="Net"        value={`$${((portfolio.equity - portfolio.cash) / 1000).toFixed(1)}k`} />
        <Stat label="Leverage"   value="1.00×" />
        <Stat label="β-portfolio" value="1.00" />
        <Stat label="VaR₉₅ 1d"   value="--" />
      </div>

      <table className="lx-table">
        <thead>
          <tr>
            <th>Ticker</th>
            <th>Side</th>
            <th className="r">Qty</th>
            <th className="r">Last</th>
            <th className="r">Mkt Val</th>
            <th className="r">Weight</th>
            <th className="r">β</th>
            <th className="r">VaR₉₅</th>
            <th className="r">P&amp;L</th>
            <th className="r" style={{ width: 130 }}>Exposure</th>
          </tr>
        </thead>
        <tbody>
          {portfolio.positions.length === 0 ? (
            <tr>
              <td colSpan={10} style={{ textAlign: "center", color: "var(--text-muted)", padding: 24 }}>
                No open positions.
              </td>
            </tr>
          ) : (
            portfolio.positions.map((p) => {
              const meta = syntheticRow(p, portfolio.equity);
              const side: "long" | "short" = p.qty >= 0 ? "long" : "short";
              const pnlColor = p.unrealized_pnl >= 0 ? "var(--green)" : "var(--red)";
              return (
                <tr key={p.ticker}>
                  <td style={{ fontWeight: 600, color: "var(--text-primary)" }}>{p.ticker}</td>
                  <td><span className={`lx-tag ${side}`}>{side.toUpperCase()}</span></td>
                  <td className="r">{Math.abs(p.qty).toLocaleString()}</td>
                  <td className="r">{p.avg_entry_price.toFixed(2)}</td>
                  <td className="r">${p.market_value.toLocaleString()}</td>
                  <td className="r">{formatPct(meta.weight, 1)}</td>
                  <td className="r">{meta.beta.toFixed(2)}</td>
                  <td className="r" style={{ color: "var(--amber)" }}>${meta.var95}</td>
                  <td className="r" style={{ color: pnlColor, fontWeight: 600 }}>
                    {p.unrealized_pnl >= 0 ? "+" : "−"}${Math.abs(p.unrealized_pnl).toLocaleString()}
                  </td>
                  <td className="r"><ExposureBar pct={meta.weight} side={side} /></td>
                </tr>
              );
            })
          )}
        </tbody>
      </table>
    </>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <span style={{ display: "inline-flex", alignItems: "baseline", gap: 8 }}>
      <span style={{ fontSize: 11, color: "var(--text-muted)" }}>{label}</span>
      <span className="lx-mono" style={{ color: "var(--text-primary)", fontWeight: 500 }}>{value}</span>
    </span>
  );
}

export function RiskPanel() {
  const { portfolio, error } = usePortfolio();
  return (
    <section className="lx-panel">
      <header
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          marginBottom: 8,
        }}
      >
        <div>
          <div className="lx-label">Risk · Positions</div>
        </div>
        <button className="lx-btn ghost" type="button">EXPORT</button>
      </header>

      {error && (
        <div
          style={{
            padding: 8,
            background: "var(--red-soft)",
            color: "var(--red)",
            borderRadius: 6,
            fontSize: 12,
            marginBottom: 10,
            border: "1px solid rgba(239,68,68,0.4)",
          }}
        >
          Live broker feed unreachable ({error.message}).
        </div>
      )}
      {portfolio ? (
        <RiskPanelBody portfolio={portfolio} />
      ) : (
        <div className="lx-dim" style={{ padding: 24, textAlign: "center", fontSize: 13 }}>
          No real portfolio snapshot available yet.
        </div>
      )}
    </section>
  );
}
