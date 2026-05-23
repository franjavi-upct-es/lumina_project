// frontend/src/pages/Dashboard.tsx
//
// Main operator dashboard. Composes:
//   * AgentStatusPanel  — direction + uncertainty + veto state
//   * EquityCurve       — historical P&L over the session
//   * RiskPanel         — positions + drawdown
//   * AttentionHeatmap  — cross-modal attention (3 × 3)
//   * KillSwitchButton  — emergency liquidation
//
// The connection status indicator (top right) shows whether the live
// WebSocket is up. The perception-health indicator (top right, smaller)
// shows whether the API health endpoint is happy with infrastructure.

import { ConnectionStatus } from "../components/common/ConnectionStatus";
import { AgentStatusPanel } from "../components/dashboard/AgentStatusPanel";
import { AttentionHeatmap } from "../components/dashboard/AttentionHeatmap";
import { EquityCurve } from "../components/dashboard/EquityCurve";
import { KillSwitchButton } from "../components/dashboard/KillSwitchButton";
import { RiskPanel } from "../components/dashboard/RiskPanel";
import { useAgentStream } from "../hooks/useAgentStream";
import { usePerceptionHealth } from "../hooks/usePerceptionHealth";

// Placeholder attention matrix used until the fusion service exposes a
// real one. Strong diagonal (each modality attends to itself first) +
// modest cross-modal bleed, which is the typical trained shape.
const DEMO_ATTENTION = [
  [0.62, 0.22, 0.16],
  [0.18, 0.68, 0.14],
  [0.20, 0.18, 0.62],
];

export function Dashboard() {
  const { connected } = useAgentStream();
  const health = usePerceptionHealth();

  return (
    <div style={{ padding: 16, maxWidth: 1200, margin: "0 auto" }}>
      <header
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          marginBottom: 16,
        }}
      >
        <h1 style={{ margin: 0, fontSize: 22, fontWeight: 600 }}>Lumina V3</h1>
        <div style={{ display: "flex", alignItems: "center", gap: 16, fontSize: 12 }}>
          <span title="WebSocket connection to /api/agent/stream">
            <ConnectionStatus connected={connected} />
          </span>
          <span
            title={health.error ? `Error: ${health.error.message}` : "Aggregate backend health"}
            style={{ color: health.healthy ? "#1a7f37" : "#cf222e" }}
          >
            ● {health.status ?? "no data"}
          </span>
        </div>
      </header>

      <AgentStatusPanel />

      <div style={{ display: "grid", gridTemplateColumns: "2fr 1fr", gap: 16, marginBottom: 16 }}>
        <section style={panelStyle}>
          <h2 style={{ margin: 0, fontSize: 16, fontWeight: 600, marginBottom: 8 }}>Equity</h2>
          <EquityCurve data={[]} />
        </section>
        <AttentionHeatmap weights={DEMO_ATTENTION} />
      </div>

      <RiskPanel />
      <KillSwitchButton />
    </div>
  );
}

const panelStyle: React.CSSProperties = {
  border: "1px solid #d0d7de",
  borderRadius: 6,
  padding: 16,
  background: "#ffffff",
};
