// frontend/src/components/dashboard/AgentStatusPanel.tsx
//
// At-a-glance live status of the cognitive agent.
//
// Data sources
// ------------
// 1. WebSocket stream (preferred; sub-second latency, pushed)
//      Reads the zustand `agentSlice` which `useAgentStream` keeps in
//      sync. This is the hot path — when the dashboard is connected
//      the panel updates as fast as the agent decides.
// 2. Polled HTTP fallback at 3 s cadence
//      Used when the WebSocket is down or has not yet delivered its
//      first frame. The polled endpoint is /api/agent/status which
//      returns the same fields as the WS payload.
//
// The two sources never fight: the polled fetch is only USED when the
// WS payload has not yet arrived OR is stale (more than 10 s old).
//
// Visual layout
// -------------
//   ┌────────────────────────────────────────────────┐
//   │ Agent Status                       last_update │
//   ├────────────────────────────────────────────────┤
//   │  Action vector (4-D bar chart of -1..+1)       │
//   │  Uncertainty gauge                             │
//   │  Veto badge (green / red)   Consecutive: N     │
//   └────────────────────────────────────────────────┘

import { useEffect, useMemo, useState } from "react";
import { agentApi } from "../../api/agent";
import { useAgentStore } from "../../store/agentSlice";
import type { AgentStatus } from "../../types/agent.types";
import { UncertaintyGauge } from "./UncertaintyGauge";

const POLL_INTERVAL_MS = 3000;
const STALE_AFTER_MS = 10_000;

// Background colour for the action-direction bar. Positive = long (green-ish),
// negative = short (red-ish). The exact hex values match the broader
// dashboard palette.
const COLOR_LONG = "#2ea043";
const COLOR_SHORT = "#cf222e";
const COLOR_BAR_BG = "#f6f8fa";

function ActionDirectionBar({ value }: { value: number }) {
  // value is in [-1, 1]; map to half-bar width on either side of centre.
  const halfWidthPct = Math.min(100, Math.abs(value) * 100);
  const color = value >= 0 ? COLOR_LONG : COLOR_SHORT;
  const side = value >= 0 ? "left: 50%" : `left: ${50 - halfWidthPct}%`;
  return (
    <div
      style={{
        position: "relative",
        height: 18,
        background: COLOR_BAR_BG,
        borderRadius: 4,
        overflow: "hidden",
        border: "1px solid #d0d7de",
      }}
    >
      {/* Centre tick */}
      <div
        style={{
          position: "absolute",
          left: "50%",
          top: 0,
          bottom: 0,
          width: 1,
          background: "#8c959f",
        }}
      />
      {/* Filled portion */}
      <div
        style={{
          position: "absolute",
          ...(value >= 0 ? { left: "50%" } : { left: `${50 - halfWidthPct}%` }),
          width: `${halfWidthPct}%`,
          top: 0,
          bottom: 0,
          background: color,
          transition: "left 200ms ease-out, width 200ms ease-out",
        }}
      />
      {/* Numeric overlay */}
      <span
        style={{
          position: "absolute",
          left: "50%",
          top: "50%",
          transform: "translate(-50%, -50%)",
          fontSize: 11,
          fontWeight: 600,
          color: "#24292f",
          mixBlendMode: "difference",
          pointerEvents: "none",
        }}
        // eslint-disable-next-line react/no-unknown-property
        data-side={side}
      >
        {value.toFixed(3)}
      </span>
    </div>
  );
}

function VetoBadge({ active, count }: { active: boolean; count: number }) {
  return (
    <div
      style={{
        display: "inline-flex",
        alignItems: "center",
        padding: "4px 10px",
        borderRadius: 12,
        background: active ? "#ffeef0" : "#dafbe1",
        color: active ? "#cf222e" : "#1a7f37",
        fontSize: 12,
        fontWeight: 600,
        border: `1px solid ${active ? "#cf222e" : "#1a7f37"}`,
      }}
    >
      <span
        style={{
          width: 8,
          height: 8,
          borderRadius: "50%",
          marginRight: 6,
          background: active ? "#cf222e" : "#1a7f37",
        }}
      />
      {active ? "GATE ACTIVE" : "GATE OPEN"}
      {count > 0 && (
        <span style={{ marginLeft: 8, fontWeight: 400 }}>
          consec. vetoes: <b>{count}</b>
        </span>
      )}
    </div>
  );
}

export function AgentStatusPanel() {
  const wsState = useAgentStore();
  const [polled, setPolled] = useState<AgentStatus | null>(null);
  const [lastWsUpdateMs, setLastWsUpdateMs] = useState<number>(0);

  // Track when the WebSocket store last changed so we can decide whether
  // the polled fallback is still needed.
  useEffect(() => {
    setLastWsUpdateMs(Date.now());
  }, [wsState.currentAction, wsState.uncertainty, wsState.vetoed]);

  // Polled fallback (also serves as the initial-load fetch).
  useEffect(() => {
    let cancelled = false;
    const tick = async () => {
      try {
        const status = await agentApi.getStatus();
        if (!cancelled) setPolled(status);
      } catch {
        // Network errors are surfaced via the global axios interceptor;
        // we silently keep the previous value here so the panel
        // doesn't flicker on a transient blip.
      }
    };
    tick();
    const id = setInterval(tick, POLL_INTERVAL_MS);
    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, []);

  // Effective values: prefer WS if recent, else polled, else zeros.
  const effective = useMemo(() => {
    const wsAgeMs = Date.now() - lastWsUpdateMs;
    const wsFresh = lastWsUpdateMs > 0 && wsAgeMs < STALE_AFTER_MS;
    if (wsFresh) {
      return {
        currentAction: wsState.currentAction,
        uncertainty: wsState.uncertainty,
        vetoed: wsState.vetoed,
        consecutiveVetoes: 0,
        lastUpdate: new Date(lastWsUpdateMs).toISOString(),
        source: "websocket" as const,
      };
    }
    if (polled) {
      return {
        currentAction: polled.current_action,
        uncertainty: polled.uncertainty,
        vetoed: polled.gate_active,
        consecutiveVetoes: polled.consecutive_vetoes,
        lastUpdate: polled.last_update,
        source: "polling" as const,
      };
    }
    return {
      currentAction: 0,
      uncertainty: 0,
      vetoed: false,
      consecutiveVetoes: 0,
      lastUpdate: "—",
      source: "none" as const,
    };
  }, [wsState, polled, lastWsUpdateMs]);

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
      <header
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "baseline",
          marginBottom: 12,
        }}
      >
        <h2 style={{ margin: 0, fontSize: 16, fontWeight: 600 }}>Agent Status</h2>
        <small style={{ color: "#586069", fontFamily: "monospace" }}>
          last update: {formatTimestamp(effective.lastUpdate)} · src: {effective.source}
        </small>
      </header>

      <div style={{ display: "grid", gridTemplateColumns: "1fr 220px", gap: 24 }}>
        <div>
          <label style={{ fontSize: 12, color: "#586069" }}>Direction (target portfolio fraction)</label>
          <ActionDirectionBar value={effective.currentAction} />
          <div style={{ marginTop: 12 }}>
            <VetoBadge active={effective.vetoed} count={effective.consecutiveVetoes} />
          </div>
        </div>
        <div style={{ textAlign: "center" }}>
          <UncertaintyGauge value={effective.uncertainty} />
        </div>
      </div>
    </section>
  );
}

function formatTimestamp(iso: string): string {
  if (iso === "—") return iso;
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  return d.toLocaleTimeString();
}
