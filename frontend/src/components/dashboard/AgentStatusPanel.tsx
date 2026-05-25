// frontend/src/components/dashboard/AgentStatusPanel.tsx
//
// Compact "Agent · Direction" panel that lives in the right column of
// the dashboard. Shows the gate state and the target portfolio fraction
// with a –1…+1 slider-style indicator and the most-recent predicted
// values when available.

import { useEffect, useMemo, useState } from "react";
import { agentApi } from "../../api/agent";
import { useAgentStore } from "../../store/agentSlice";
import type { AgentStatus } from "../../types/agent.types";

const POLL_INTERVAL_MS = 3000;
const STALE_AFTER_MS = 10_000;

function DirectionBar({ value }: { value: number }) {
  const clamped = Math.max(-1, Math.min(1, value));
  const positionPct = ((clamped + 1) / 2) * 100;
  const color = clamped >= 0 ? "var(--accent-bright)" : "var(--red)";
  return (
    <div style={{ marginTop: 10 }}>
      <div
        style={{
          position: "relative",
          height: 8,
          borderRadius: 4,
          background:
            "linear-gradient(90deg, rgba(239,68,68,0.20), rgba(148,163,184,0.10) 50%, rgba(74,144,217,0.20))",
          border: "1px solid var(--border)",
        }}
      >
        <div
          style={{
            position: "absolute",
            top: "50%",
            left: "50%",
            transform: "translate(-50%, -50%)",
            width: 1,
            height: "100%",
            background: "rgba(148,163,184,0.4)",
          }}
        />
        <div
          style={{
            position: "absolute",
            top: "50%",
            left: `${positionPct}%`,
            transform: "translate(-50%, -50%)",
            width: 14,
            height: 14,
            borderRadius: 3,
            background: color,
            boxShadow: `0 0 12px ${color}`,
            transition: "left 250ms ease-out",
          }}
        />
      </div>
      <div
        style={{
          marginTop: 6,
          display: "flex",
          justifyContent: "space-between",
          fontFamily: "var(--font-mono)",
          fontSize: 10,
          color: "var(--text-muted)",
        }}
      >
        <span>−1.0</span>
        <span style={{ color: "var(--red)" }}>short</span>
        <span>0</span>
        <span style={{ color: "var(--accent-bright)" }}>long</span>
        <span>+1.0</span>
      </div>
    </div>
  );
}

export function AgentStatusPanel() {
  const wsState = useAgentStore();
  const [polled, setPolled] = useState<AgentStatus | null>(null);
  const [lastWsUpdateMs, setLastWsUpdateMs] = useState<number>(0);

  useEffect(() => {
    setLastWsUpdateMs(Date.now());
  }, [wsState.currentAction, wsState.uncertainty, wsState.vetoed]);

  useEffect(() => {
    let cancelled = false;
    const tick = async () => {
      try {
        const status = await agentApi.getStatus();
        if (!cancelled) setPolled(status);
      } catch { /* keep last good */ }
    };
    tick();
    const id = setInterval(tick, POLL_INTERVAL_MS);
    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, []);

  const effective = useMemo(() => {
    const wsAgeMs = Date.now() - lastWsUpdateMs;
    const wsFresh = lastWsUpdateMs > 0 && wsAgeMs < STALE_AFTER_MS;
    if (wsFresh) {
      return {
        currentAction: wsState.currentAction,
        uncertainty: wsState.uncertainty,
        vetoed: wsState.vetoed,
      };
    }
    if (polled) {
      return {
        currentAction: polled.current_action,
        uncertainty: polled.uncertainty,
        vetoed: polled.gate_active,
      };
    }
    return { currentAction: 0, uncertainty: 0, vetoed: false };
  }, [wsState, polled, lastWsUpdateMs]);

  const directionColor = effective.currentAction >= 0 ? "var(--accent-bright)" : "var(--red)";
  const gateLabel = effective.vetoed ? "GATE ACTIVE" : "GATE OPEN";

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
          <div className="lx-label">Agent</div>
          <h2 style={{ margin: "2px 0 0 0", fontSize: 14, fontWeight: 600 }}>Direction</h2>
        </div>
        <span
          className={`lx-pill ${effective.vetoed ? "bad" : "ok"}`}
          style={{ fontWeight: 600 }}
        >
          {gateLabel}
        </span>
      </header>

      <div style={{ marginTop: 12 }}>
        <div className="lx-label">Target portfolio fraction</div>
        <div
          style={{
            display: "flex",
            justifyContent: "flex-end",
            fontFamily: "var(--font-mono)",
            fontSize: 32,
            fontWeight: 600,
            color: directionColor,
            letterSpacing: "-0.02em",
            lineHeight: 1.1,
          }}
        >
          {effective.currentAction >= 0 ? "+" : "−"}
          {Math.abs(effective.currentAction).toFixed(3)}
        </div>
        <DirectionBar value={effective.currentAction} />
      </div>

      <hr className="lx-divider" />

      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          fontSize: 11,
          color: "var(--text-secondary)",
        }}
      >
        <span>σ uncertainty</span>
        <span className="lx-mono" style={{ color: "var(--text-primary)" }}>
          {effective.uncertainty.toFixed(3)}
        </span>
      </div>
    </section>
  );
}
