// frontend/src/components/common/AppShell.tsx
//
// Top-level application chrome: brand mark, primary tab bar and the
// always-on status pill strip (socket, market, tick, run, regime).
// Each page renders inside <main className="lx-page">.

import { useEffect, useState, type ReactNode } from "react";
import { useAgentStream } from "../../hooks/useAgentStream";
import { usePerceptionHealth } from "../../hooks/usePerceptionHealth";
import { BackendStatusBanner } from "./BackendStatusBanner";

type View = "dashboard" | "backtest" | "arena" | "settings";

const TABS: Array<{ id: View; label: string; badgeFor?: "arena" }> = [
  { id: "dashboard", label: "Dashboard" },
  { id: "backtest", label: "Backtest" },
  { id: "arena", label: "Arena", badgeFor: "arena" },
  { id: "settings", label: "Settings" },
];

function nowLocalTime(): string {
  const d = new Date();
  const hh = String(d.getHours()).padStart(2, "0");
  const mm = String(d.getMinutes()).padStart(2, "0");
  const ss = String(d.getSeconds()).padStart(2, "0");
  return `${hh}:${mm}:${ss}`;
}

function useTicker(intervalMs: number): string {
  const [now, setNow] = useState(nowLocalTime());
  useEffect(() => {
    const id = setInterval(() => setNow(nowLocalTime()), intervalMs);
    return () => clearInterval(id);
  }, [intervalMs]);
  return now;
}

interface AppShellProps {
  active: View;
  children: ReactNode;
  /** Optional arena queue length to drive the tab badge. */
  arenaBadge?: number;
}

export function AppShell({ active, children, arenaBadge }: AppShellProps) {
  const { connected } = useAgentStream();
  const health = usePerceptionHealth();
  const clock = useTicker(1000);

  const socketStatus = connected ? "CONNECTED" : "OFFLINE";
  const marketStatus = health.healthy ? "OPEN" : health.status ?? "—";
  const regimeStatus = health.healthy ? "NORMAL" : "ALERT";

  return (
    <div style={{ minHeight: "100vh", display: "flex", flexDirection: "column" }}>
      <header
        style={{
          display: "flex",
          alignItems: "center",
          gap: 0,
          padding: "0 20px",
          borderBottom: "1px solid var(--border)",
          background: "var(--bg-app)",
          position: "sticky",
          top: 0,
          zIndex: 5,
        }}
      >
        <BrandMark />

        <span className={`lx-pill ${connected ? "ok" : "warn"}`} style={{ marginLeft: 14 }}>
          <span className="lx-dot" style={{ background: "var(--accent)" }} />
          {connected ? "LIVE" : "WAITING"}
        </span>

        <nav style={{ display: "flex", marginLeft: 20 }}>
          {TABS.map((tab) => (
            <a
              key={tab.id}
              href={`#/${tab.id}`}
              className={`lx-tab${active === tab.id ? " active" : ""}`}
              style={{ textDecoration: "none" }}
            >
              {tab.label}
              {tab.badgeFor === "arena" && arenaBadge != null && arenaBadge > 0 && (
                <span className="lx-tab-badge">{arenaBadge}</span>
              )}
            </a>
          ))}
        </nav>

        <div style={{ flex: 1 }} />

        <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
          <span className={`lx-pill ${connected ? "ok" : "bad"}`} title="Live websocket health">
            <span className="lx-dot" />
            SOCKET · {socketStatus}
          </span>
          <span className={`lx-pill ${health.healthy ? "ok" : "warn"}`} title="Backend aggregate health">
            <span className="lx-dot" />
            MARKET {marketStatus}
          </span>
          <span className="lx-pill">
            TICK {clock}
          </span>
          <span className="lx-pill">
            RUN —
          </span>
          <span className={`lx-pill ${health.healthy ? "info" : "warn"}`}>
            <span className="lx-dot" />
            {regimeStatus}
          </span>
        </div>
      </header>

      <BackendStatusBanner />

      <main style={{ flex: 1, padding: "20px", overflow: "auto" }}>{children}</main>
    </div>
  );
}

function BrandMark() {
  return (
    <a href="#/dashboard" style={{ display: "flex", alignItems: "center", gap: 10, textDecoration: "none" }}>
      <div
        aria-hidden
        style={{
          width: 32,
          height: 32,
          borderRadius: 8,
          background:
            "linear-gradient(135deg, rgba(74,144,217,0.45), rgba(74,144,217,0.10))",
          border: "1px solid var(--accent)",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          color: "var(--accent-bright)",
          fontWeight: 700,
          fontFamily: "var(--font-mono)",
          letterSpacing: "0.04em",
          fontSize: 16,
        }}
      >
        L
      </div>
      <div style={{ display: "flex", flexDirection: "column", lineHeight: 1 }}>
        <span
          style={{
            color: "var(--text-primary)",
            fontWeight: 600,
            letterSpacing: "0.08em",
            fontSize: 14,
          }}
        >
          LUMINA
        </span>
        <span
          style={{
            color: "var(--text-secondary)",
            fontSize: 9,
            letterSpacing: "0.18em",
            marginTop: 3,
          }}
        >
          MULTI-MODAL · v3.2.1
        </span>
      </div>
    </a>
  );
}
