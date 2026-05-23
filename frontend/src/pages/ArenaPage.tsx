// frontend/src/pages/ArenaPage.tsx
//
// Spartan Arena view. Layout:
//
//   ┌─ Transport row (Start / Pause / Reset · Speed) ────────────────┐
//   │ Sim time · Tick · Wall                                          │
//   ├──────────────────────────────────────────────────────────────────┤
//   │ Equity Race chart (overlaid trajectories)  │ Leaderboard         │
//   ├──────────────────────────────────────────────────────────────────┤
//   │ Divergence timeline       │ Counterfactual pairs                 │
//   ├──────────────────────────────────────────────────────────────────┤
//   │ Run summary narrative                                           │
//   └──────────────────────────────────────────────────────────────────┘

import { useEffect, useMemo, useState } from "react";
import { cancelArenaRun, getRunSummary, startArenaRun } from "../api/arena";
import { ArenaTrajectoriesPanel } from "../components/panels/ArenaTrajectoriesPanel";
import { CounterfactualPairsPanel } from "../components/panels/CounterfactualPairsPanel";
import { DivergenceTimelinePanel } from "../components/panels/DivergenceTimelinePanel";
import { useArenaStore } from "../store/arenaSlice";
import type { RunSummary } from "../types/arena.types";

interface LeaderboardEntry {
  rank: number;
  name: string;
  equity: number;
  retPct: number;
  ddPct: number;
  winPct: number;
  color: string;
  isLumina?: boolean;
}

const DEMO_LEADERBOARD: LeaderboardEntry[] = [
  { rank: 1, name: "Lumina v3.2",  equity: 124_700, retPct: 0.247, ddPct: -0.034, winPct: 0.582, color: "var(--accent-bright)", isLumina: true },
  { rank: 2, name: "Lumina v3.1",  equity: 118_400, retPct: 0.184, ddPct: -0.048, winPct: 0.541, color: "var(--accent)",          isLumina: true },
  { rank: 3, name: "Baseline (BL-12)", equity: 112_800, retPct: 0.128, ddPct: -0.062, winPct: 0.510, color: "var(--purple)" },
  { rank: 4, name: "Random walk",  equity: 101_200, retPct: 0.012, ddPct: -0.081, winPct: 0.494, color: "rgba(148,163,184,0.7)" },
];

function fmtTime(secs: number): string {
  const m = Math.floor(secs / 60);
  const s = secs % 60;
  return `${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
}

function fmtClock(date: Date): string {
  return `${String(date.getHours()).padStart(2, "0")}:${String(date.getMinutes()).padStart(2, "0")}`;
}

export function ArenaPage() {
  const activeRunId = useArenaStore((s) => s.activeRunId);
  const setActiveRunId = useArenaStore((s) => s.setActiveRunId);
  const playbackMultiplier = useArenaStore((s) => s.playbackMultiplier);
  const setPlaybackMultiplier = useArenaStore((s) => s.setPlaybackMultiplier);

  const [summary, setSummary] = useState<RunSummary | null>(null);
  const [starting, setStarting] = useState(false);
  const [startError, setStartError] = useState<string | null>(null);
  const [playing, setPlaying] = useState(false);
  const [tick, setTick] = useState(3482);
  const [wallElapsed, setWallElapsed] = useState(261);
  const [simTime, setSimTime] = useState(new Date("2026-04-12T14:32:00Z"));

  // Drive demo clocks so the transport row never reads as frozen.
  useEffect(() => {
    if (!playing) return;
    const id = setInterval(() => {
      setTick((t) => t + playbackMultiplier);
      setWallElapsed((e) => e + 1);
      setSimTime((d) => new Date(d.getTime() + 60_000 * playbackMultiplier));
    }, 1000);
    return () => clearInterval(id);
  }, [playing, playbackMultiplier]);

  useEffect(() => {
    if (!activeRunId) {
      setSummary(null);
      return;
    }
    let cancelled = false;
    getRunSummary(activeRunId).then((s) => {
      if (!cancelled) setSummary(s);
    }).catch(() => {});
    return () => { cancelled = true; };
  }, [activeRunId]);

  async function handleStart() {
    setStartError(null);
    setStarting(true);
    setPlaying(true);
    try {
      const today = new Date();
      const startDate = new Date(today);
      startDate.setUTCDate(today.getUTCDate() - 30);
      const response = await startArenaRun({
        ticker: "AAPL",
        start_date: startDate.toISOString(),
        end_date: today.toISOString(),
        playback_multiplier: playbackMultiplier,
      });
      setActiveRunId(response.run_id);
    } catch (err: unknown) {
      setStartError(err instanceof Error ? err.message : String(err));
    } finally {
      setStarting(false);
    }
  }

  async function handlePause() {
    setPlaying((p) => !p);
  }

  async function handleReset() {
    setPlaying(false);
    if (activeRunId) {
      try { await cancelArenaRun(activeRunId); } catch (e) { console.warn(e); }
    }
    setActiveRunId(null);
    setTick(0);
    setWallElapsed(0);
  }

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
      <TransportBar
        playing={playing}
        starting={starting}
        startError={startError}
        playbackMultiplier={playbackMultiplier}
        setPlaybackMultiplier={setPlaybackMultiplier}
        onStart={handleStart}
        onPause={handlePause}
        onReset={handleReset}
        simTime={simTime}
        tick={tick}
        wallElapsed={wallElapsed}
      />

      <section style={{ display: "grid", gridTemplateColumns: "minmax(0, 2.2fr) minmax(320px, 1fr)", gap: 16 }}>
        <EquityRaceCard runId={activeRunId} />
        <Leaderboard rows={DEMO_LEADERBOARD} />
      </section>

      {activeRunId && (
        <>
          <section style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16 }}>
            <DivergenceTimelinePanel runId={activeRunId} />
            <CounterfactualPairsPanel runId={activeRunId} />
          </section>
          <SummaryBlock summary={summary} />
        </>
      )}
    </div>
  );
}

function TransportBar({
  playing,
  starting,
  startError,
  playbackMultiplier,
  setPlaybackMultiplier,
  onStart,
  onPause,
  onReset,
  simTime,
  tick,
  wallElapsed,
}: {
  playing: boolean;
  starting: boolean;
  startError: string | null;
  playbackMultiplier: 1 | 10 | 100;
  setPlaybackMultiplier: (m: 1 | 10 | 100) => void;
  onStart: () => void;
  onPause: () => void;
  onReset: () => void;
  simTime: Date;
  tick: number;
  wallElapsed: number;
}) {
  return (
    <section className="lx-panel" style={{ display: "flex", alignItems: "center", gap: 16, flexWrap: "wrap" }}>
      <button
        className="lx-btn primary"
        onClick={onStart}
        disabled={starting}
        style={{ padding: "9px 18px", letterSpacing: "0.08em", fontWeight: 600 }}
      >
        ▶ {starting ? "STARTING…" : "START"}
      </button>
      <button className="lx-btn ghost" onClick={onPause}>
        {playing ? "❚❚ PAUSE" : "▶ RESUME"}
      </button>
      <button className="lx-btn danger" onClick={onReset}>RESET</button>

      <span style={{ marginLeft: 12, display: "inline-flex", alignItems: "center", gap: 6 }}>
        <span className="lx-label">Speed</span>
        {[1, 10, 100, 1000].map((m) => {
          const label = m === 1000 ? "MAX" : `${m}×`;
          const value = (m === 1000 ? 100 : m) as 1 | 10 | 100;
          const active = playbackMultiplier === value && (m !== 1000 || playbackMultiplier === 100);
          return (
            <button
              key={label}
              className={`lx-btn ${active ? "active" : "ghost"}`}
              style={{ padding: "5px 10px", minWidth: 48, justifyContent: "center" }}
              onClick={() => setPlaybackMultiplier(value)}
            >
              {label}
            </button>
          );
        })}
      </span>

      <div style={{ flex: 1 }} />

      <div style={{ display: "flex", alignItems: "center", gap: 18 }}>
        <Stat label="Sim time" value={`${simTime.toISOString().slice(0, 10)} ${fmtClock(simTime)}`} />
        <Stat label="Tick" value={`#${tick.toLocaleString()} / 9,600`} />
        <Stat label="Wall" value={fmtTime(wallElapsed)} />
      </div>

      {startError && (
        <div
          style={{
            width: "100%",
            padding: "8px 10px",
            background: "var(--red-soft)",
            color: "var(--red)",
            borderRadius: 6,
            fontSize: 12,
          }}
        >
          {startError}
        </div>
      )}
    </section>
  );
}

function EquityRaceCard({ runId }: { runId: string | null }) {
  return (
    <section className="lx-panel" style={{ display: "flex", flexDirection: "column", gap: 12 }}>
      <header style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline" }}>
        <div>
          <div className="lx-label">Arena · Equity Race</div>
          <div style={{ marginTop: 4, fontSize: 11, color: "var(--text-dim)" }}>
            NOW · tick 3482
          </div>
        </div>
        <Legend />
      </header>
      {runId ? (
        <ArenaTrajectoriesPanel runId={runId} />
      ) : (
        <div
          style={{
            border: "1px dashed var(--border)",
            borderRadius: 8,
            padding: "60px 12px",
            textAlign: "center",
            color: "var(--text-secondary)",
            fontSize: 13,
          }}
        >
          Start a new arena run to populate the race chart.
        </div>
      )}
    </section>
  );
}

function Leaderboard({ rows }: { rows: LeaderboardEntry[] }) {
  return (
    <section className="lx-panel" style={{ display: "flex", flexDirection: "column", gap: 10 }}>
      <header className="lx-label">Leaderboard</header>
      <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
        {rows.map((row) => (
          <LeaderboardRow key={row.name} row={row} />
        ))}
      </div>
    </section>
  );
}

function LeaderboardRow({ row }: { row: LeaderboardEntry }) {
  const isTop = row.rank === 1;
  return (
    <div
      style={{
        padding: 12,
        borderRadius: 8,
        background: isTop ? "var(--accent-soft)" : "var(--bg-panel-2)",
        border: `1px solid ${isTop ? "var(--accent)" : "var(--border-soft)"}`,
        display: "flex",
        flexDirection: "column",
        gap: 6,
      }}
    >
      <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
        <span
          style={{
            width: 22,
            height: 22,
            borderRadius: 6,
            background: "var(--bg-panel)",
            border: "1px solid var(--border)",
            display: "inline-flex",
            alignItems: "center",
            justifyContent: "center",
            fontFamily: "var(--font-mono)",
            fontSize: 11,
            color: "var(--text-secondary)",
          }}
        >
          {row.rank}
        </span>
        <div style={{ flex: 1 }}>
          <div style={{ fontSize: 13, fontWeight: 600 }}>{row.name}</div>
          <div className="lx-mono" style={{ fontSize: 10, color: "var(--text-dim)", letterSpacing: "0.04em" }}>
            LIVE · {(row.winPct * 100).toFixed(1)}% win
          </div>
        </div>
        <div style={{ textAlign: "right" }}>
          <div className="lx-mono" style={{ fontSize: 14, fontWeight: 600 }}>
            ${(row.equity / 1000).toFixed(1)}k
          </div>
          <div className="lx-mono" style={{ fontSize: 10, color: row.retPct >= 0 ? "var(--green)" : "var(--red)" }}>
            {row.retPct >= 0 ? "+" : ""}{(row.retPct * 100).toFixed(1)}% · DD {(row.ddPct * 100).toFixed(1)}%
          </div>
        </div>
      </div>
      <MiniRaceLine color={row.color} variance={row.rank} />
    </div>
  );
}

function MiniRaceLine({ color, variance }: { color: string; variance: number }) {
  const points = useMemo(() => {
    const w = 260;
    const h = 22;
    const pts: string[] = [];
    let y = h / 2;
    for (let i = 0; i < 36; i++) {
      const wiggle = Math.sin((i + variance) / 2) * (1 + variance * 0.4);
      y = h / 2 + wiggle + (variance % 2 === 0 ? -i * 0.18 : -i * 0.10);
      pts.push(`${(i / 35) * w},${Math.max(3, Math.min(h - 3, y))}`);
    }
    return pts.join(" ");
  }, [variance]);
  return (
    <svg width="100%" height={22} viewBox="0 0 260 22" preserveAspectRatio="none">
      <polyline points={points} fill="none" stroke={color} strokeWidth={1.2} />
    </svg>
  );
}

function Legend() {
  const items = [
    { color: "var(--accent-bright)", label: "Lumina v3.2" },
    { color: "var(--accent)",        label: "Lumina v3.1" },
    { color: "var(--purple)",        label: "Baseline (BL-12)" },
    { color: "rgba(148,163,184,0.7)", label: "Random walk" },
  ];
  return (
    <div style={{ display: "flex", gap: 12, fontSize: 11, color: "var(--text-secondary)" }}>
      {items.map((it) => (
        <span key={it.label} style={{ display: "inline-flex", alignItems: "center", gap: 5 }}>
          <span style={{ width: 8, height: 8, borderRadius: "50%", background: it.color }} />
          {it.label}
        </span>
      ))}
    </div>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <span style={{ display: "inline-flex", flexDirection: "column", alignItems: "flex-start" }}>
      <span className="lx-label" style={{ fontSize: 9 }}>{label}</span>
      <span className="lx-mono" style={{ fontSize: 12, fontWeight: 500, marginTop: 2 }}>{value}</span>
    </span>
  );
}

function SummaryBlock({ summary }: { summary: RunSummary | null }) {
  if (!summary) {
    return (
      <section className="lx-panel" style={{ color: "var(--text-secondary)", fontSize: 12, fontStyle: "italic" }}>
        Run summary will appear here when the run finishes.
      </section>
    );
  }
  return (
    <section className="lx-panel">
      <header
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          marginBottom: 10,
        }}
      >
        <div className="lx-label">Run Summary</div>
        <span className="lx-mono lx-dim" style={{ fontSize: 10 }}>
          method · {summary.summary_method}
        </span>
      </header>
      <pre
        style={{
          whiteSpace: "pre-wrap",
          fontSize: 12,
          margin: 0,
          color: "var(--text-secondary)",
          lineHeight: 1.55,
        }}
      >
        {summary.narrative}
      </pre>
    </section>
  );
}
