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

import { useEffect, useState } from "react";
import { cancelArenaRun, getArenaRun, getDecisions, getRunSummary, startArenaRun } from "../api/arena";
import { ArenaTrajectoriesPanel } from "../components/panels/ArenaTrajectoriesPanel";
import { CounterfactualPairsPanel } from "../components/panels/CounterfactualPairsPanel";
import { DivergenceTimelinePanel } from "../components/panels/DivergenceTimelinePanel";
import { useArenaStore } from "../store/arenaSlice";
import type { ArenaRunStatus, DecisionRecord, RunSummary } from "../types/arena.types";

const SUMMARY_POLL_INTERVAL_MS = 3_000;
const RUN_POLL_INTERVAL_MS = 2_000;
const PALETTE = [
  "var(--accent-bright)",
  "var(--accent)",
  "var(--purple)",
  "rgba(148,163,184,0.7)",
  "var(--green)",
  "var(--amber)",
];

interface LeaderboardEntry {
  rank: number;
  name: string;
  reward: number;
  ddPct: number;
  winPct: number;
  color: string;
}

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
  const [tick, setTick] = useState(0);
  const [wallElapsed, setWallElapsed] = useState(0);
  const [simTime, setSimTime] = useState<Date | null>(null);
  const [runStatus, setRunStatus] = useState<ArenaRunStatus | null>(null);
  const [leaderboard, setLeaderboard] = useState<LeaderboardEntry[]>([]);

  useEffect(() => {
    if (!playing) return;
    const id = setInterval(() => setWallElapsed((e) => e + 1), 1000);
    return () => clearInterval(id);
  }, [playing]);

  useEffect(() => {
    if (!activeRunId) {
      setRunStatus(null);
      setLeaderboard([]);
      setTick(0);
      setSimTime(null);
      return;
    }
    let cancelled = false;
    let timer: number | undefined;

    const load = async () => {
      try {
        const [run, decisions] = await Promise.all([
          getArenaRun(activeRunId),
          getDecisions(activeRunId, undefined, { limit: 10_000 }),
        ]);
        if (cancelled) return;
        setRunStatus(run.status);
        setLeaderboard(buildLeaderboard(decisions));
        if (decisions.length > 0) {
          const last = decisions.reduce((a, b) => (a.step_index > b.step_index ? a : b));
          setTick(last.step_index);
          setSimTime(new Date(last.sim_timestamp));
        }
        if (["COMPLETED", "FAILED", "CANCELLED"].includes(run.status)) {
          setPlaying(false);
          return;
        }
      } catch (err) {
        if (!cancelled) console.warn("[arena] run poll failed", err);
      }
      if (!cancelled) timer = window.setTimeout(() => void load(), RUN_POLL_INTERVAL_MS);
    };

    void load();
    return () => {
      cancelled = true;
      if (timer !== undefined) window.clearTimeout(timer);
    };
  }, [activeRunId]);

  useEffect(() => {
    if (!activeRunId) {
      setSummary(null);
      return;
    }
    let cancelled = false;
    let timer: number | undefined;

    const load = async () => {
      try {
        const s = await getRunSummary(activeRunId);
        if (!cancelled) setSummary(s);
      } catch {
        if (!cancelled) {
          timer = window.setTimeout(() => void load(), SUMMARY_POLL_INTERVAL_MS);
        }
      }
    };

    void load();
    return () => {
      cancelled = true;
      if (timer !== undefined) window.clearTimeout(timer);
    };
  }, [activeRunId]);

  async function handleStart() {
    setStartError(null);
    setStarting(true);
    setPlaying(true);
    setTick(0);
    setWallElapsed(0);
    setSimTime(null);
    setRunStatus(null);
    setLeaderboard([]);
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
        <Leaderboard rows={leaderboard} status={runStatus} />
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
  simTime: Date | null;
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
        <Stat label="Sim time" value={simTime ? `${simTime.toISOString().slice(0, 10)} ${fmtClock(simTime)}` : "—"} />
        <Stat label="Tick" value={tick > 0 ? `#${tick.toLocaleString()}` : "—"} />
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
            recorded trajectory rewards
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

function Leaderboard({ rows, status }: { rows: LeaderboardEntry[]; status: ArenaRunStatus | null }) {
  return (
    <section className="lx-panel" style={{ display: "flex", flexDirection: "column", gap: 10 }}>
      <header style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <span className="lx-label">Leaderboard</span>
        <span className="lx-mono lx-dim" style={{ fontSize: 11 }}>{status ?? "—"}</span>
      </header>
      {rows.length === 0 ? (
        <div className="lx-dim" style={{ padding: 18, textAlign: "center", fontSize: 12 }}>
          No trajectory decisions recorded yet.
        </div>
      ) : (
        <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
          {rows.map((row) => (
            <LeaderboardRow key={row.name} row={row} />
          ))}
        </div>
      )}
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
            {(row.winPct * 100).toFixed(1)}% positive steps
          </div>
        </div>
        <div style={{ textAlign: "right" }}>
          <div className="lx-mono" style={{ fontSize: 14, fontWeight: 600 }}>
            {row.reward >= 0 ? "+" : ""}{row.reward.toFixed(2)}
          </div>
          <div className="lx-mono" style={{ fontSize: 10, color: row.reward >= 0 ? "var(--green)" : "var(--red)" }}>
            max drawdown {(row.ddPct * 100).toFixed(1)}%
          </div>
        </div>
      </div>
    </div>
  );
}

function Legend() {
  const items = PALETTE.slice(0, 4).map((color, i) => ({ color, label: `T${i}` }));
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

function buildLeaderboard(decisions: DecisionRecord[]): LeaderboardEntry[] {
  const byTrajectory = new Map<number, DecisionRecord[]>();
  for (const decision of decisions) {
    const bucket = byTrajectory.get(decision.trajectory_id) ?? [];
    bucket.push(decision);
    byTrajectory.set(decision.trajectory_id, bucket);
  }

  const rows = Array.from(byTrajectory.entries()).map(([trajectoryId, records]) => {
    const sorted = [...records].sort((a, b) => a.step_index - b.step_index);
    let cumulative = 0;
    let peak = 0;
    let maxDrawdown = 0;
    let positive = 0;
    for (const record of sorted) {
      const reward = record.realized_reward ?? 0;
      cumulative += reward;
      peak = Math.max(peak, cumulative);
      maxDrawdown = Math.max(maxDrawdown, peak - cumulative);
      if (reward > 0) positive += 1;
    }
    return {
      rank: 0,
      name: `Trajectory T${trajectoryId}`,
      reward: cumulative,
      ddPct: peak > 0 ? maxDrawdown / peak : 0,
      winPct: sorted.length > 0 ? positive / sorted.length : 0,
      color: PALETTE[trajectoryId % PALETTE.length],
    };
  });

  return rows
    .sort((a, b) => b.reward - a.reward)
    .map((row, index) => ({ ...row, rank: index + 1 }));
}
