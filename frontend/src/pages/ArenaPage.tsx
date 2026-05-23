// frontend/src/pages/ArenaPage.tsx
//
// Container page for the Spartan Arena. Mirrors the spec layout:
//   row 1: control bar
//   row 2: trajectories chart (full width)
//   row 3: divergence timeline | counterfactual pairs
//   row 4: run summary text block

import { useEffect, useState } from "react";
import { getRunSummary, startArenaRun, cancelArenaRun } from "../api/arena";
import { ArenaTrajectoriesPanel } from "../components/panels/ArenaTrajectoriesPanel";
import { DivergenceTimelinePanel } from "../components/panels/DivergenceTimelinePanel";
import { CounterfactualPairsPanel } from "../components/panels/CounterfactualPairsPanel";
import { useArenaStore } from "../store/arenaSlice";
import type { RunSummary } from "../types/arena.types";

export function ArenaPage() {
  const activeRunId = useArenaStore((s) => s.activeRunId);
  const setActiveRunId = useArenaStore((s) => s.setActiveRunId);
  const playbackMultiplier = useArenaStore((s) => s.playbackMultiplier);
  const setPlaybackMultiplier = useArenaStore((s) => s.setPlaybackMultiplier);
  const [summary, setSummary] = useState<RunSummary | null>(null);
  const [startError, setStartError] = useState<string | null>(null);
  const [starting, setStarting] = useState(false);

  useEffect(() => {
    if (!activeRunId) {
      setSummary(null);
      return;
    }
    let cancelled = false;
    getRunSummary(activeRunId)
      .then((s) => {
        if (!cancelled) setSummary(s);
      })
      .catch(() => {
        // summary may not exist yet (run still in flight) — silently swallow.
      });
    return () => {
      cancelled = true;
    };
  }, [activeRunId]);

  async function handleStart() {
    setStartError(null);
    setStarting(true);
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
      const msg = err instanceof Error ? err.message : String(err);
      setStartError(msg);
    } finally {
      setStarting(false);
    }
  }

  async function handleCancel() {
    if (!activeRunId) return;
    try {
      await cancelArenaRun(activeRunId);
    } catch (err) {
      // Surface but don't block.
      console.error("Cancel failed", err);
    }
  }

  return (
    <div style={{ padding: "16px", display: "grid", gap: "16px" }}>
      <ControlBar
        onStart={handleStart}
        onCancel={handleCancel}
        canCancel={Boolean(activeRunId)}
        playbackMultiplier={playbackMultiplier}
        setPlaybackMultiplier={setPlaybackMultiplier}
        starting={starting}
        startError={startError}
      />

      {activeRunId ? (
        <>
          <ArenaTrajectoriesPanel runId={activeRunId} />
          <div
            style={{
              display: "grid",
              gridTemplateColumns: "1fr 1fr",
              gap: "16px",
            }}
          >
            <DivergenceTimelinePanel runId={activeRunId} />
            <CounterfactualPairsPanel runId={activeRunId} />
          </div>
          <SummaryBlock summary={summary} />
        </>
      ) : (
        <div style={emptyState}>
          Start a new arena run to populate the view.
        </div>
      )}
    </div>
  );
}

function ControlBar({
  onStart,
  onCancel,
  canCancel,
  playbackMultiplier,
  setPlaybackMultiplier,
  starting,
  startError,
}: {
  onStart: () => void;
  onCancel: () => void;
  canCancel: boolean;
  playbackMultiplier: 1 | 10 | 100;
  setPlaybackMultiplier: (m: 1 | 10 | 100) => void;
  starting: boolean;
  startError: string | null;
}) {
  return (
    <div
      style={{
        display: "flex",
        gap: "12px",
        alignItems: "center",
        padding: "8px 12px",
        background: "#fff",
        border: "1px solid #d0d7de",
        borderRadius: "6px",
      }}
    >
      <button onClick={onStart} disabled={starting} style={primaryBtn}>
        {starting ? "Starting…" : "Start"}
      </button>
      <button onClick={onCancel} disabled={!canCancel} style={btn}>
        Cancel
      </button>
      <div style={{ marginLeft: "auto", display: "flex", gap: "4px", alignItems: "center" }}>
        <span style={{ fontSize: "12px", color: "#586069" }}>Speed</span>
        {[1, 10, 100].map((m) => (
          <button
            key={m}
            onClick={() => setPlaybackMultiplier(m as 1 | 10 | 100)}
            style={{
              ...btn,
              borderColor: playbackMultiplier === m ? "#0969da" : "#d0d7de",
              background: playbackMultiplier === m ? "#0969da" : "white",
              color: playbackMultiplier === m ? "white" : "#24292f",
            }}
          >
            {m}x
          </button>
        ))}
      </div>
      {startError && (
        <div style={{ marginLeft: "12px", color: "#c44", fontSize: "12px" }}>
          {startError}
        </div>
      )}
    </div>
  );
}

function SummaryBlock({ summary }: { summary: RunSummary | null }) {
  if (!summary) {
    return (
      <div style={{ ...panelStyle, fontStyle: "italic", color: "#586069" }}>
        Run summary will appear here when the run finishes.
      </div>
    );
  }
  return (
    <div style={panelStyle}>
      <h3 style={{ marginTop: 0 }}>
        Run summary{" "}
        <span style={{ fontSize: "11px", color: "#586069" }}>
          (method: {summary.summary_method})
        </span>
      </h3>
      <pre style={{ whiteSpace: "pre-wrap", fontSize: "13px", margin: 0 }}>
        {summary.narrative}
      </pre>
    </div>
  );
}

const panelStyle: React.CSSProperties = {
  background: "#fff",
  border: "1px solid #d0d7de",
  borderRadius: "6px",
  padding: "12px",
};
const btn: React.CSSProperties = {
  padding: "4px 12px",
  border: "1px solid #d0d7de",
  borderRadius: "4px",
  background: "white",
  cursor: "pointer",
  fontSize: "12px",
};
const primaryBtn: React.CSSProperties = {
  ...btn,
  background: "#0969da",
  color: "white",
  borderColor: "#0969da",
};
const emptyState: React.CSSProperties = {
  background: "#f6f8fa",
  border: "1px dashed #d0d7de",
  borderRadius: "6px",
  padding: "32px",
  textAlign: "center",
  color: "#586069",
};
