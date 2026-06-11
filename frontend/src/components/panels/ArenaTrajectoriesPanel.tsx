// frontend/src/components/panels/ArenaTrajectoriesPanel.tsx
//
// Overlaid equity curves — one Line per trajectory — with optional
// red-dot markers at the x-coords of pivotal divergences. The currently
// selected trajectory (from the Zustand store) is rendered at a thicker
// stroke; everything else is rendered thin.

import { useEffect, useMemo, useState } from "react";
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ReferenceDot,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { getDecisions, getDivergences } from "../../api/arena";
import type { DecisionRecord, DivergencePoint } from "../../types/arena.types";
import { useArenaStore } from "../../store/arenaSlice";

const POLL_INTERVAL_MS = 2_000;
const PALETTE = [
  "#4a90d9",
  "#50c878",
  "#ffd700",
  "#ff6b6b",
  "#7b68ee",
  "#ff8c42",
  "#00d4aa",
  "#c44dd9",
];

interface Props {
  runId: string;
  divergenceMarkers?: boolean;
  onDivergenceClick?: (d: DivergencePoint) => void;
}

interface ChartRow {
  step_index: number;
  [trajectoryKey: string]: number;
}

export function ArenaTrajectoriesPanel({
  runId,
  divergenceMarkers = true,
  onDivergenceClick,
}: Props) {
  const [decisions, setDecisions] = useState<DecisionRecord[]>([]);
  const [divergences, setDivergences] = useState<DivergencePoint[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const selectedTrajectoryId = useArenaStore((s) => s.selectedTrajectoryId);
  const setSelectedTrajectoryId = useArenaStore((s) => s.setSelectedTrajectoryId);

  useEffect(() => {
    let cancelled = false;
    let timer: number | undefined;

    const load = async (initial = false) => {
      if (initial) setLoading(true);
      try {
        const [recs, divs] = await Promise.all([
          getDecisions(runId, undefined, { limit: 10_000 }),
          getDivergences(runId, { limit: 1000 }),
        ]);
        if (cancelled) return;
        setDecisions(recs);
        setDivergences(divs);
        setError(null);
      } catch (err: unknown) {
        if (!cancelled) setError(err instanceof Error ? err.message : String(err));
      } finally {
        if (!cancelled) setLoading(false);
      }
      if (!cancelled) timer = window.setTimeout(() => void load(false), POLL_INTERVAL_MS);
    };

    void load(true);
    return () => {
      cancelled = true;
      if (timer !== undefined) window.clearTimeout(timer);
    };
  }, [runId]);

  const { rows, trajectoryIds } = useMemo(() => {
    const byStep = new Map<number, ChartRow>();
    const traj = new Set<number>();
    // Equity proxy: cumulative realized_reward per trajectory.
    const cumulative = new Map<number, number>();
    const sorted = [...decisions].sort((a, b) =>
      a.step_index === b.step_index
        ? a.trajectory_id - b.trajectory_id
        : a.step_index - b.step_index,
    );
    for (const rec of sorted) {
      traj.add(rec.trajectory_id);
      const prior = cumulative.get(rec.trajectory_id) ?? 0;
      const next = prior + (rec.realized_reward ?? 0);
      cumulative.set(rec.trajectory_id, next);
      const key = `T${rec.trajectory_id}`;
      const row = byStep.get(rec.step_index) ?? { step_index: rec.step_index };
      row[key] = next;
      byStep.set(rec.step_index, row);
    }
    return {
      rows: Array.from(byStep.values()).sort((a, b) => a.step_index - b.step_index),
      trajectoryIds: Array.from(traj).sort((a, b) => a - b),
    };
  }, [decisions]);

  if (loading) return <div className="lx-dim" style={{ padding: 12 }}>Loading arena trajectories…</div>;
  if (error) return <div style={{ color: "var(--red)", padding: 12 }}>Error: {error}</div>;
  if (rows.length === 0) {
    return <div className="lx-dim" style={{ padding: 12 }}>No decisions recorded for this run yet.</div>;
  }

  return (
    <div>
      <div style={{ display: "flex", gap: 6, marginBottom: 8, flexWrap: "wrap" }}>
        {trajectoryIds.map((tid) => {
          const active = selectedTrajectoryId === tid;
          const color = PALETTE[tid % PALETTE.length];
          return (
            <button
              key={tid}
              onClick={() =>
                setSelectedTrajectoryId(selectedTrajectoryId === tid ? null : tid)
              }
              className={`lx-btn ${active ? "active" : "ghost"}`}
              style={{
                borderColor: active ? color : "var(--border)",
                color: active ? "#fff" : color,
                background: active ? color : "transparent",
                padding: "3px 10px",
                fontFamily: "var(--font-mono)",
                fontSize: 11,
              }}
            >
              T{tid}
            </button>
          );
        })}
      </div>
      <ResponsiveContainer width="100%" height={320}>
        <LineChart data={rows} margin={{ top: 5, right: 16, bottom: 24, left: 16 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="rgba(148,163,184,0.10)" />
          <XAxis
            dataKey="step_index"
            stroke="rgba(148,163,184,0.5)"
            tick={{ fontSize: 10, fill: "rgba(148,163,184,0.7)", fontFamily: "var(--font-mono)" }}
            tickLine={false}
            axisLine={false}
          />
          <YAxis
            stroke="rgba(148,163,184,0.5)"
            tick={{ fontSize: 10, fill: "rgba(148,163,184,0.7)", fontFamily: "var(--font-mono)" }}
            tickLine={false}
            axisLine={false}
          />
          <Tooltip
            contentStyle={{
              background: "#0b1220",
              border: "1px solid rgba(148,163,184,0.2)",
              borderRadius: 6,
              fontSize: 11,
              color: "#e2e8f0",
            }}
          />
          <Legend verticalAlign="top" height={24} wrapperStyle={{ fontSize: 11, color: "var(--text-secondary)" }} />
          {trajectoryIds.map((tid) => (
            <Line
              key={tid}
              type="monotone"
              dataKey={`T${tid}`}
              stroke={PALETTE[tid % PALETTE.length]}
              dot={false}
              strokeWidth={selectedTrajectoryId === tid ? 2.5 : 1.2}
              isAnimationActive={false}
            />
          ))}
          {divergenceMarkers &&
            divergences.map((d) => (
              <ReferenceDot
                key={`d-${d.step_index}-${d.best_trajectory_id}-${d.worst_trajectory_id}`}
                x={d.step_index}
                y={0}
                r={4}
                fill="var(--red)"
                stroke="var(--red)"
                isFront
                onClick={() => onDivergenceClick?.(d)}
                style={{ cursor: onDivergenceClick ? "pointer" : "default" }}
              />
            ))}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
