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
    setLoading(true);
    Promise.all([
      getDecisions(runId, undefined, { limit: 10_000 }),
      getDivergences(runId, { limit: 1000 }),
    ])
      .then(([recs, divs]) => {
        if (cancelled) return;
        setDecisions(recs);
        setDivergences(divs);
        setError(null);
      })
      .catch((err) => {
        if (!cancelled) setError(String(err?.message ?? err));
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
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

  if (loading) return <div style={panelStyle}>Loading arena trajectories…</div>;
  if (error) return <div style={{ ...panelStyle, color: "#c44" }}>Error: {error}</div>;
  if (rows.length === 0) {
    return <div style={panelStyle}>No decisions recorded for this run yet.</div>;
  }

  return (
    <div style={panelStyle}>
      <div style={{ display: "flex", gap: "8px", marginBottom: "8px", flexWrap: "wrap" }}>
        {trajectoryIds.map((tid) => (
          <button
            key={tid}
            onClick={() =>
              setSelectedTrajectoryId(selectedTrajectoryId === tid ? null : tid)
            }
            style={{
              border: `1px solid ${PALETTE[tid % PALETTE.length]}`,
              background: selectedTrajectoryId === tid ? PALETTE[tid % PALETTE.length] : "white",
              color: selectedTrajectoryId === tid ? "white" : PALETTE[tid % PALETTE.length],
              padding: "2px 8px",
              borderRadius: "12px",
              cursor: "pointer",
              fontSize: "12px",
            }}
          >
            T{tid}
          </button>
        ))}
      </div>
      <ResponsiveContainer width="100%" height={320}>
        <LineChart data={rows} margin={{ top: 5, right: 16, bottom: 24, left: 16 }}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="step_index" label={{ value: "Step", position: "bottom" }} />
          <YAxis label={{ value: "Cumulative reward", angle: -90, position: "insideLeft" }} />
          <Tooltip />
          <Legend verticalAlign="top" height={24} />
          {trajectoryIds.map((tid) => (
            <Line
              key={tid}
              type="monotone"
              dataKey={`T${tid}`}
              stroke={PALETTE[tid % PALETTE.length]}
              dot={false}
              strokeWidth={selectedTrajectoryId === tid ? 3 : 1}
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
                fill="#c44"
                stroke="#c44"
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

const panelStyle: React.CSSProperties = {
  background: "#fff",
  border: "1px solid #d0d7de",
  borderRadius: "6px",
  padding: "12px",
  boxSizing: "border-box",
};
