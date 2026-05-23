// frontend/src/components/panels/DivergenceTimelinePanel.tsx
//
// Vertical timeline of every pivotal divergence in the run. Each row
// summarises the step, the best/worst trajectories, and the Sharpe
// delta. Hovering a row surfaces the best trajectory's full
// StepExplanation.

import { useEffect, useMemo, useState } from "react";
import { getDivergences, getExplanations } from "../../api/arena";
import type {
  DivergencePoint,
  StepExplanation,
} from "../../types/arena.types";

interface Props {
  runId: string;
  onSelect?: (d: DivergencePoint) => void;
}

export function DivergenceTimelinePanel({ runId, onSelect }: Props) {
  const [divergences, setDivergences] = useState<DivergencePoint[]>([]);
  const [explanations, setExplanations] = useState<StepExplanation[]>([]);
  const [hoveredRecordId, setHoveredRecordId] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    Promise.all([
      getDivergences(runId, { limit: 500 }),
      getExplanations(runId, { limit: 5000 }),
    ])
      .then(([divs, exps]) => {
        if (cancelled) return;
        setDivergences(divs);
        setExplanations(exps);
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

  const explanationByRecordId = useMemo(() => {
    const map = new Map<string, StepExplanation>();
    for (const e of explanations) map.set(e.record_id, e);
    return map;
  }, [explanations]);

  if (loading) return <div style={panelStyle}>Loading divergences…</div>;
  if (error) return <div style={{ ...panelStyle, color: "#c44" }}>Error: {error}</div>;
  if (divergences.length === 0) {
    return <div style={panelStyle}>No pivotal divergences detected.</div>;
  }

  return (
    <div style={panelStyle}>
      <h3 style={{ margin: "0 0 8px 0", fontSize: "14px" }}>
        Pivotal divergences ({divergences.length})
      </h3>
      <ol style={{ margin: 0, paddingLeft: "20px", maxHeight: "320px", overflowY: "auto" }}>
        {divergences.map((d) => {
          const key = `${d.step_index}-${d.best_trajectory_id}-${d.worst_trajectory_id}`;
          const exp = explanationByRecordId.get(key);
          const isHovered = hoveredRecordId === key;
          return (
            <li
              key={key}
              onMouseEnter={() => setHoveredRecordId(key)}
              onMouseLeave={() => setHoveredRecordId(null)}
              onClick={() => onSelect?.(d)}
              style={{
                cursor: onSelect ? "pointer" : "default",
                padding: "4px 0",
                borderBottom: "1px solid #eef1f4",
              }}
            >
              <div style={{ fontFamily: "monospace", fontSize: "12px" }}>
                {new Date(d.sim_timestamp).toISOString().slice(0, 19)} · step{" "}
                <strong>{d.step_index}</strong>
              </div>
              <div style={{ fontSize: "12px" }}>
                T{d.best_trajectory_id} vs T{d.worst_trajectory_id} | L2=
                {d.action_l2_distance.toFixed(2)} | Δ Sharpe={" "}
                <span style={{ color: d.sharpe_delta > 0 ? "#2a8" : "#c44" }}>
                  {d.sharpe_delta > 0 ? "+" : ""}
                  {d.sharpe_delta.toFixed(2)}
                </span>
              </div>
              {isHovered && exp && (
                <pre
                  style={{
                    margin: "4px 0 0 0",
                    background: "#f6f8fa",
                    padding: "6px",
                    fontSize: "11px",
                    whiteSpace: "pre-wrap",
                  }}
                >
                  {exp.text}
                </pre>
              )}
            </li>
          );
        })}
      </ol>
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
