// frontend/src/components/panels/DivergenceTimelinePanel.tsx
//
// Vertical timeline of every pivotal divergence in the run. Dark-theme.

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

  return (
    <section className="lx-panel">
      <header style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 10 }}>
        <div className="lx-label">Pivotal Divergences</div>
        <span className="lx-mono lx-dim" style={{ fontSize: 11 }}>{divergences.length}</span>
      </header>

      {loading && <div className="lx-dim" style={{ fontSize: 12 }}>Loading divergences…</div>}
      {error && <div style={{ color: "var(--red)", fontSize: 12 }}>Error: {error}</div>}
      {!loading && !error && divergences.length === 0 && (
        <div className="lx-dim" style={{ fontSize: 12 }}>No pivotal divergences detected.</div>
      )}

      {divergences.length > 0 && (
        <ol style={{ margin: 0, padding: 0, listStyle: "none", maxHeight: 320, overflowY: "auto" }}>
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
                  padding: "10px 12px",
                  borderRadius: 6,
                  borderLeft: `2px solid ${d.sharpe_delta > 0 ? "var(--green)" : "var(--red)"}`,
                  background: isHovered ? "var(--bg-row-hover)" : "transparent",
                  marginBottom: 6,
                }}
              >
                <div
                  className="lx-mono"
                  style={{ fontSize: 11, color: "var(--text-secondary)", display: "flex", gap: 12 }}
                >
                  <span>{new Date(d.sim_timestamp).toISOString().slice(0, 19)}</span>
                  <span>step <strong style={{ color: "var(--text-primary)" }}>{d.step_index}</strong></span>
                </div>
                <div className="lx-mono" style={{ fontSize: 12, marginTop: 4 }}>
                  T{d.best_trajectory_id} <span className="lx-dim">vs</span> T{d.worst_trajectory_id}{" "}
                  <span className="lx-dim">· L2=</span>{d.action_l2_distance.toFixed(2)}{" "}
                  <span className="lx-dim">· Δ Sharpe</span>{" "}
                  <span style={{ color: d.sharpe_delta > 0 ? "var(--green)" : "var(--red)", fontWeight: 600 }}>
                    {d.sharpe_delta > 0 ? "+" : ""}{d.sharpe_delta.toFixed(2)}
                  </span>
                </div>
                {isHovered && exp && (
                  <pre
                    style={{
                      margin: "8px 0 0 0",
                      background: "var(--bg-panel-2)",
                      border: "1px solid var(--border-soft)",
                      padding: 8,
                      borderRadius: 4,
                      fontSize: 11,
                      whiteSpace: "pre-wrap",
                      color: "var(--text-secondary)",
                    }}
                  >
                    {exp.text}
                  </pre>
                )}
              </li>
            );
          })}
        </ol>
      )}
    </section>
  );
}
