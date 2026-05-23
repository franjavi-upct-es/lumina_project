// frontend/src/components/panels/CounterfactualPairsPanel.tsx
//
// Sortable table view of every CounterfactualPair produced by the run.
// Clicking "Inspect" opens a modal with the good/bad action vectors
// side by side. Dark-theme styling.

import { useEffect, useMemo, useState } from "react";
import { getCounterfactualPairs } from "../../api/arena";
import type { CounterfactualPair } from "../../types/arena.types";

interface Props {
  runId: string;
}

type SortKey =
  | "confidence_score"
  | "good_outcome_sharpe"
  | "bad_outcome_sharpe"
  | "divergence_step_index";

export function CounterfactualPairsPanel({ runId }: Props) {
  const [pairs, setPairs] = useState<CounterfactualPair[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [sortKey, setSortKey] = useState<SortKey>("confidence_score");
  const [sortDesc, setSortDesc] = useState(true);
  const [inspecting, setInspecting] = useState<CounterfactualPair | null>(null);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    getCounterfactualPairs(runId, { limit: 500 })
      .then((items) => {
        if (cancelled) return;
        setPairs(items);
        setError(null);
      })
      .catch((err) => !cancelled && setError(String(err?.message ?? err)))
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [runId]);

  const sortedPairs = useMemo(() => {
    const copy = [...pairs];
    copy.sort((a, b) => {
      const av = (a[sortKey] as number) ?? 0;
      const bv = (b[sortKey] as number) ?? 0;
      return sortDesc ? bv - av : av - bv;
    });
    return copy;
  }, [pairs, sortKey, sortDesc]);

  function toggleSort(key: SortKey) {
    if (sortKey === key) {
      setSortDesc((d) => !d);
    } else {
      setSortKey(key);
      setSortDesc(true);
    }
  }

  return (
    <section className="lx-panel">
      <header style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 10 }}>
        <div className="lx-label">Counterfactual Pairs</div>
        <span className="lx-mono lx-dim" style={{ fontSize: 11 }}>{sortedPairs.length}</span>
      </header>

      {loading && <div className="lx-dim" style={{ fontSize: 12 }}>Loading counterfactual pairs…</div>}
      {error && <div style={{ color: "var(--red)", fontSize: 12 }}>Error: {error}</div>}
      {!loading && !error && sortedPairs.length === 0 && (
        <div className="lx-dim" style={{ fontSize: 12 }}>No counterfactual pairs for this run.</div>
      )}

      {sortedPairs.length > 0 && (
        <div style={{ maxHeight: 320, overflow: "auto" }}>
          <table className="lx-table">
            <thead>
              <tr>
                <Th onClick={() => toggleSort("divergence_step_index")} active={sortKey === "divergence_step_index"} desc={sortDesc}>Step</Th>
                <Th onClick={() => toggleSort("confidence_score")} active={sortKey === "confidence_score"} desc={sortDesc}>Confidence</Th>
                <Th onClick={() => toggleSort("good_outcome_sharpe")} active={sortKey === "good_outcome_sharpe"} desc={sortDesc}>Good Sh</Th>
                <Th onClick={() => toggleSort("bad_outcome_sharpe")} active={sortKey === "bad_outcome_sharpe"} desc={sortDesc}>Bad Sh</Th>
                <th className="r" />
              </tr>
            </thead>
            <tbody>
              {sortedPairs.map((p) => (
                <tr key={p.pair_id}>
                  <td>{p.divergence_step_index}</td>
                  <td>{p.confidence_score.toFixed(2)}</td>
                  <td style={{ color: "var(--green)" }}>+{p.good_outcome_sharpe.toFixed(2)}</td>
                  <td style={{ color: "var(--red)" }}>{p.bad_outcome_sharpe.toFixed(2)}</td>
                  <td className="r">
                    <button className="lx-btn primary" style={{ padding: "3px 10px", fontSize: 10 }} onClick={() => setInspecting(p)}>
                      INSPECT
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {inspecting && <PairModal pair={inspecting} onClose={() => setInspecting(null)} />}
    </section>
  );
}

function PairModal({ pair, onClose }: { pair: CounterfactualPair; onClose: () => void }) {
  return (
    <div
      onClick={onClose}
      style={{
        position: "fixed",
        inset: 0,
        background: "rgba(2, 6, 18, 0.7)",
        backdropFilter: "blur(2px)",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        zIndex: 50,
      }}
    >
      <div
        className="lx-panel"
        onClick={(e) => e.stopPropagation()}
        style={{ minWidth: 480, maxWidth: 640, boxShadow: "0 14px 40px rgba(0,0,0,0.4)" }}
      >
        <header style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 14 }}>
          <div className="lx-label">Counterfactual Pair · {pair.pair_id.slice(0, 8)}</div>
          <button className="lx-btn ghost" onClick={onClose}>Close ✕</button>
        </header>
        <div style={{ display: "flex", gap: 16 }}>
          <ActionColumn
            title={`Good · Sharpe ${pair.good_outcome_sharpe.toFixed(2)}`}
            vector={pair.good_action_vector}
            color="var(--green)"
          />
          <ActionColumn
            title={`Bad · Sharpe ${pair.bad_outcome_sharpe.toFixed(2)}`}
            vector={pair.bad_action_vector}
            color="var(--red)"
          />
        </div>
        <p style={{ fontSize: 12, marginTop: 12, marginBottom: 0, color: "var(--text-secondary)" }}>
          Confidence score:{" "}
          <strong className="lx-mono" style={{ color: "var(--text-primary)" }}>
            {pair.confidence_score.toFixed(2)}
          </strong>
        </p>
      </div>
    </div>
  );
}

function ActionColumn({ title, vector, color }: { title: string; vector: number[]; color: string }) {
  const labels = ["direction", "urgency", "sizing", "stop"];
  return (
    <div style={{ flex: 1, background: "var(--bg-panel-2)", borderRadius: 8, padding: 12, border: "1px solid var(--border-soft)" }}>
      <div style={{ color, fontWeight: 600, marginBottom: 8, fontSize: 12 }}>{title}</div>
      <ul style={{ padding: 0, margin: 0, listStyle: "none", fontFamily: "var(--font-mono)", fontSize: 12 }}>
        {vector.map((v, i) => (
          <li
            key={i}
            style={{
              display: "flex",
              justifyContent: "space-between",
              padding: "4px 0",
              borderBottom: i < vector.length - 1 ? "1px solid var(--border-soft)" : "none",
            }}
          >
            <span className="lx-dim">{labels[i] ?? `a${i}`}</span>
            <span style={{ color: "var(--text-primary)" }}>{v.toFixed(3)}</span>
          </li>
        ))}
      </ul>
    </div>
  );
}

function Th({
  children,
  onClick,
  active,
  desc,
}: {
  children: React.ReactNode;
  onClick?: () => void;
  active?: boolean;
  desc?: boolean;
}) {
  return (
    <th
      onClick={onClick}
      style={{
        cursor: onClick ? "pointer" : "default",
        color: active ? "var(--accent-bright)" : undefined,
      }}
    >
      {children}
      {active && (
        <span style={{ marginLeft: 4, fontFamily: "var(--font-mono)", fontSize: 10 }}>
          {desc ? "↓" : "↑"}
        </span>
      )}
    </th>
  );
}
