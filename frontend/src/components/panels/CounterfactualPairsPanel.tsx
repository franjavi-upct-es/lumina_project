// frontend/src/components/panels/CounterfactualPairsPanel.tsx
//
// Sortable table view of every CounterfactualPair produced by the run.
// Clicking "Inspect" opens a modal with the good/bad action vectors
// side by side.

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

  if (loading) return <div style={panelStyle}>Loading counterfactual pairs…</div>;
  if (error) return <div style={{ ...panelStyle, color: "#c44" }}>Error: {error}</div>;
  if (sortedPairs.length === 0) {
    return <div style={panelStyle}>No counterfactual pairs available for this run.</div>;
  }

  return (
    <div style={panelStyle}>
      <h3 style={{ margin: "0 0 8px 0", fontSize: "14px" }}>
        Counterfactual pairs ({sortedPairs.length})
      </h3>
      <div style={{ maxHeight: "320px", overflow: "auto" }}>
        <table style={{ width: "100%", fontSize: "12px", borderCollapse: "collapse" }}>
          <thead>
            <tr style={{ borderBottom: "2px solid #d0d7de", background: "#f6f8fa" }}>
              <Th onClick={() => toggleSort("divergence_step_index")}>Step</Th>
              <Th onClick={() => toggleSort("confidence_score")}>Confidence</Th>
              <Th onClick={() => toggleSort("good_outcome_sharpe")}>Good Sharpe</Th>
              <Th onClick={() => toggleSort("bad_outcome_sharpe")}>Bad Sharpe</Th>
              <Th>Action</Th>
            </tr>
          </thead>
          <tbody>
            {sortedPairs.map((p) => (
              <tr key={p.pair_id} style={{ borderBottom: "1px solid #eef1f4" }}>
                <td style={td}>{p.divergence_step_index}</td>
                <td style={td}>{p.confidence_score.toFixed(2)}</td>
                <td style={{ ...td, color: "#2a8" }}>+{p.good_outcome_sharpe.toFixed(2)}</td>
                <td style={{ ...td, color: "#c44" }}>{p.bad_outcome_sharpe.toFixed(2)}</td>
                <td style={td}>
                  <button onClick={() => setInspecting(p)} style={inspectBtn}>
                    Inspect
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {inspecting && (
        <PairModal pair={inspecting} onClose={() => setInspecting(null)} />
      )}
    </div>
  );
}

function PairModal({ pair, onClose }: { pair: CounterfactualPair; onClose: () => void }) {
  return (
    <div style={modalBackdrop} onClick={onClose}>
      <div style={modalBody} onClick={(e) => e.stopPropagation()}>
        <h3 style={{ marginTop: 0 }}>Pair {pair.pair_id.slice(0, 8)}</h3>
        <div style={{ display: "flex", gap: "16px" }}>
          <ActionColumn
            title={`Good (Sharpe ${pair.good_outcome_sharpe.toFixed(2)})`}
            vector={pair.good_action_vector}
            color="#2a8"
          />
          <ActionColumn
            title={`Bad (Sharpe ${pair.bad_outcome_sharpe.toFixed(2)})`}
            vector={pair.bad_action_vector}
            color="#c44"
          />
        </div>
        <p style={{ fontSize: "12px", marginBottom: 0 }}>
          Confidence score: <strong>{pair.confidence_score.toFixed(2)}</strong>
        </p>
        <button onClick={onClose} style={{ marginTop: "12px" }}>
          Close
        </button>
      </div>
    </div>
  );
}

function ActionColumn({
  title,
  vector,
  color,
}: {
  title: string;
  vector: number[];
  color: string;
}) {
  const labels = ["direction", "urgency", "sizing", "stop"];
  return (
    <div style={{ flex: 1 }}>
      <div style={{ color, fontWeight: 600, marginBottom: "4px" }}>{title}</div>
      <ul style={{ paddingLeft: "16px", margin: 0, fontFamily: "monospace" }}>
        {vector.map((v, i) => (
          <li key={i}>
            {labels[i] ?? `a${i}`}: {v.toFixed(3)}
          </li>
        ))}
      </ul>
    </div>
  );
}

function Th({
  children,
  onClick,
}: {
  children: React.ReactNode;
  onClick?: () => void;
}) {
  return (
    <th
      onClick={onClick}
      style={{
        textAlign: "left",
        padding: "4px 8px",
        cursor: onClick ? "pointer" : "default",
      }}
    >
      {children}
    </th>
  );
}

const td: React.CSSProperties = { padding: "4px 8px" };
const inspectBtn: React.CSSProperties = {
  background: "#0969da",
  color: "white",
  border: "none",
  borderRadius: "4px",
  padding: "2px 8px",
  cursor: "pointer",
  fontSize: "11px",
};
const panelStyle: React.CSSProperties = {
  background: "#fff",
  border: "1px solid #d0d7de",
  borderRadius: "6px",
  padding: "12px",
  boxSizing: "border-box",
};
const modalBackdrop: React.CSSProperties = {
  position: "fixed",
  inset: 0,
  background: "rgba(0, 0, 0, 0.4)",
  display: "flex",
  alignItems: "center",
  justifyContent: "center",
  zIndex: 50,
};
const modalBody: React.CSSProperties = {
  background: "white",
  padding: "24px",
  borderRadius: "8px",
  minWidth: "400px",
  maxWidth: "640px",
  boxShadow: "0 10px 25px rgba(0,0,0,0.2)",
};
