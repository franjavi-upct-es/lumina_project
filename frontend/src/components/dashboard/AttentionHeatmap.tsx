// frontend/src/components/dashboard/AttentionHeatmap.tsx
//
// Cross-Modal Attention heatmap (3×3 — one row/column per modality).
//
// Reads a square weight matrix from props. Rows are "query" modalities
// (which modality is asking) and columns are "key" modalities (which
// modality is being attended to). A weight of 1.0 means the row
// completely attends to the column; 0.0 means it ignores the column.
// The diagonal is typically dominant after training because each
// modality retains its own information.
//
// Visual encoding
// ---------------
//   * Per-cell background colour interpolated from a light grey
//     (low weight) through pale blue to saturated indigo (high weight).
//   * Numeric weight rendered inside the cell when the cell is large
//     enough to be readable (always true at the default size).
//   * Row + column labels with the modality name and its dimension
//     (e.g. "Price (128)") so operators can see at a glance which
//     modality contributes more or less than its capacity suggests.
//   * Hover state shows a small tooltip with the exact weight to 4 d.p.
//
// Robustness
// ----------
// The component accepts a `weights` prop that may be `undefined` (no
// data yet) or non-square. In both cases it shows a labelled empty
// state rather than throwing — the dashboard remains usable even when
// the fusion service has not yet produced an attention map.

import { useMemo, useState } from "react";

const MODALITIES = [
  { name: "Price", dim: 128, color: "#0969da" },
  { name: "Semantic", dim: 64, color: "#a475f9" },
  { name: "Graph", dim: 32, color: "#1a7f37" },
] as const;

const CELL_PX = 72;
const LABEL_PX = 80;

// Map weight in [0, 1] to an RGBA background. We use HSL because it
// gives a smoother perceptual ramp than RGB linear interpolation.
function weightToColor(weight: number): string {
  const w = Math.max(0, Math.min(1, weight));
  // Hue 220 = indigo; saturation 70%; lightness 95% (light) → 35% (dark).
  const lightness = 95 - w * 60;
  return `hsl(220, 70%, ${lightness}%)`;
}

function isSquareMatrix(arr: unknown): arr is number[][] {
  if (!Array.isArray(arr) || arr.length === 0) return false;
  const n = arr.length;
  return arr.every((row) => Array.isArray(row) && row.length === n);
}

interface AttentionHeatmapProps {
  weights?: number[][];
  /** Label used in the panel header. Defaults to "Cross-Modal Attention". */
  title?: string;
}

export function AttentionHeatmap({ weights, title = "Cross-Modal Attention" }: AttentionHeatmapProps) {
  const [hovered, setHovered] = useState<{ i: number; j: number; v: number } | null>(null);

  // Validate input. If the matrix is not square or not 3×3 we render
  // an explicit empty state with diagnostics.
  const valid = useMemo(() => isSquareMatrix(weights) && weights!.length === MODALITIES.length, [weights]);

  if (!valid) {
    return (
      <section style={panelStyle}>
        <header style={headerStyle}>
          <h2 style={{ margin: 0, fontSize: 16, fontWeight: 600 }}>{title}</h2>
        </header>
        <div style={{ padding: 12, color: "#8c959f", fontSize: 13, textAlign: "center" }}>
          {weights === undefined
            ? "Waiting for attention weights from the fusion service…"
            : "Invalid attention matrix shape (expected 3 × 3)."}
        </div>
      </section>
    );
  }

  const w = weights!;

  return (
    <section style={panelStyle}>
      <header style={{ ...headerStyle, justifyContent: "space-between" }}>
        <h2 style={{ margin: 0, fontSize: 16, fontWeight: 600 }}>{title}</h2>
        {hovered && (
          <small style={{ color: "#586069", fontFamily: "monospace" }}>
            {MODALITIES[hovered.i].name} → {MODALITIES[hovered.j].name}: <b>{hovered.v.toFixed(4)}</b>
          </small>
        )}
      </header>

      <div style={{ display: "inline-block" }}>
        {/* Top axis: key-modality labels */}
        <div style={{ display: "flex" }}>
          <div style={{ width: LABEL_PX }} />
          {MODALITIES.map((m) => (
            <div
              key={m.name}
              style={{
                width: CELL_PX,
                textAlign: "center",
                fontSize: 11,
                fontWeight: 600,
                color: m.color,
                padding: "4px 0",
              }}
            >
              {m.name}
              <div style={{ fontSize: 10, color: "#8c959f", fontWeight: 400 }}>{m.dim}-d</div>
            </div>
          ))}
        </div>

        {/* Rows */}
        {w.map((row, i) => (
          <div key={i} style={{ display: "flex" }}>
            <div
              style={{
                width: LABEL_PX,
                display: "flex",
                alignItems: "center",
                justifyContent: "flex-end",
                paddingRight: 8,
                fontSize: 11,
                fontWeight: 600,
                color: MODALITIES[i].color,
              }}
            >
              {MODALITIES[i].name}
            </div>
            {row.map((v, j) => {
              const isHover = hovered?.i === i && hovered?.j === j;
              return (
                <div
                  key={j}
                  onMouseEnter={() => setHovered({ i, j, v })}
                  onMouseLeave={() => setHovered(null)}
                  style={{
                    width: CELL_PX,
                    height: CELL_PX,
                    background: weightToColor(v),
                    border: isHover ? "2px solid #24292f" : "1px solid #eaecef",
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                    fontSize: 14,
                    fontWeight: 600,
                    fontFamily: "monospace",
                    color: v > 0.5 ? "#ffffff" : "#24292f",
                    transition: "background 200ms ease-out",
                    cursor: "default",
                  }}
                >
                  {v.toFixed(2)}
                </div>
              );
            })}
          </div>
        ))}
      </div>

      <small style={{ color: "#8c959f", fontSize: 11, marginTop: 8, display: "block" }}>
        Rows: query modality &nbsp;·&nbsp; Columns: attended modality &nbsp;·&nbsp; Values in [0, 1]
      </small>
    </section>
  );
}

const panelStyle: React.CSSProperties = {
  border: "1px solid #d0d7de",
  borderRadius: 6,
  padding: 16,
  marginBottom: 16,
  background: "#ffffff",
};

const headerStyle: React.CSSProperties = {
  marginBottom: 12,
  display: "flex",
  alignItems: "baseline",
};
