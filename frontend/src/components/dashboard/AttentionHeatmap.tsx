// frontend/src/components/dashboard/AttentionHeatmap.tsx
//
// Cross-Modal Attention heatmap (3×3) — dark-theme variant.

import { useMemo, useState } from "react";

const MODALITIES = [
  { name: "Price",    dim: 128, color: "#5ba6f5" },
  { name: "Semantic", dim: 64,  color: "#a475f9" },
  { name: "Graph",    dim: 32,  color: "#22c55e" },
] as const;

const CELL_PX  = 96;
const LABEL_PX = 80;

function isSquareMatrix(arr: unknown): arr is number[][] {
  if (!Array.isArray(arr) || arr.length === 0) return false;
  const n = arr.length;
  return arr.every((row) => Array.isArray(row) && row.length === n);
}

// Diagonal cells get an accent hue (Price/Semantic/Graph). Off-diagonal
// cells use a neutral indigo-blue ramp. Lightness is driven by the
// weight so high-attention cells visibly glow.
function cellColor(weight: number, i: number, j: number): string {
  const w = Math.max(0, Math.min(1, weight));
  if (i === j) {
    const hueByMod = [212, 268, 142];
    const h = hueByMod[i];
    const sat = 65;
    const light = 18 + w * 28;
    return `hsl(${h}, ${sat}%, ${light}%)`;
  }
  const sat = 30 + w * 30;
  const light = 12 + w * 18;
  return `hsl(212, ${sat}%, ${light}%)`;
}

interface AttentionHeatmapProps {
  weights?: number[][];
  title?: string;
}

export function AttentionHeatmap({ weights, title = "Cross-Modal Attention" }: AttentionHeatmapProps) {
  const [hovered, setHovered] = useState<{ i: number; j: number; v: number } | null>(null);

  const valid = useMemo(
    () => isSquareMatrix(weights) && weights!.length === MODALITIES.length,
    [weights],
  );

  return (
    <section className="lx-panel" style={{ height: "100%" }}>
      <header
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "baseline",
          marginBottom: 14,
        }}
      >
        <div>
          <div className="lx-label">Cross-Modal</div>
          <h2 style={{ margin: "2px 0 0 0", fontSize: 14, fontWeight: 600 }}>{title}</h2>
        </div>
        <small className="lx-dim lx-mono" style={{ fontSize: 11 }}>
          τ=0.31 · 8 heads
        </small>
      </header>

      {!valid ? (
        <div className="lx-dim" style={{ padding: 12, textAlign: "center", fontSize: 12 }}>
          {weights === undefined
            ? "Waiting for attention weights from the fusion service…"
            : "Invalid attention matrix shape (expected 3 × 3)."}
        </div>
      ) : (
        <>
          <div style={{ display: "flex", justifyContent: "center" }}>
            <div>
              <div style={{ display: "flex" }}>
                <div style={{ width: LABEL_PX }} />
                {MODALITIES.map((m) => (
                  <div
                    key={m.name}
                    style={{
                      width: CELL_PX,
                      textAlign: "center",
                      fontSize: 10,
                      fontWeight: 600,
                      letterSpacing: "0.12em",
                      color: m.color,
                      padding: "4px 0",
                      textTransform: "uppercase",
                    }}
                  >
                    {m.name}
                    <div className="lx-dim" style={{ fontSize: 9, fontWeight: 400, letterSpacing: 0 }}>
                      {m.dim}-d
                    </div>
                  </div>
                ))}
              </div>

              {weights!.map((row, i) => (
                <div key={i} style={{ display: "flex" }}>
                  <div
                    style={{
                      width: LABEL_PX,
                      display: "flex",
                      alignItems: "center",
                      justifyContent: "flex-end",
                      paddingRight: 10,
                      fontSize: 10,
                      fontWeight: 600,
                      letterSpacing: "0.12em",
                      textTransform: "uppercase",
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
                          background: cellColor(v, i, j),
                          margin: 2,
                          border: isHover ? "1px solid var(--accent-bright)" : "1px solid rgba(255,255,255,0.04)",
                          borderRadius: 6,
                          display: "flex",
                          alignItems: "center",
                          justifyContent: "center",
                          fontSize: 18,
                          fontWeight: 500,
                          fontFamily: "var(--font-mono)",
                          color: v > 0.4 ? "#fff" : "rgba(226,232,240,0.85)",
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
          </div>

          <hr className="lx-divider" />

          <div className="lx-label" style={{ marginBottom: 6 }}>
            Attention head entropy · 8 heads
          </div>
          <div style={{ display: "flex", justifyContent: "space-between", padding: "4px 4px" }}>
            {[0, 1, 2, 3, 4, 5, 6, 7].map((h) => (
              <span
                key={h}
                className="lx-dim lx-mono"
                style={{ fontSize: 10, letterSpacing: "0.06em" }}
              >
                h{h}
              </span>
            ))}
          </div>
          <div className="lx-dim" style={{ display: "flex", justifyContent: "space-between", fontSize: 10, marginTop: 14 }}>
            <span>Rows: query · Cols: key</span>
            <span className="lx-mono">last fwd · 84ms</span>
          </div>
        </>
      )}
    </section>
  );
}
