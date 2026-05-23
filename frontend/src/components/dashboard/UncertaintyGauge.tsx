// frontend/src/components/dashboard/UncertaintyGauge.tsx
//
// SVG arc gauge that visualises the agent's epistemic uncertainty.
//
// Range:        0..1   (clamped on render — values outside the range are
//                       defensively projected to the nearest endpoint).
// Colour bands: GREEN     when value < 0.50
//               AMBER     when 0.50 <= value < 0.85
//               RED       when value >= 0.85
//
// The thresholds match the architecture spec
// (UNCERTAINTY_THRESHOLD = 0.85 from constants.py) so what the dashboard
// shows is exactly the threshold the Uncertainty Gate uses to veto
// actions. Operators see "the gauge just turned red" at the same moment
// the Gate engages — no cognitive translation step.
//
// Animation: a CSS transition on the needle's `stroke-dashoffset`
// gives a 250 ms ease-out movement. Recharts is intentionally NOT used
// here — a single SVG path is dramatically cheaper than a full chart
// component for a value that updates 10+ times per second.

import { useMemo } from "react";

const SIZE = 180;
const STROKE_WIDTH = 14;
const RADIUS = (SIZE - STROKE_WIDTH) / 2;
const CENTER = SIZE / 2;
// The arc covers the lower half of a circle, from 180° (left) to 360°
// (right), giving a familiar speedometer feel.
const ARC_START_ANGLE_DEG = 180;
const ARC_END_ANGLE_DEG = 360;
const ARC_SWEEP_DEG = ARC_END_ANGLE_DEG - ARC_START_ANGLE_DEG;

// Total arc length in SVG units. Used to compute the partial-fill
// stroke-dashoffset that produces the needle effect.
const ARC_LENGTH = Math.PI * RADIUS;

// Threshold bands. The colours match those used elsewhere in the
// dashboard (the kill-switch button uses #d73a49 for "armed").
const BAND_GREEN_MAX = 0.5;
const BAND_AMBER_MAX = 0.85;
const COLOR_GREEN = "#28a745";
const COLOR_AMBER = "#ffb020";
const COLOR_RED = "#d73a49";
const COLOR_TRACK = "#e1e4e8";

function clamp01(v: number): number {
  if (!Number.isFinite(v)) return 0;
  return Math.max(0, Math.min(1, v));
}

function bandColor(value: number): string {
  if (value < BAND_GREEN_MAX) return COLOR_GREEN;
  if (value < BAND_AMBER_MAX) return COLOR_AMBER;
  return COLOR_RED;
}

// Convert (cx, cy, r, angle_deg) → (x, y) on the circle. SVG uses
// y-down so angles count clockwise from the +x axis.
function polarToCartesian(cx: number, cy: number, r: number, angleDeg: number) {
  const a = (angleDeg * Math.PI) / 180;
  return { x: cx + r * Math.cos(a), y: cy + r * Math.sin(a) };
}

function describeArc(cx: number, cy: number, r: number, startDeg: number, endDeg: number) {
  const start = polarToCartesian(cx, cy, r, startDeg);
  const end = polarToCartesian(cx, cy, r, endDeg);
  const largeArc = endDeg - startDeg <= 180 ? 0 : 1;
  return `M ${start.x} ${start.y} A ${r} ${r} 0 ${largeArc} 1 ${end.x} ${end.y}`;
}

interface UncertaintyGaugeProps {
  value: number;
  /** Optional label rendered above the numeric readout. */
  label?: string;
}

export function UncertaintyGauge({ value, label = "Uncertainty" }: UncertaintyGaugeProps) {
  const safeValue = clamp01(value);
  const color = bandColor(safeValue);

  // The arc goes from `start` to `start + fillSweep`. We *render* the
  // full arc as the track, then overlay a partial-coloured arc whose
  // length is proportional to `safeValue`.
  const fillSweepDeg = ARC_SWEEP_DEG * safeValue;
  const fillArcPath = useMemo(
    () => describeArc(CENTER, CENTER, RADIUS, ARC_START_ANGLE_DEG, ARC_START_ANGLE_DEG + fillSweepDeg),
    [fillSweepDeg],
  );
  const trackArcPath = useMemo(
    () => describeArc(CENTER, CENTER, RADIUS, ARC_START_ANGLE_DEG, ARC_END_ANGLE_DEG),
    [],
  );

  // Tick positions for the band-boundary markers.
  const greenTick = polarToCartesian(CENTER, CENTER, RADIUS, ARC_START_ANGLE_DEG + ARC_SWEEP_DEG * BAND_GREEN_MAX);
  const amberTick = polarToCartesian(CENTER, CENTER, RADIUS, ARC_START_ANGLE_DEG + ARC_SWEEP_DEG * BAND_AMBER_MAX);

  return (
    <div style={{ display: "inline-block", textAlign: "center" }}>
      <svg width={SIZE} height={SIZE / 2 + 24} viewBox={`0 0 ${SIZE} ${SIZE / 2 + 24}`}>
        {/* Track (full arc, neutral colour) */}
        <path
          d={trackArcPath}
          fill="none"
          stroke={COLOR_TRACK}
          strokeWidth={STROKE_WIDTH}
          strokeLinecap="round"
        />
        {/* Filled arc, length proportional to value, colour matches band */}
        <path
          d={fillArcPath}
          fill="none"
          stroke={color}
          strokeWidth={STROKE_WIDTH}
          strokeLinecap="round"
          style={{ transition: "all 250ms ease-out" }}
        />
        {/* Band-boundary markers, so operators see the 0.50 / 0.85 lines */}
        <circle cx={greenTick.x} cy={greenTick.y} r={3} fill="#ffffff" stroke="#888888" />
        <circle cx={amberTick.x} cy={amberTick.y} r={3} fill="#ffffff" stroke="#888888" />
        {/* Numeric readout in the centre of the arc */}
        <text
          x={CENTER}
          y={CENTER - 4}
          textAnchor="middle"
          fontSize="26"
          fontWeight="600"
          fill={color}
        >
          {safeValue.toFixed(2)}
        </text>
      </svg>
      <div style={{ fontSize: 12, color: "#586069", marginTop: -4 }}>{label}</div>
    </div>
  );
}

// Total arc length export for unit tests / future overlays. Marking as
// `ARC_LENGTH` keeps it tree-shakeable.
export const _ARC_LENGTH = ARC_LENGTH;
