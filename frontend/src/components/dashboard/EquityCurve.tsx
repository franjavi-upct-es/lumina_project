// frontend/src/components/dashboard/EquityCurve.tsx
//
// Equity / benchmark area chart styled for the dark operator console.

import { useMemo } from "react";
import {
  Area,
  CartesianGrid,
  ComposedChart,
  Line,
  ReferenceDot,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

export interface EquityPoint {
  time: string;
  equity: number;
  benchmark?: number;
}

interface Props {
  data: EquityPoint[];
  /** Height in pixels. Defaults to 320 for the dashboard layout. */
  height?: number;
  /** Optional marker for a notable event (e.g. risk-off click). */
  marker?: { time: string; equity: number; label: string };
}

export function EquityCurve({ data, height = 320, marker }: Props) {
  const series = useMemo(() => data, [data]);
  if (series.length === 0) {
    return (
      <div
        className="lx-dim"
        style={{
          height,
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          textAlign: "center",
          fontSize: 13,
          border: "1px dashed var(--border)",
          borderRadius: 6,
        }}
      >
        No portfolio history recorded yet.
      </div>
    );
  }
  return (
    <ResponsiveContainer width="100%" height={height}>
      <ComposedChart data={series} margin={{ top: 4, right: 12, bottom: 0, left: 0 }}>
        <defs>
          <linearGradient id="lx-equity-fill" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%"   stopColor="#4a90d9" stopOpacity={0.45} />
            <stop offset="100%" stopColor="#4a90d9" stopOpacity={0.02} />
          </linearGradient>
        </defs>
        <CartesianGrid stroke="rgba(148,163,184,0.08)" vertical={false} />
        <XAxis
          dataKey="time"
          stroke="rgba(148,163,184,0.5)"
          tick={{ fontSize: 10, fill: "rgba(148,163,184,0.7)", fontFamily: "var(--font-mono)" }}
          tickLine={false}
          axisLine={false}
        />
        <YAxis
          orientation="right"
          stroke="rgba(148,163,184,0.5)"
          tick={{ fontSize: 10, fill: "rgba(148,163,184,0.7)", fontFamily: "var(--font-mono)" }}
          tickLine={false}
          axisLine={false}
          tickFormatter={(v) => `$${Math.round(v / 1000)}k`}
          width={50}
        />
        <Tooltip
          cursor={{ stroke: "rgba(74,144,217,0.4)", strokeWidth: 1 }}
          contentStyle={{
            background: "#0b1220",
            border: "1px solid rgba(148,163,184,0.2)",
            borderRadius: 6,
            fontSize: 11,
            color: "#e2e8f0",
          }}
        />
        <Area
          type="monotone"
          dataKey="equity"
          stroke="#5ba6f5"
          strokeWidth={1.5}
          fill="url(#lx-equity-fill)"
          isAnimationActive={false}
        />
        {series.some((p) => p.benchmark != null) && (
          <Line
            type="monotone"
            dataKey="benchmark"
            stroke="rgba(148,163,184,0.5)"
            strokeWidth={1}
            strokeDasharray="3 3"
            dot={false}
            isAnimationActive={false}
          />
        )}
        {marker && (
          <ReferenceDot x={marker.time} y={marker.equity} r={4} fill="#ef4444" stroke="#ef4444" />
        )}
      </ComposedChart>
    </ResponsiveContainer>
  );
}
