// frontend/react-app/src/components/charts/CandlestickChart.jsx
// OHLCV candlestick chart with volume subplot.
// React equivalent of create_candlestick_chart() from charts.py
// Uses Recharts ComposedCHart; a lightweight-charts or D3 upgrade if possible
// for higher-frequency (1m) data in a future iteration.

import {
  ComposedChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
} from "recharts";
import { format } from "date-fns";
import {
  COLORS,
  CHART_AXIS_COLOR,
  CHART_TOOLTIP_BG,
  CHART_TOOLTIP_BORDER,
  CHART_GRID_COLOR,
} from "../../constants";

/** Custom OHLC candle redered as a single bar group using a D3-style shape. */
function CandleShape({ x, y, width, height, open, close, high, low }) {
  const isGreen = close >= open;
  const color = isGreen ? COLORS.sucess : COLORS.danger;
  const bodyTop = isGreen ? y : y + height;
  const bodyH = Math.abs(height);
  const midX = x + width / 2;

  return (
    <g>
      {/* Wick */}
      <line
        x1={midX}
        y1={y - (high - Math.max(open, close))}
        x2={midX}
        y2={y + height + (Math.min(open, close) - low)}
        stroke={color}
        strokeWidth={1}
      />
      {/* Body */}
      <rect
        x={x + 1}
        y={bodyTop}
        width={width - 2}
        height={Math.max(bodyH, 1)}
        fill={color}
      />
    </g>
  );
}

const CustomTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null;
  const d = payload[0]?.payload;
  if (!d) return null;
  return (
    <div className="rounded-lg p-3 text-xs font-mono border">
      <p className="text-surface-300 mb-1">{label}</p>
      <p style={{ color: d.close >= d.open ? COLORS.sucess : COLORS.danger }}>
        0 {d.open?.toFixed(2)} H {d.high?.toFixed(2)} L {d.low?.toFixed(2)} C{" "}
        {d.close?.toFixed(2)}
      </p>
      {d.volume && (
        <p className="text-surface-300 mt-1">
          Vol {(d.volume / 1e6).toFixed(2)}M
        </p>
      )}
    </div>
  );
};

/**
 * @param {Array}   data        - Array of { time, open, high, low, close, volume }
 * @param {string}  [ticker]    - Displayed in chart title
 * @param {boolean} [showVolume]
 * @param {number}  [height]
 */
export function CandlestickChart({
  data = [],
  ticker = "",
  showVolume = true,
  height = 500,
}) {
  const formatted = data.map((d) => ({
    ...d,
    label:
      typeof d.time === "string"
        ? d.time.slice(0, 10)
        : format(new Date(d.time), "MMM d"),
  }));

  return (
    <div style={{ height }}>
      <ResponsiveContainer width="100%" height="100%">
        <ComposedChart
          data={formatted}
          margin={{ top: 10, right: 10, left: 0, bottom: 0 }}
        >
          <CartesianGrid
            stroke={CHART_GRID_COLOR}
            strokeDasharray="3 3"
            vertical={false}
          />
          <XAxis
            dataKey="label"
            tick={{ fontSize: 11, fill: CHART_AXIS_COLOR }}
            tickLine={false}
            axisLine={{ stroke: CHART_GRID_COLOR }}
          />
          <YAxis
            yAxisId="price"
            orientation="right"
            tick={{
              fontSize: 11,
              fill: CHART_AXIS_COLOR,
              fontFamily: "monospace",
            }}
            tickFormatter={(v) => `$${v.toFixed(0)}`}
            tickLine={false}
            axisLine={false}
            width={60}
          />
          {showVolume && (
            <YAxis
              yAxisId="vol"
              orientation="left"
              tick={{
                fontSize: 10,
                fill: CHART_AXIS_COLOR,
              }}
              tickFormatter={(v) => `${(v / 1e6).toFixed(0)}M`}
              tickLine={false}
              axisLine={false}
              width={45}
              domain={[0, (max) => max * 4]}
            />
          )}
          <Tooltip content={<CustomTooltip />} />
          {/* Volume bars in background */}
          {showVolume && (
            <Bar
              yAxisId="vol"
              dataKey="volume"
              fill={COLORS.primary}
              opacity={0.25}
              radius={0}
            />
          )}
          {/* Close line as proxy for candle bodies (Recharts doesn't have native candles) */}
          <Bar
            yAxisId="price"
            dataKey="close"
            shape={<CandleShape />}
            fill={COLORS.sucess}
          />
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  );
}
