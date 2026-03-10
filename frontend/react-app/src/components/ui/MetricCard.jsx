// frontend/react-app/src/components/ui/MetricCard.jsx
// Reusable metric card component - React equivalent of the Streamlit
// render_metric_card() and reder_styled_metric_card() helpers in metrics.py.

import clsx from "clsx";

/**
 * Displays a single KPI metric with optional delta and accent color.
 *
 * @param {string} label        - Metric label (e.g. "Sharpe Ratio")
 * @param {string} value        - Formatted value string (e.g. "1.85")
 * @param {string} [delta]      - Optional delta string (e.g. "+0.12")
 * @param {boolean} [deltaGood] - Whether a positive delta is favorable
 * @param {string} [accentColor]- Left border accent color (default brand.primary)
 * @param {string} [icon]       - Optional emoji or icon character
 * @param {string} [helpText]   - Optional tooltip text
 * @param {string} [className]  - Additional Tailwind classes
 */
export function MetricCard({
  label,
  value,
  delta,
  deltaGood = true,
  accentColor = "#1f77b4",
  icon,
  helpText,
  className,
}) {
  const isPositiveDelta =
    delta && (delta.startsWith("+") || parseFloat(delta) > 0);
  const isNegativeDelta =
    delta && (delta.startsWith("-") || parseFloat(delta) < 0);

  // If deltaGood=false (e.g. drawdown), colors are inverted
  const deltaColor = delta
    ? isPositiveDelta
      ? deltaGood
        ? "text-brand-success"
        : "text-brand-danger"
      : isNegativeDelta
        ? deltaGood
          ? "text-brand-danger"
          : "text-brand-success"
        : "text-surface-300"
    : null;

  return (
    <div
      className={clsx(
        "bg-surface-700 rounded-lg p-4 border-l-4 transition-colors hover:bg-surface-600",
        className,
      )}
      style={{ borderLeftColor: accentColor }}
      title={helpText}
    >
      {/* Label */}
      <p className="text-surface-300 text-xs font-medium uppercase tracking-wide mb-1">
        {icon && <span className="mr-1">{icon}</span>}
        {label}
      </p>

      {/* Value */}
      <p className="text-white text-2xl font-bold font-mono leading-tight">
        {value}
      </p>

      {/* Delta */}
      {delta && (
        <p className={clsx("text-sm font-medium mt-1", deltaColor)}>{delta}</p>
      )}
    </div>
  );
}

/**
 * Renders a row of MetricCard componenets in an auto-fit grid.
 *
 * @param {Array} metrics - Array of MetricCard props objects
 * @param {number} [cols] - Fixed number of columns (default: auto)
 */
export function MetricRow({ metrics, cols }) {
  const gridClass = cols
    ? `grid-cols-${cols}`
    : "grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5";

  return (
    <div className={clsx("grid gap-4", gridClass)}>
      {metrics.map((m, i) => (
        <MetricCard key={i} {...m} />
      ))}
    </div>
  );
}
