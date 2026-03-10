// frontend/react-app/src/components/ui/AlertBanner.jsx
// Alert banner for safety system events: circuit breaker triggers,
// uncertainty gate activations, and profit-take executions.
// Replaces Streamlit's st.error / st.warning / st.info components.

import { clsx } from "clsx";

const VARIANTS = {
  info: { bg: "bg-brand-info/10", border: "border-brand-info", icon: "ℹ️" },
  success: {
    bg: "bg-brand-success/10",
    border: "border-brand-success",
    icon: "✅",
  },
  warning: {
    bg: "bg-brand-warning/10",
    border: "border-brand-warning",
    icon: "⚠️",
  },
  danger: {
    bg: "bg-brand-danger/10",
    border: "border-brand-danger",
    icon: "🚨",
  },
};

/**
 * @param {"info"|"success"|"warning"|"danger"} variant
 * @param {string}   title     - Bold title text
 * @param {string}   [message] - Supporting detail text
 * @param {Function} [onClose] - If provided, renders a close button
 */
export function AlertBanner({
  variant = "info",
  title,
  message,
  onClose,
  className,
}) {
  const v = VARIANTS[variant] || VARIANTS.info;

  return (
    <div
      className={clsx(
        "rounded-lg border p-4 flex items-start gap-3 animate-fade-in",
        v.bg,
        v.border,
        className,
      )}
    >
      <span className="text-lg flex-shrink-0">{v.icon}</span>
      <div className="flex-1 min-w-0">
        <p className="font-semibold text-white text-sm">{title}</p>
        {message && (
          <p className="text-surface-300 text-xs mt-0.5">{message}</p>
        )}
      </div>
      {onClose && (
        <button
          onClick={onClose}
          className="text-surface-300 hover:text-white transition-colors flex-shrink-0 text-lg leading-none"
          aria-label="Dismiss alert"
        >
          ×
        </button>
      )}
    </div>
  );
}
