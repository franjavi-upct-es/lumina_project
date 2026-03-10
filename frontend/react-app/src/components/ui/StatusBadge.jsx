// frontend/react-app/src/components/ui/StatusBadge.jsx
// Displays a colored status pill for agent mode, safety status, and API health.
// Used in the Sidebar system status section and Agent Monitor page.

import { clsx } from "clsx";
import { SAFETY_STATUS } from "../../constants";

const STATUS_STYLES = {
  // Safety Arbitrator states
  [SAFETY_STATUS.ACTIVE]: {
    dot: "bg-brand-success",
    text: "text-brand-success",
    label: "Active",
  },
  [SAFETY_STATUS.GUARDED]: {
    dot: "bg-brand-warning",
    text: "text-brand-warning",
    label: "Guarded",
  },
  [SAFETY_STATUS.CRITICAL]: {
    dot: "bg-brand-danger",
    text: "text-brand-danger",
    label: "Critical",
  },
  [SAFETY_STATUS.OFFLINE]: {
    dot: "bg-surface-300",
    text: "text-surface-300",
    label: "Offline",
  },
  // Generic statuses
  online: {
    dot: "bg-brand-success",
    text: "text-brand-success",
    label: "Online",
  },
  offline: {
    dot: "bg-brand-danger",
    text: "text-brand-danger",
    label: "Offline",
  },
  connecting: {
    dot: "bg-brand-warning",
    text: "text-brand-warning",
    label: "Connecting",
  },
  connected: {
    dot: "bg-brand-success",
    text: "text-brand-success",
    label: "Connected",
  },
  error: { dot: "bg-brand-danger", text: "text-brand-danger", label: "Error" },
  live: {
    dot: "bg-brand-success animate-pulse",
    text: "text-brand-success",
    label: "Live",
  },
  paper: { dot: "bg-brand-info", text: "text-brand-info", label: "Paper" },
  training: {
    dot: "bg-brand-purple animate-pulse",
    text: "text-brand-purple",
    label: "Training",
  },
  unknown: {
    dot: "bg-surface-300",
    text: "text-surface-300",
    label: "Unknown",
  },
};

/**
 * @param {string}  status    - One of the STATUS_STYLES keys
 * @param {string}  [label]   - Override the default label
 * @param {boolean} [pulse]   - Force a pulsing dot (indicates live data)
 * @param {string}  [className]
 */
export function StatusBadge({
  status = "unknown",
  label,
  pulse = false,
  className,
}) {
  const style = STATUS_STYLES[status] || STATUS_STYLES.unknown;
  const dotCls = clsx(
    "w-2 h-2 rounded-full flex-shrink-0",
    style.dot,
    pulse && "animate-pulse",
  );

  return (
    <span
      className={clsx(
        "inline-flex items-center gap-1.5 text-sm font-medium",
        style.text,
        className,
      )}
    >
      <span className={dotCls} />
      {label ?? style.label}
    </span>
  );
}
