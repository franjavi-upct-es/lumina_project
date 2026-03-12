// frontend/react-app/src/components/ui/LoadingSpinner.jsx
// Loading states for async data fetching — replaces Streamlit's st.spinner().

import { clsx } from "clsx";

/** Inline animated spinner. */
export function Spinner({ size = "md", className }) {
  const sizeClass = { sm: "w-4 h-4", md: "w-6 h-6", lg: "w-10 h-10" }[size];
  return (
    <span
      className={clsx(
        "inline-block border-2 border-surface-400 border-t-brand-primary rounded-full animate-spin",
        sizeClass,
        className,
      )}
      role="status"
      aria-label="Loading"
    />
  );
}

/** Full-area loading overlay for page-level fetches. */
export function PageLoader({ message = "Loading data..." }) {
  return (
    <div className="flex flex-col items-center justify-center min-h-[300px] gap-4">
      <Spinner size="lg" />
      <p className="text-surface-300 text-sm">{message}</p>
    </div>
  );
}

/** Inline error state shown when a query fails. */
export function ErrorState({ message, onRetry }) {
  return (
    <div className="flex flex-col items-center justify-center min-h-[200px] gap-3">
      <span className="text-4xl">⚠️</span>
      <p className="text-brand-danger text-sm font-medium text-center max-w-sm">
        {message}
      </p>
      {onRetry && (
        <button
          onClick={onRetry}
          className="px-4 py-2 bg-brand-primary/20 hover:bg-brand-primary/30 text-brand-primary rounded-lg text-sm transition-colors"
        >
          Retry
        </button>
      )}
    </div>
  );
}
