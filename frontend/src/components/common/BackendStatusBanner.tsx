// frontend/src/components/common/BackendStatusBanner.tsx
//
// A single authoritative banner for backend connectivity problems. It reads
// the shared connection store (fed by the axios interceptor and the WS hooks)
// so the operator sees ONE clear message — auth failure, contract-version
// drift, or an unreachable backend — instead of each panel silently erroring
// on its own timer.

import { EXPECTED_API_VERSION, useConnectionStore } from "../../store/connectionSlice";

type Tone = "error" | "warn";

const TONE_STYLE: Record<Tone, React.CSSProperties> = {
  error: {
    background: "rgba(215, 58, 73, 0.12)",
    color: "var(--red, #d73a49)",
    borderBottom: "1px solid var(--red, #d73a49)",
  },
  warn: {
    background: "rgba(224, 168, 0, 0.12)",
    color: "var(--amber, #e0a800)",
    borderBottom: "1px solid var(--amber, #e0a800)",
  },
};

export function BackendStatusBanner() {
  const apiReachable = useConnectionStore((s) => s.apiReachable);
  const authError = useConnectionStore((s) => s.authError);
  const versionMismatch = useConnectionStore((s) => s.versionMismatch);
  const serverVersion = useConnectionStore((s) => s.serverVersion);

  let tone: Tone;
  let text: string;
  let showReload = false;

  if (authError) {
    tone = "error";
    text =
      "Authentication failed (401/403). Set VITE_API_KEY to match the backend's API_KEY, then reload.";
    showReload = true;
  } else if (versionMismatch) {
    tone = "warn";
    text = `Backend API version (${serverVersion ?? "unknown"}) differs from this app (${EXPECTED_API_VERSION}). Reload to load a compatible UI.`;
    showReload = true;
  } else if (apiReachable === false) {
    tone = "error";
    text = "Backend unreachable — retrying. Displayed data may be stale.";
  } else {
    return null;
  }

  return (
    <div
      role="alert"
      style={{
        ...TONE_STYLE[tone],
        display: "flex",
        alignItems: "center",
        gap: 12,
        padding: "8px 20px",
        fontFamily: "var(--font-mono)",
        fontSize: 12,
        position: "sticky",
        top: 0,
        zIndex: 6,
      }}
    >
      <span className="lx-dot" style={{ background: "currentColor" }} />
      <span style={{ flex: 1 }}>{text}</span>
      {showReload && (
        <button
          type="button"
          onClick={() => window.location.reload()}
          style={{
            background: "transparent",
            border: "1px solid currentColor",
            color: "currentColor",
            borderRadius: 4,
            padding: "2px 10px",
            cursor: "pointer",
            font: "inherit",
          }}
        >
          Reload
        </button>
      )}
    </div>
  );
}
