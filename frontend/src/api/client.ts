// frontend/src/api/client.ts
import axios, { type AxiosError } from "axios";

import { useConnectionStore } from "../store/connectionSlice";

/** Per-request options callers may forward (e.g. an AbortSignal from usePolling). */
export interface RequestOpts {
  signal?: AbortSignal;
}

// Same-origin by default. In dev, Vite proxies `/api` and the WebSocket routes
// to the backend (see vite.config.ts), so the app needs no CORS and no
// absolute URL. Set VITE_API_BASE / VITE_WS_BASE to target a remote backend.
const API_BASE = import.meta.env.VITE_API_BASE ?? "";

// Exported so WebSocket hooks can pass it as a `?token=` query param — browsers
// cannot set custom headers on the WS handshake (audit F2).
export const API_KEY = import.meta.env.VITE_API_KEY || "";

const WEAK_DEFAULT_API_KEY = "change_me_in_production";
if (import.meta.env.DEV && (API_KEY === "" || API_KEY === WEAK_DEFAULT_API_KEY)) {
  console.warn(
    `[Lumina] VITE_API_KEY is ${API_KEY === "" ? "empty" : "the placeholder value"}. ` +
      "Requests only succeed while the backend runs with ENVIRONMENT=development and " +
      "no API_KEY set. Before staging/prod, set VITE_API_KEY to match the backend API_KEY.",
  );
}

/**
 * Resolve the WebSocket origin. When VITE_WS_BASE is unset we derive it from
 * the page origin so the Vite dev proxy can forward it same-origin (and prod
 * behind a reverse proxy works without configuration).
 */
export function wsBase(): string {
  const explicit = import.meta.env.VITE_WS_BASE;
  if (explicit) return explicit;
  if (typeof window !== "undefined" && window.location?.host) {
    const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
    return `${proto}//${window.location.host}`;
  }
  return "ws://localhost:8000";
}

/** Append the API key as a `token` query param for WebSocket URLs. */
export function withWsToken(url: string): string {
  if (!API_KEY) return url;
  const sep = url.includes("?") ? "&" : "?";
  return `${url}${sep}token=${encodeURIComponent(API_KEY)}`;
}

export const apiClient = axios.create({
  baseURL: API_BASE,
  headers: { "x-api-key": API_KEY, "Content-Type": "application/json" },
  timeout: 10_000,
});

// Feed connectivity, auth, and contract-version signals into the shared store
// so the UI can show one authoritative banner instead of silent per-panel
// failures (audit #5, #11, #12).
function noteVersion(headers: unknown): void {
  const v = (headers as Record<string, unknown> | undefined)?.["x-api-version"];
  if (typeof v === "string") useConnectionStore.getState().reportServerVersion(v);
}

apiClient.interceptors.response.use(
  (response) => {
    const conn = useConnectionStore.getState();
    conn.setApiReachable(true);
    conn.setAuthError(false);
    noteVersion(response.headers);
    return response;
  },
  (err: AxiosError) => {
    const conn = useConnectionStore.getState();
    if (err.response) {
      // The backend answered: it's reachable. Inspect for auth + version.
      conn.setApiReachable(true);
      noteVersion(err.response.headers);
      if (err.response.status === 401 || err.response.status === 403) {
        conn.setAuthError(true);
      }
    } else if (err.code !== "ERR_CANCELED" && !axios.isCancel(err)) {
      // No response and not a cancelled request => backend unreachable.
      conn.setApiReachable(false);
    }
    return Promise.reject(err);
  },
);
