// frontend/src/store/connectionSlice.ts
//
// Cross-cutting connectivity state shared by every data hook so the app can
// render a single, authoritative "degraded backend" banner instead of each
// panel failing silently on its own timer.
//
//   * apiReachable   — did the most recent REST call reach the backend?
//   * streamConnected — is the agent WebSocket currently open?
//   * authError      — was a request rejected for auth (REST 401/403, WS 1008)?
//   * versionMismatch — does the backend's advertised contract MAJOR differ
//                       from the version this build was compiled against?
//
// Setters no-op when the value is unchanged so the high-frequency REST pollers
// don't trigger needless re-renders.

import { create } from "zustand";

/**
 * Contract version this frontend was built against. Compared (MAJOR only)
 * against the backend's `X-API-Version` response header. Keep in step with
 * backend `API_VERSION`; a MAJOR bump on either side surfaces the banner.
 */
export const EXPECTED_API_VERSION = "3.0.0";

const major = (v: string): string => v.split(".", 1)[0] ?? v;

interface ConnectionStore {
  apiReachable: boolean | null;
  streamConnected: boolean;
  authError: boolean;
  versionMismatch: boolean;
  serverVersion: string | null;
  setApiReachable: (v: boolean) => void;
  setStreamConnected: (v: boolean) => void;
  setAuthError: (v: boolean) => void;
  /** Record the backend-advertised version and recompute versionMismatch. */
  reportServerVersion: (v: string | null) => void;
}

export const useConnectionStore = create<ConnectionStore>((set) => ({
  apiReachable: null,
  streamConnected: false,
  authError: false,
  versionMismatch: false,
  serverVersion: null,
  setApiReachable: (v) =>
    set((s) => (s.apiReachable === v ? s : { apiReachable: v })),
  setStreamConnected: (v) =>
    set((s) => (s.streamConnected === v ? s : { streamConnected: v })),
  setAuthError: (v) => set((s) => (s.authError === v ? s : { authError: v })),
  reportServerVersion: (v) =>
    set((s) => {
      if (v === s.serverVersion) return s;
      const mismatch = v !== null && major(v) !== major(EXPECTED_API_VERSION);
      return { serverVersion: v, versionMismatch: mismatch };
    }),
}));
