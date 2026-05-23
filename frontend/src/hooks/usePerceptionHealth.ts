// frontend/src/hooks/usePerceptionHealth.ts
//
// Polls /api/monitoring/health every 10 s and exposes the result to
// the dashboard. The endpoint is cheap (no DB writes; one ping to
// each subsystem) so 10 s is well below any realistic load concern.
//
// Why 10 s, not 3 s?
// -------------------
// Two reasons:
//   1. The dashboard uses this for *infrastructure* status; transient
//      blips don't need pixel-accurate timing.
//   2. The endpoint's own RTT is dominated by the broker call which
//      includes a network hop. Polling more aggressively yields no
//      useful information and burns API quota.

import { useEffect, useState } from "react";
import { monitoringApi } from "../api/monitoring";
import type { HealthResponse, HealthStatus } from "../types/monitoring.types";

const POLL_INTERVAL_MS = 10_000;

export interface PerceptionHealthState {
  /** True when the aggregate API status is "ok". False on "degraded" / "down" / no data. */
  healthy: boolean;
  /** Raw aggregate status from the backend; null if no successful fetch yet. */
  status: HealthStatus | null;
  /** Last full HealthResponse for callers that want to drill in. */
  components: Record<string, Record<string, unknown>>;
  /** ISO timestamp of the last successful fetch; null before the first fetch. */
  lastChecked: string | null;
  /** Last fetch error, if any. Cleared on success. */
  error: Error | null;
}

const INITIAL: PerceptionHealthState = {
  healthy: false,
  status: null,
  components: {},
  lastChecked: null,
  error: null,
};

export function usePerceptionHealth(): PerceptionHealthState {
  const [state, setState] = useState<PerceptionHealthState>(INITIAL);

  useEffect(() => {
    let cancelled = false;

    const tick = async () => {
      try {
        const response: HealthResponse = await monitoringApi.getHealth();
        if (cancelled) return;
        setState({
          healthy: response.status === "ok",
          status: response.status,
          components: response.components,
          lastChecked: new Date().toISOString(),
          error: null,
        });
      } catch (err) {
        if (cancelled) return;
        setState((prev) => ({
          ...prev,
          healthy: false,
          error: err instanceof Error ? err : new Error(String(err)),
        }));
      }
    };

    tick();
    const id = setInterval(tick, POLL_INTERVAL_MS);
    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, []);

  return state;
}
