// frontend/src/hooks/usePerceptionHealth.ts
//
// Polls /api/monitoring/health every 10 s and exposes the result to the
// dashboard. The endpoint is cheap (no DB writes; one ping per subsystem) so
// 10 s is well below any realistic load concern.
//
// Why 10 s, not 3 s?
//   1. The dashboard uses this for *infrastructure* status; transient blips
//      don't need pixel-accurate timing.
//   2. The endpoint's RTT is dominated by the broker network hop. Polling
//      harder yields no useful information and burns API quota.
//
// Built on the shared usePolling primitive (which handles interval lifecycle,
// error clearing, and in-flight cancellation); this hook just shapes the
// result into the PerceptionHealthState the dashboard consumes.

import { useMemo } from "react";
import { monitoringApi } from "../api/monitoring";
import { usePolling } from "./usePolling";
import type { HealthStatus } from "../types/monitoring.types";

const POLL_INTERVAL_MS = 10_000;

export interface PerceptionHealthState {
  /** True when the aggregate API status is "ok". False on "degraded" / "down" / error / no data. */
  healthy: boolean;
  /** Raw aggregate status from the backend; null if no successful fetch yet. */
  status: HealthStatus | null;
  /** Last full HealthResponse components for callers that want to drill in. */
  components: Record<string, Record<string, unknown>>;
  /** ISO timestamp of the last successful fetch; null before the first fetch. */
  lastChecked: string | null;
  /** Last fetch error, if any. Cleared on success. */
  error: Error | null;
}

export function usePerceptionHealth(): PerceptionHealthState {
  const { data, error, lastUpdated } = usePolling(
    (signal) => monitoringApi.getHealth({ signal }),
    POLL_INTERVAL_MS,
  );

  return useMemo<PerceptionHealthState>(
    () => ({
      // On error we keep the last-good status/components but report unhealthy.
      healthy: !error && data?.status === "ok",
      status: data?.status ?? null,
      components: data?.components ?? {},
      lastChecked: lastUpdated !== null ? new Date(lastUpdated).toISOString() : null,
      error,
    }),
    [data, error, lastUpdated],
  );
}
