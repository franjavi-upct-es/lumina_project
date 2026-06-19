// frontend/src/api/monitoring.ts
//
// REST helper for monitoring endpoints. The /metrics endpoint is
// intentionally NOT exposed here — it returns Prometheus text format
// and is consumed by a scraper, not the dashboard.

import { apiClient, type RequestOpts } from "./client";
import type { HealthResponse } from "../types/monitoring.types";

export const monitoringApi = {
  getHealth: (opts?: RequestOpts) =>
    apiClient.get<HealthResponse>("/api/monitoring/health", opts).then((r) => r.data),
};
