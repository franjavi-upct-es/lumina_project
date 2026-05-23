// frontend/src/types/monitoring.types.ts
//
// Wire format for `/api/monitoring/health`. Mirrors
// `backend.api.schemas.HealthResponse` exactly.

export type HealthStatus = "ok" | "degraded" | "down";

export interface HealthResponse {
  status: HealthStatus;
  /**
   * Free-form per-subsystem health objects. Known keys:
   *   redis        — { connected: boolean; latency_ms?: number; error?: string }
   *   timescale    — { connected: boolean; latency_ms?: number; error?: string }
   *   broker       — { connected: boolean; equity?: number;     error?: string }
   *   kill_switch  — { state: "NORMAL" | "CLOSE_ONLY" | "LIQUIDATE_ALL" }
   *
   * The TypeScript type is intentionally permissive because new
   * subsystems may be added on the backend without a frontend change;
   * callers should defensive-check the fields they read.
   */
  components: Record<string, Record<string, unknown>>;
}
