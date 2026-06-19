// frontend/src/types/monitoring.types.ts
//
// Wire format for `/api/monitoring/health`, aliased from the backend-generated
// OpenAPI schemas (./api.generated.ts) so it stays in lockstep with
// backend.api.schemas.HealthResponse. Regenerate with `make openapi`.
//
// `components` is a free-form per-subsystem health mapping. Known keys:
//   redis        — { connected: boolean; latency_ms?: number; error?: string }
//   timescale    — { connected: boolean; latency_ms?: number; error?: string }
//   broker       — { connected: boolean; equity?: number;     error?: string }
//   kill_switch  — { state: "NORMAL" | "CLOSE_ONLY" | "LIQUIDATE_ALL" }
// It is intentionally permissive; callers should defensive-check fields.

import type { components } from "./api.generated";

export type HealthResponse = components["schemas"]["HealthResponse"];
export type HealthStatus = HealthResponse["status"];
