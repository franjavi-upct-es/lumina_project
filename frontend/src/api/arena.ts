// frontend/src/api/arena.ts
//
// REST client for the Spartan Arena endpoints. Mirrors the backtest.ts
// pattern: thin axios wrappers, no client-side caching.

import { apiClient } from "./client";
import type {
  ArenaRunCreatedResponse,
  ArenaRunMetadata,
  ArenaRunRequest,
  CounterfactualPair,
  DecisionRecord,
  DivergencePoint,
  RunSummary,
  StepExplanation,
} from "../types/arena.types";

interface ListOpts {
  limit?: number;
  offset?: number;
}

export async function startArenaRun(req: ArenaRunRequest): Promise<ArenaRunCreatedResponse> {
  return apiClient.post<ArenaRunCreatedResponse>("/arena/run", req).then((r) => r.data);
}

export async function getArenaRun(runId: string): Promise<ArenaRunMetadata> {
  return apiClient.get<ArenaRunMetadata>(`/arena/runs/${runId}`).then((r) => r.data);
}

export async function listArenaRuns(opts: ListOpts = {}): Promise<ArenaRunMetadata[]> {
  return apiClient
    .get<ArenaRunMetadata[]>("/arena/runs", { params: opts })
    .then((r) => r.data);
}

export async function getDecisions(
  runId: string,
  trajectoryId?: number,
  opts: ListOpts = {},
): Promise<DecisionRecord[]> {
  const params: Record<string, unknown> = { ...opts };
  if (trajectoryId !== undefined) params.trajectory_id = trajectoryId;
  return apiClient
    .get<DecisionRecord[]>(`/arena/runs/${runId}/decisions`, { params })
    .then((r) => r.data);
}

export async function getDivergences(
  runId: string,
  opts: ListOpts = {},
): Promise<DivergencePoint[]> {
  return apiClient
    .get<DivergencePoint[]>(`/arena/runs/${runId}/divergences`, { params: opts })
    .then((r) => r.data);
}

export async function getExplanations(
  runId: string,
  opts: ListOpts = {},
): Promise<StepExplanation[]> {
  return apiClient
    .get<StepExplanation[]>(`/arena/runs/${runId}/explanations`, { params: opts })
    .then((r) => r.data);
}

export async function getCounterfactualPairs(
  runId: string,
  opts: ListOpts = {},
): Promise<CounterfactualPair[]> {
  return apiClient
    .get<CounterfactualPair[]>(`/arena/runs/${runId}/pairs`, { params: opts })
    .then((r) => r.data);
}

export async function getRunSummary(runId: string): Promise<RunSummary> {
  return apiClient.get<RunSummary>(`/arena/runs/${runId}/summary`).then((r) => r.data);
}

export async function cancelArenaRun(runId: string): Promise<void> {
  await apiClient.post(`/arena/runs/${runId}/cancel`);
}
