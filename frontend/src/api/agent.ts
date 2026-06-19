// frontend/src/api/agent.ts
import { apiClient, withWsToken, wsBase } from "./client";
import type { AgentStatus } from "../types/agent.types";

/** Canonical WebSocket URL for the agent stream (carries the API token). */
export const AGENT_STREAM_URL = withWsToken(`${wsBase()}/api/agent/stream`);

export const agentApi = {
  getStatus: () => apiClient.get<AgentStatus>("/api/agent/status").then((r) => r.data),
  getHistory: (limit = 100) =>
    apiClient.get<Array<Record<string, unknown>>>("/api/agent/history", { params: { limit } })
      .then((r) => r.data),
};
