// frontend/src/api/agent.ts
import { apiClient } from "./client";
import type { AgentStatus } from "../types/agent.types";

export const agentApi = {
  getStatus: () => apiClient.get<AgentStatus>("/api/agent/status").then((r) => r.data),
  getHistory: (limit = 100) =>
    apiClient.get<Array<Record<string, unknown>>>("/api/agent/history", { params: { limit } })
      .then((r) => r.data),
};
