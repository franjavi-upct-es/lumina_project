// frontend/src/api/agent.ts
import { apiClient } from "./client";
import type { AgentStatus } from "../types/agent.types";

const WS_BASE = import.meta.env.VITE_WS_BASE || "ws://localhost:8000";

/** Canonical WebSocket URL for the agent stream. */
export const AGENT_STREAM_URL = `${WS_BASE}/api/agent/stream`;

export const agentApi = {
  getStatus: () => apiClient.get<AgentStatus>("/api/agent/status").then((r) => r.data),
  getHistory: (limit = 100) =>
    apiClient.get<Array<Record<string, unknown>>>("/api/agent/history", { params: { limit } })
      .then((r) => r.data),
};
