// frontend/src/api/risk.ts
import { apiClient } from "./client";
import type { KillSwitchResponse, KillSwitchState } from "../types/risk.types";

export const riskApi = {
  getKillSwitch: () => apiClient.get<KillSwitchResponse>("/api/risk/kill-switch").then((r) => r.data),
  setKillSwitch: (state: KillSwitchState, reason = "") =>
    apiClient.post<KillSwitchResponse>("/api/risk/kill-switch", { state, reason }).then((r) => r.data),
};
