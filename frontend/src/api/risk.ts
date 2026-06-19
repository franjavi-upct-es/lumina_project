// frontend/src/api/risk.ts
import { apiClient, type RequestOpts } from "./client";
import type { KillSwitchResponse, KillSwitchState } from "../types/risk.types";

export const riskApi = {
  getKillSwitch: (opts?: RequestOpts) =>
    apiClient.get<KillSwitchResponse>("/api/risk/kill-switch", opts).then((r) => r.data),
  setKillSwitch: (state: KillSwitchState, reason = "") =>
    apiClient.post<KillSwitchResponse>("/api/risk/kill-switch", { state, reason }).then((r) => r.data),
};
