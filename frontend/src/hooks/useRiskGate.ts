// frontend/src/hooks/useRiskGate.ts
import { useCallback } from "react";
import { riskApi } from "../api/risk";
import type { KillSwitchState } from "../types/risk.types";
import { usePolling } from "./usePolling";

export function useRiskGate() {
  const { data } = usePolling(() => riskApi.getKillSwitch(), 3000);
  const setState = useCallback(async (state: KillSwitchState, reason = "") => {
    return await riskApi.setKillSwitch(state, reason);
  }, []);
  return { state: data?.state ?? "NORMAL", setState };
}
