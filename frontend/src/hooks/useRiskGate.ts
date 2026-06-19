// frontend/src/hooks/useRiskGate.ts
import { useCallback } from "react";
import { riskApi } from "../api/risk";
import type { KillSwitchState } from "../types/risk.types";
import { usePolling } from "./usePolling";

export function useRiskGate() {
  const { data } = usePolling((signal) => riskApi.getKillSwitch({ signal }), 3000);
  const setState = useCallback(async (state: KillSwitchState, reason = "") => {
    return await riskApi.setKillSwitch(state, reason);
  }, []);
  // The response declares `state` as a plain string; narrow to the domain enum.
  return { state: (data?.state as KillSwitchState | undefined) ?? "NORMAL", setState };
}
