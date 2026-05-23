// frontend/src/store/riskSlice.ts
import { create } from "zustand";
import type { KillSwitchState } from "../types/risk.types";

interface RiskStore {
  killSwitchState: KillSwitchState;
  setKillSwitch: (s: KillSwitchState) => void;
}

export const useRiskStore = create<RiskStore>((set) => ({
  killSwitchState: "NORMAL",
  setKillSwitch: (s) => set({ killSwitchState: s }),
}));
