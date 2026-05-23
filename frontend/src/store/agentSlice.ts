// frontend/src/store/agentSlice.ts
import { create } from "zustand";

interface AgentStore {
  currentAction: number;
  uncertainty: number;
  vetoed: boolean;
  update: (data: Partial<AgentStore>) => void;
}

export const useAgentStore = create<AgentStore>((set) => ({
  currentAction: 0,
  uncertainty: 0,
  vetoed: false,
  update: (data) => set((s) => ({ ...s, ...data })),
}));
