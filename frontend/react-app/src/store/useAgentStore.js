// frontend/react-app/src/store/useAgentStore.js
// Zustand store for the V3 Chimera agent state.
// Holds live data received over WebSocket: decisions, uncertainty scores,
// circuit breaker status, and safety arbitrator override flags.

import { create } from "zustand";
import { SAFETY_STATUS } from "../constants.js";

const useAgentStore = create((set, get) => ({
  // --- Connection ---
  wsStatus: "offline", // "offline" | "connecting" | "connected" | "error"
  setWsStatus: (status) => set({ wsStatus: status }),

  // --- Live Agent Decisions ---
  // Array of the most recent agent decisions pushed over /ws/agent
  // Each entry: { timestamp, ticker, direction, urgency, sizing, stopDistance, uncertaintyScore, vetoed }
  decisions: [],
  addDecision: (decision) =>
    set((state) => ({
      decisions: [decision, ...state.decisions].slice(0, 200), // Keep last 200
    })),
  clearDecisions: () => set({ decisions: [] }),

  // --- Agent Status ---
  // Status received from GET /api/v3/agent/status
  agentStatus: {
    phase: null, // "A" | "B" | "C" | null (not training)
    mode: "offline", // "live" | "paper" | "training" | "offline"
    runningEpisode: null,
    totalSteps: 0,
    lastDecisionAt: null,
  },
  setAgentStatus: (status) =>
    set({ agentStatus: { ...get().agentStatus, ...status } }),

  // --- Uncertainty ---
  // Monte Carlo uncertainty score for the current tick (0.0 - 1.0)
  uncertainty: null,
  setUncertaintyScore: (score) => set({ uncertaintyScore: score }),

  // --- Safety Arbitrator ---
  // Current safety system status
  safetyStatus: SAFETY_STATUS.OFFLINE,
  setSafetyStatus: (status) => set({ safetyStatus: status }),

  // Active circuit breakers
  activeBreakers: [],
  setActiveBreakers: (breakers) => set({ activeBreakers: breakers }),

  // Whether the manual kill switch is engaged
  killSwitchEngaged: false,
  setKillSwitch: (engaged) => set({ killSwitchEngaged: engaged }),

  // --- Performance Snapshot ---
  // Live P&L snapshot pushed via WebSocket
  pnlSnapshot: {
    totalPnl: 0,
    dailyPnl: 0,
    openPositions: [],
    equity: null,
    drawdown: 0,
  },
  updatePnlSnapshot: (data) =>
    set((state) => ({
      pnlSnapshot: { ...state.pnlSnapshot, ...data },
    })),
}));

export default useAgentStore;
