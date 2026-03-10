// frontend/react-app/src/hooks/useAgent.js
// React Query hooks for V3 Chimera agent data (NEW in V3).
// Combines REST polling for status with WebSocket-driven live updates.

import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { useEffect } from "react";
import {
  fetchAgentStatus,
  fetchAgentDecisions,
  fetchUncertainty,
  fetchSafetyStatus,
  fetchCircuitBreakers,
  triggerKillSwitch,
} from "../services/api";
import { createWebSocket } from "../services/websocket";
import useAgentStore from "../store/useAgentStore";
import { WS_CHANNELS } from "../constants";

/** Polls the agent status endpoint every 5 seconds. */
export function useAgentStatus() {
  const setAgentStatus = useAgentStore((s) => s.setAgentStatus);
  return useQuery({
    queryKey: ["agentStatus"],
    queryFn: () => fetchAgentStatus().then((r) => r.data),
    refetchInterval: 5_000,
    onSuccess: (data) => setAgentStatus(data),
  });
}

/** Fetches the last N agent decisions with uncertainty scores. */
export function useAgentDecisions(limit = 50) {
  return useQuery({
    queryKey: ["agentDecisions", limit],
    queryFn: () => fetchAgentDecisions({ limit }).then((r) => r.data),
    refetchInterval: 10_000,
  });
}

/** Fetches Monte Carlo uncertainty estimate for a specific ticker. */
export function useUncertainty(ticker) {
  return useQuery({
    queryKey: ["uncertainty", ticker],
    queryFn: () => fetchUncertainty(ticker).then((r) => r.data),
    enabled: Boolean(ticker),
    refetchInterval: 5_000,
  });
}

/** Polls the Safety Arbitrator status every 3 seconds. */
export function useSafetyStatus() {
  const setSafetyStatus = useAgentStore((s) => s.setSafetyStatus);
  const setActiveBreakers = useAgentStore((s) => s.setActiveBreakers);
  return useQuery({
    queryKey: ["safetyStatus"],
    queryFn: () => fetchSafetyStatus().then((r) => r.data),
    refetchInterval: 3_000,
    onSuccess: (data) => {
      setSafetyStatus(data.status);
      setActiveBreakers(data.active_breakers || []);
    },
  });
}

/** Polls the circuit breaker states. */
export function useCircuitBreakers() {
  return useQuery({
    queryKey: ["circuitBreakers"],
    queryFn: () => fetchCircuitBreakers().then((r) => r.data),
    refetchInterval: 3_000,
  });
}

/** Mutation to trigger the emergency kill switch. */
export function useKillSwitch() {
  const qc = useQueryClient();
  const setKillSwitch = useAgentStore((s) => s.setKillSwitch);
  return useMutation({
    mutationFn: triggerKillSwitch,
    onSuccess: () => {
      setKillSwitch(true);
      qc.invalidateQueries({ queryKey: ["safetyStatus"] });
    },
  });
}

/**
 * Opens the live WebSocket channel for agent decisions.
 * Pushes each incoming decision into the Zustand store.
 * Must be called from a component that is always mounted (e.g. Layout).
 */
export function useAgentWebSocket() {
  const addDecision = useAgentStore((s) => s.addDecision);
  const setWsStatus = useAgentStore((s) => s.setWsStatus);
  const updatePnlSnapshot = useAgentStore((s) => s.updatePnlSnapshot);

  useEffect(() => {
    const cleanup = createWebSocket(
      WS_CHANNELS.AGENT_DECISIONS,
      (msg) => {
        if (msg.type === "decision") addDecision(msg.data);
        if (msg.type === "pnl") updatePnlSnapshot(msg.data);
      },
      setWsStatus,
    );
    return cleanup;
  }, [addDecision, setWsStatus, updatePnlSnapshot]);
}
