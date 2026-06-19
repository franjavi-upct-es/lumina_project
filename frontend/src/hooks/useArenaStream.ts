// frontend/src/hooks/useArenaStream.ts
//
// Subscribes to the arena WebSocket and exposes the live event tail.
// Messages are JSON-shaped DecisionRecord OR DivergencePoint; we keep
// the two arrays separate and cap their sizes so the React tree stays
// responsive across long runs.

import { useCallback, useEffect, useRef, useState } from "react";
import { withWsToken, wsBase } from "../api/client";
import { useConnectionStore } from "../store/connectionSlice";
import type { DecisionRecord, DivergencePoint } from "../types/arena.types";

const MAX_DECISIONS = 1000;
const MAX_DIVERGENCES = 200;
const MAX_RETRY_ATTEMPTS = 5;
// RFC 6455 policy-violation code; the backend uses it for auth rejection
// (bad origin / token). Reconnecting can't fix auth, so we stop and surface it.
const WS_POLICY_VIOLATION = 1008;

export type ConnectionStatus = "connecting" | "open" | "closed" | "error";

interface ArenaStream {
  decisions: DecisionRecord[];
  divergences: DivergencePoint[];
  connectionStatus: ConnectionStatus;
  reconnect: () => void;
}

function isDivergence(payload: unknown): payload is DivergencePoint {
  return (
    typeof payload === "object" &&
    payload !== null &&
    "sharpe_delta" in payload &&
    "best_trajectory_id" in payload
  );
}

function isDecision(payload: unknown): payload is DecisionRecord {
  return (
    typeof payload === "object" &&
    payload !== null &&
    "trajectory_id" in payload &&
    "action_vector" in payload
  );
}

export function useArenaStream(runId: string | null): ArenaStream {
  const [decisions, setDecisions] = useState<DecisionRecord[]>([]);
  const [divergences, setDivergences] = useState<DivergencePoint[]>([]);
  const [connectionStatus, setConnectionStatus] = useState<ConnectionStatus>("closed");
  const wsRef = useRef<WebSocket | null>(null);
  const retryRef = useRef<number>(0);
  const cancelledRef = useRef<boolean>(false);

  const connect = useCallback(() => {
    if (!runId) return;
    cancelledRef.current = false;
    setConnectionStatus("connecting");
    const ws = new WebSocket(withWsToken(`${wsBase()}/arena/runs/${runId}/live`));
    wsRef.current = ws;

    ws.onopen = () => {
      retryRef.current = 0;
      useConnectionStore.getState().setAuthError(false);
      setConnectionStatus("open");
    };

    ws.onmessage = (event) => {
      try {
        const payload = JSON.parse(event.data);
        if (isDivergence(payload)) {
          setDivergences((prev) => [...prev.slice(-(MAX_DIVERGENCES - 1)), payload]);
        } else if (isDecision(payload)) {
          setDecisions((prev) => [...prev.slice(-(MAX_DECISIONS - 1)), payload]);
        }
      } catch {
        // Discard malformed frames silently.
      }
    };

    ws.onerror = () => {
      setConnectionStatus("error");
      ws.close();
    };

    ws.onclose = (event) => {
      if (cancelledRef.current) {
        setConnectionStatus("closed");
        return;
      }
      if (event.code === WS_POLICY_VIOLATION) {
        // Auth/origin rejection — retrying is futile and would loop forever.
        useConnectionStore.getState().setAuthError(true);
        setConnectionStatus("error");
        return;
      }
      if (retryRef.current >= MAX_RETRY_ATTEMPTS) {
        setConnectionStatus("error");
        return;
      }
      const delayMs = Math.min(8000, 1000 * 2 ** retryRef.current);
      retryRef.current += 1;
      setConnectionStatus("connecting");
      window.setTimeout(connect, delayMs);
    };
  }, [runId]);

  useEffect(() => {
    setDecisions([]);
    setDivergences([]);
    if (!runId) {
      setConnectionStatus("closed");
      return;
    }
    connect();
    return () => {
      cancelledRef.current = true;
      wsRef.current?.close();
    };
  }, [runId, connect]);

  const reconnect = useCallback(() => {
    retryRef.current = 0;
    wsRef.current?.close();
    connect();
  }, [connect]);

  return { decisions, divergences, connectionStatus, reconnect };
}
