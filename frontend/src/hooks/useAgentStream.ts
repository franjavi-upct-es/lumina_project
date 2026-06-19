// frontend/src/hooks/useAgentStream.ts
import { useEffect, useRef, useState } from "react";
import { AGENT_STREAM_URL } from "../api/agent";
import { useConnectionStore } from "../store/connectionSlice";
import type { AgentStreamMessage, AgentStreamType } from "../types/agent.types";
import { useAgentStore } from "../store/agentSlice";

const MAX_BACKOFF_MS = 30_000;
// RFC 6455 policy-violation code; the backend uses it for auth rejection
// (bad origin / token). Reconnecting can't fix auth, so we stop and surface it.
const WS_POLICY_VIOLATION = 1008;

const STREAM_TYPES: readonly AgentStreamType[] = ["action", "veto", "liquidate", "heartbeat"];

function isStreamType(v: unknown): v is AgentStreamType {
  return typeof v === "string" && (STREAM_TYPES as readonly string[]).includes(v);
}

export function useAgentStream() {
  const [messages, setMessages] = useState<AgentStreamMessage[]>([]);
  const [connected, setConnected] = useState(false);
  const wsRef = useRef<WebSocket | null>(null);
  const backoffRef = useRef(1000);
  const updateAgent = useAgentStore((s) => s.update);

  useEffect(() => {
    let cancelled = false;

    function scalarAction(payload: Record<string, unknown>): number | null {
      if (typeof payload.final_action === "number") return payload.final_action;
      if (typeof payload.action === "number") return payload.action;
      if (Array.isArray(payload.action) && typeof payload.action[0] === "number") {
        return payload.action[0];
      }
      return null;
    }

    // Classify by the actual `type` discriminator. `payload` is optional on the
    // wire (heartbeats carry none), so we default it to {} rather than
    // demanding its presence — the old code required `payload` and therefore
    // mis-coerced heartbeats into "action" frames.
    function normalize(raw: unknown): AgentStreamMessage {
      const obj = raw && typeof raw === "object" ? (raw as Record<string, unknown>) : {};
      const ts = typeof obj.ts === "string" ? obj.ts : new Date().toISOString();
      const payload =
        obj.payload && typeof obj.payload === "object"
          ? (obj.payload as Record<string, unknown>)
          : {};
      if (isStreamType(obj.type)) {
        return { type: obj.type, ts, payload };
      }
      // Legacy/unwrapped frame (bare payload): treat as an action.
      return { type: "action", ts, payload: obj };
    }

    function connect() {
      if (cancelled) return;
      const ws = new WebSocket(AGENT_STREAM_URL);
      wsRef.current = ws;
      ws.onopen = () => {
        setConnected(true);
        backoffRef.current = 1000;
        const conn = useConnectionStore.getState();
        conn.setStreamConnected(true);
        conn.setAuthError(false);
      };
      ws.onmessage = (e) => {
        try {
          const msg = normalize(JSON.parse(e.data));
          if (msg.type === "heartbeat") {
            // Keep-alive only: confirms liveness, carries no event to record.
            setConnected(true);
            return;
          }
          setMessages((prev) => [...prev.slice(-99), msg]);
          if (msg.type === "action" || msg.type === "veto") {
            const action = scalarAction(msg.payload);
            updateAgent({
              ...(action === null ? {} : { currentAction: action }),
              ...(typeof msg.payload.uncertainty === "number"
                ? { uncertainty: msg.payload.uncertainty }
                : {}),
              ...(typeof msg.payload.vetoed === "boolean"
                ? { vetoed: msg.payload.vetoed }
                : {}),
            });
          }
        } catch {
          // Discard malformed frames silently.
        }
      };
      ws.onclose = (event) => {
        setConnected(false);
        useConnectionStore.getState().setStreamConnected(false);
        if (cancelled) return;
        if (event.code === WS_POLICY_VIOLATION) {
          // Auth/origin rejection — retrying is futile and would loop forever.
          useConnectionStore.getState().setAuthError(true);
          return;
        }
        setTimeout(connect, backoffRef.current);
        backoffRef.current = Math.min(backoffRef.current * 2, MAX_BACKOFF_MS);
      };
      ws.onerror = () => ws.close();
    }
    connect();
    return () => {
      cancelled = true;
      wsRef.current?.close();
    };
  }, [updateAgent]);
  return { messages, connected };
}
