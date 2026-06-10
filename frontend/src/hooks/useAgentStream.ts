// frontend/src/hooks/useAgentStream.ts
import { useEffect, useRef, useState } from "react";
import type { AgentStreamMessage } from "../types/agent.types";
import { useAgentStore } from "../store/agentSlice";

const WS_BASE = import.meta.env.VITE_WS_BASE || "ws://localhost:8000";
const MAX_BACKOFF_MS = 30_000;

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

    function normalize(raw: unknown): AgentStreamMessage {
      if (
        raw &&
        typeof raw === "object" &&
        "type" in raw &&
        "payload" in raw &&
        "ts" in raw
      ) {
        return raw as AgentStreamMessage;
      }
      const payload = raw && typeof raw === "object" ? raw as Record<string, unknown> : {};
      return {
        type: "action",
        ts: typeof payload.ts === "string" ? payload.ts : new Date().toISOString(),
        payload,
      };
    }

    function connect() {
      if (cancelled) return;
      const ws = new WebSocket(`${WS_BASE}/api/agent/stream`);
      wsRef.current = ws;
      ws.onopen = () => { setConnected(true); backoffRef.current = 1000; };
      ws.onmessage = (e) => {
        try {
          const msg = normalize(JSON.parse(e.data));
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
        } catch {}
      };
      ws.onclose = () => {
        setConnected(false);
        if (!cancelled) {
          setTimeout(connect, backoffRef.current);
          backoffRef.current = Math.min(backoffRef.current * 2, MAX_BACKOFF_MS);
        }
      };
      ws.onerror = () => ws.close();
    }
    connect();
    return () => { cancelled = true; wsRef.current?.close(); };
  }, [updateAgent]);
  return { messages, connected };
}
