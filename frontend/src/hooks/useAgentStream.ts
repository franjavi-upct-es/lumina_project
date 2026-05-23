// frontend/src/hooks/useAgentStream.ts
import { useEffect, useRef, useState } from "react";
import type { AgentStreamMessage } from "../types/agent.types";

const WS_BASE = import.meta.env.VITE_WS_BASE || "ws://localhost:8000";
const MAX_BACKOFF_MS = 30_000;

export function useAgentStream() {
  const [messages, setMessages] = useState<AgentStreamMessage[]>([]);
  const [connected, setConnected] = useState(false);
  const wsRef = useRef<WebSocket | null>(null);
  const backoffRef = useRef(1000);

  useEffect(() => {
    let cancelled = false;
    function connect() {
      if (cancelled) return;
      const ws = new WebSocket(`${WS_BASE}/api/agent/stream`);
      wsRef.current = ws;
      ws.onopen = () => { setConnected(true); backoffRef.current = 1000; };
      ws.onmessage = (e) => {
        try {
          const msg: AgentStreamMessage = JSON.parse(e.data);
          setMessages((prev) => [...prev.slice(-99), msg]);
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
  }, []);
  return { messages, connected };
}
