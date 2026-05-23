// frontend/src/types/agent.types.ts
export interface AgentStatus {
  current_action: number;
  uncertainty: number;
  gate_active: boolean;
  last_update: string;
  consecutive_vetoes: number;
}

export interface AgentStreamMessage {
  type: "action" | "veto" | "liquidate" | "heartbeat";
  ts: string;
  payload: Record<string, unknown>;
}
