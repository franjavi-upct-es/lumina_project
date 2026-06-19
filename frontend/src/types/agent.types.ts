// frontend/src/types/agent.types.ts
//
// AgentStatus is aliased from the backend-generated OpenAPI schema
// (./api.generated.ts). Regenerate with `make openapi`.

import type { components } from "./api.generated";

export type AgentStatus = components["schemas"]["AgentStatusResponse"];

// AgentStreamMessage is a WebSocket envelope, NOT part of the REST OpenAPI
// surface (no route references it), so it cannot be generated and is kept in
// sync by hand with backend.api.schemas.AgentStreamMessage.
export type AgentStreamType = "action" | "veto" | "liquidate" | "heartbeat";

export interface AgentStreamMessage {
  type: AgentStreamType;
  ts: string;
  payload: Record<string, unknown>;
}
