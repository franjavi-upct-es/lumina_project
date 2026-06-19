// frontend/src/types/risk.types.ts
//
// Risk wire types aliased from the backend-generated OpenAPI schemas
// (./api.generated.ts). Regenerate with `make openapi`.

import type { components } from "./api.generated";

// The canonical state enum is the (constrained) request field; the response
// declares `state: str` on the backend, so KillSwitchResponse.state is widened
// to `string` and callers narrow at the boundary.
export type KillSwitchState = components["schemas"]["KillSwitchRequest"]["state"];
export type KillSwitchResponse = components["schemas"]["KillSwitchResponse"];
