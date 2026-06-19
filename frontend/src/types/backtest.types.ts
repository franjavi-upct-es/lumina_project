// frontend/src/types/backtest.types.ts
//
// Backtest wire types aliased from the backend-generated OpenAPI schemas
// (./api.generated.ts). Regenerate with `make openapi`.

import type { components } from "./api.generated";

export type BacktestRequest = components["schemas"]["BacktestRequest"];
export type BacktestResult = components["schemas"]["BacktestResultResponse"];
