// frontend/src/types/market.types.ts
//
// Market/portfolio wire types. These are ALIASES of the backend-generated
// OpenAPI schemas (./api.generated.ts) so they cannot drift from
// backend.api.schemas. Regenerate with `make openapi`.

import type { components } from "./api.generated";

type S = components["schemas"];

export type OHLCV = S["OHLCVResponse"];
export type Position = S["PositionResponse"];
export type Portfolio = S["PortfolioResponse"];
export type EquityPoint = S["EquityPoint"];
export type PortfolioHistoryResponse = S["PortfolioHistoryResponse"];
