// frontend/src/api/portfolio.ts
import { apiClient, type RequestOpts } from "./client";
import type { Portfolio, PortfolioHistoryResponse } from "../types/market.types";

export const portfolioApi = {
  getPortfolio: (opts?: RequestOpts) =>
    apiClient.get<Portfolio>("/api/portfolio", opts).then((r) => r.data),
  getHistory: (range: string, opts?: RequestOpts) =>
    apiClient
      .get<PortfolioHistoryResponse>("/api/portfolio/history", { params: { range }, ...opts })
      .then((r) => r.data),
};
