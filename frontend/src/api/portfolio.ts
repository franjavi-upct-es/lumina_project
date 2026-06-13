// frontend/src/api/portfolio.ts
import { apiClient } from "./client";
import type { Portfolio, PortfolioHistoryResponse } from "../types/market.types";

export const portfolioApi = {
  getPortfolio: () => apiClient.get<Portfolio>("/api/portfolio").then((r) => r.data),
  getHistory: (range: string) =>
    apiClient.get<PortfolioHistoryResponse>("/api/portfolio/history", { params: { range } }).then((r) => r.data),
};