// frontend/src/api/portfolio.ts
import { apiClient } from "./client";
import type { Portfolio } from "../types/market.types";

interface EquityPoint {
  time: string;
  equity: number;
  benchmark?: number;
}

interface PortfolioHistoryResponse {
  history: EquityPoint[];
}

export const portfolioApi = {
  getPortfolio: () => apiClient.get<Portfolio>("/api/portfolio").then((r) => r.data),
  getHistory: (range: string) => 
    apiClient.get<PortfolioHistoryResponse>("/api/portfolio/history", { params: { range } }).then((r) => r.data),
};