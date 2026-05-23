// frontend/src/api/data.ts
import { apiClient } from "./client";
import type { OHLCV } from "../types/market.types";

export const dataApi = {
  getOhlcv: (ticker: string, start: string, end: string) =>
    apiClient.get<OHLCV[]>(`/api/data/ohlcv/${ticker}`, { params: { start, end } }).then((r) => r.data),
  getEmbedding: (kind: "price" | "semantic" | "graph", ticker: string) =>
    apiClient.get<{ ticker: string; kind: string; vector: number[] | null }>(
      `/api/data/embedding/${kind}/${ticker}`,
    ).then((r) => r.data),
};
