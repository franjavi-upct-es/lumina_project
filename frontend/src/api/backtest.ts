// frontend/src/api/backtest.ts
import { apiClient } from "./client";
import type { BacktestRequest, BacktestResult } from "../types/backtest.types";

export const backtestApi = {
  run: (req: BacktestRequest) => apiClient.post<BacktestResult>("/api/backtest/run", req).then((r) => r.data),
  getResult: (runId: string) =>
    apiClient.get<BacktestResult>(`/api/backtest/results/${runId}`).then((r) => r.data),
  getRuns: () =>
    apiClient.get<BacktestResult[]>(`/api/backtest/runs`).then((r) => r.data),
};
