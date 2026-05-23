// frontend/src/types/backtest.types.ts
export interface BacktestRequest {
  start: string;
  end: string;
  tickers: string[];
  initial_capital: number;
}

export interface BacktestResult {
  run_id: string;
  status: "pending" | "running" | "completed" | "failed";
  sharpe?: number;
  max_drawdown?: number;
  total_return?: number;
}
