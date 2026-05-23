// frontend/src/types/market.types.ts
export interface OHLCV {
  time: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

export interface Position {
  ticker: string;
  qty: number;
  avg_entry_price: number;
  unrealized_pnl: number;
  market_value: number;
}

export interface Portfolio {
  equity: number;
  cash: number;
  buying_power: number;
  positions: Position[];
  peak_equity: number;
  drawdown_pct: number;
}
