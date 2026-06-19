// frontend/src/hooks/usePortfolio.ts
//
// Polls /api/portfolio and exposes the most recent snapshot. The interval is
// intentionally low (3 s); the cost is dominated by the broker call inside the
// API (TimescaleDB is untouched on this path), so it does not hammer the DB.
//
// Errors surface via the returned `error` field rather than thrown; usePolling
// keeps the last good snapshot on failure so the RiskPanel can show a banner
// without losing data.

import { useEffect, useState } from "react";
import { portfolioApi } from "../api/portfolio";
import { usePolling } from "./usePolling";
import type { EquityPoint, Portfolio } from "../types/market.types";

const POLL_INTERVAL_MS = 3000;

interface UsePortfolioResult {
  portfolio: Portfolio | null;
  error: Error | null;
}

export function usePortfolio(): UsePortfolioResult {
  const { data: portfolio, error } = usePolling(
    (signal) => portfolioApi.getPortfolio({ signal }),
    POLL_INTERVAL_MS,
  );
  return { portfolio, error };
}

export function usePortfolioHistory(range: string) {
  const [history, setHistory] = useState<EquityPoint[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const controller = new AbortController();
    setLoading(true);
    portfolioApi
      .getHistory(range, { signal: controller.signal })
      .then((data) => {
        if (!controller.signal.aborted) setHistory(data.history);
      })
      .catch((err) => {
        if (!controller.signal.aborted) console.error(err);
      })
      .finally(() => {
        if (!controller.signal.aborted) setLoading(false);
      });

    return () => controller.abort();
  }, [range]);

  return { history, loading };
}
