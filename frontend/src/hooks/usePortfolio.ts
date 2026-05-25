// frontend/src/hooks/usePortfolio.ts
//
// React hook that polls /api/portfolio and exposes the most recent
// snapshot via React state.
//
// The polling interval is intentionally low (3 s) because the cost is
// dominated by the broker call inside the API — TimescaleDB is not
// touched on this path, so we don't have to worry about hammering the
// database.
//
// Errors are surfaced via the returned `error` field rather than
// thrown; the caller (the RiskPanel) renders a banner without losing
// the last good snapshot.

import { useEffect, useState } from "react";
import { apiClient } from "../api/client";
import { portfolioApi } from "../api/portfolio";
import type { Portfolio } from "../types/market.types";

const POLL_INTERVAL_MS = 3000;

interface UsePortfolioResult {
  portfolio: Portfolio | null;
  error: Error | null;
}

export function usePortfolio(): UsePortfolioResult {
  const [portfolio, setPortfolio] = useState<Portfolio | null>(null);
  const [error, setError] = useState<Error | null>(null);

  useEffect(() => {
    let cancelled = false;

    const tick = async () => {
      try {
        const response = await apiClient.get<Portfolio>("/api/portfolio");
        if (cancelled) return;
        setPortfolio(response.data);
        setError(null);
      } catch (err) {
        if (cancelled) return;
        // Preserve the last good snapshot — only update the error field
        // so the UI can render a banner while continuing to show data.
        setError(err instanceof Error ? err : new Error(String(err)));
      }
    };

    tick();
    const id = setInterval(tick, POLL_INTERVAL_MS);
    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, []);

  return { portfolio, error };
}

export function usePortfolioHistory(range: string) {
  const [history, setHistory] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    portfolioApi.getHistory(range)
      .then((data) => {
        if (!cancelled) setHistory(data.history);
      })
      .catch((err) => console.error(err))
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    
    return () => { cancelled = true; };
  }, [range]);

  return { history, loading };
}
