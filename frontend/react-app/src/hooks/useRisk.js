// frontend/react-app/src/hooks/useRisk.js
// React Query hooks for risk metrics.
// Replaces the requests.post() calls in 4_Risk_Dashboard.ipynb.

import { useQuery } from "@tanstack/react-query";
import { fetchVaR, fetchStressTest } from "../services/api";

/**
 * Calculates VaR and CVaR for a portfolio.
 * Replaces the fetch_risk_data() helper in 4_Risk_Dashboard.py.
 */
export function useVaR(
  tickers,
  weights,
  startDate,
  endDate,
  method = "historical",
) {
  return useQuery({
    queryKey: ["var", tickers, weights, startDate, endDate, method],
    queryFn: () =>
      fetchVaR({
        tickers,
        weights,
        start_date: startDate,
        end_date: endDate,
        confidence_levels: [0.95, 0.99],
        method,
        holding_period: 1,
      }).then((r) => r.data),
    enabled: Boolean(tickers?.length && startDate && endDate),
    staleTime: 300_000, // 5 minutes
  });
}

/**
 * Runs historical stress test scenarios.
 * Replaces fetch_stress_test() in 4_Risk_Dashboard.py.
 */
export function useStressTest(tickers, weights) {
  return useQuery({
    queryKey: ["stressTest", tickers, weights],
    queryFn: () =>
      fetchStressTest({ tickers, weights, include_historical: true }).then(
        (r) => r.data,
      ),
    enabled: Boolean(tickers?.length),
    staleTime: 600_000, // 10 minutes – scenarios don't change per tick
  });
}
