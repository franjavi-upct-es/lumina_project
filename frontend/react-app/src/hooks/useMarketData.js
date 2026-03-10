// frontend/react-app/src/hooks/useMarketData.js
// React Query hooks for market data fetching.

import { useQuery } from "@tanstack/react-query";
import { fetchPrices, fetchFeatures, fetchCompanyInfo } from "../services/api";

/**
 * Fetches OHLCV price data for a ticker.
 * Replaces the requests.get(f"{API_URL}/api/v3/data/{ticker}/prices") pattern.
 */
export function usePrices(ticker, startDate, endDate, interval = "1d") {
  return useQuery({
    queryKey: ["prices", ticker, startDate, endDate, interval],
    queryFn: () =>
      fetchPrices(ticker, {
        start_date: startDate,
        end_date: endDate,
        interval,
      }).then((r) => r.data),
    enabled: Boolean(ticker && startDate && endDate),
    staleTime: 60_000, // 1 minute - price data changes infrequently
    retry: 2,
  });
}

/**
 * Fetches engineered feature data (RSI, MACD, Bollinger Bands, etc.).
 * Replaces the feature_response.get(...) call in 1_Data_Explorer.py.
 */
export function useFeatures(ticker, startDate, endDate, categories = []) {
  return useQuery({
    queryKey: ["features", ticker, startDate, endDate, categories],
    queryFn: () =>
      fetchFeatures(ticker, {
        start_date: startDate,
        end_date: endDate,
        categories: categories.join(","),
        include_data: true,
      }).then((r) => r.data),
    enabled: Boolean(ticker && startDate && endDate && categories.length > 0),
    staleTime: 120_000, // 2 minutes – features are derived from prices
  });
}

/**
 * Fetches fundamental company information.
 * Replaces the info_response.get(f"{API_URL}/api/v2/data/{ticker}/info") call.
 */
export function useCompanyInfo(ticker) {
  return useQuery({
    queryKey: ["companyInfo", ticker],
    queryFn: () => fetchCompanyInfo(ticker).then((r) => r.data),
    enabled: Boolean(ticker),
    staleTime: 3_600_000, // 1 hour – fundamental data changes rarely
    retry: 1,
  });
}
