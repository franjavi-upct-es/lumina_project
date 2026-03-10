// frontend/react-app/src/hooks/useSentiment.js
// React Query hooks for multi-source sentiment data.
// Replaces SentimentMonitor.fetch_sentiment_data() from 5_Sentiment_Monitor.ipynb.

import { useQuery } from "@tanstack/react-query";
import { fetchSentiment, fetchSentimentAgg } from "../services/api";

/**
 * Fetches raw sentiment data broken down by source (news, reddit, twitter).
 */
export function useSentimentData(ticker, startDate, endDate, sources = []) {
  return useQuery({
    queryKey: ["sentiment", ticker, startDate, endDate, sources],
    queryFn: () =>
      fetchSentiment(ticker, {
        start_date: startDate,
        end_date: endDate,
        sources: sources.join(",") || undefined,
      }).then((r) => r.data),
    enabled: Boolean(ticker && startDate && endDate),
    refetchInterval: 300_000, // Refresh every 5 minutes
    staleTime: 120_000,
  });
}

/**
 * Fetches aggregate sentiment (weighted by confidence × log volume).
 */
export function useSentimentAggregate(ticker, startDate, endDate) {
  return useQuery({
    queryKey: ["sentimentAgg", ticker, startDate, endDate],
    queryFn: () =>
      fetchSentimentAgg(ticker, {
        start_date: startDate,
        end_date: endDate,
      }).then((r) => r.data),
    enabled: Boolean(ticker && startDate && endDate),
    refetchInterval: 300_000,
    staleTime: 120_000,
  });
}
