# backend/nlp_engine/sentiment_aggregator.py
"""
Aggregate sentiment from multiple sources for comprehensive analysis
"""

from datetime import datetime
from typing import Any

import pandas as pd
from loguru import logger


class SentimentAggregator:
    """
    Aggregates sentiment scores from multiple sources

    Sources:
    - News articles
    - Social media (Reddit, Twitter)
    - Financial reports
    - Analyst reports
    """

    def __init__(self):
        """Initialize aggregator"""
        self.source_weights = {
            "news": 0.4,
            "reddit": 0.2,
            "twitter": 0.2,
            "finbert": 0.2,
        }
        logger.info("Initialized SentimentAggregator")

    def set_source_weights(self, weights: dict[str, float]):
        """
        Set custom weights for sources

        Args:
            weights: Dictionary mapping source names to weights
        """
        total = sum(weights.values())
        self.source_weights = {k: v / total for k, v in weights.items()}
        logger.info(f"Updated source weights: {self.source_weights}")

    def aggregate_by_ticker(
        self,
        sentiment_data: dict[str, pd.DataFrame],
        ticker: str,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> dict[str, Any]:
        """
        Aggregate sentiment for a specific ticker

        Args:
            sentiment_data: Dict mapping source name to DataFrame
            ticker: Stock ticker
            start_date: Start date for filtering
            end_date: End date for filtering

        Returns:
            Aggregated sentiment dictionary
        """
        if not sentiment_data:
            return self._empty_sentiment()

        logger.info(f"Aggregating sentiment for {ticker}")

        # Filter and collect sentiments from all sources
        source_sentiments = {}

        for source, df in sentiment_data.items():
            if df is None or df.empty:
                continue

            # Filter by ticker if column exists
            if "ticker" in df.columns:
                df_filtered = df[df["ticker"] == ticker].copy()
            else:
                df_filtered = df.copy()

            # Filter by date
            if start_date and "time" in df_filtered.columns:
                df_filtered = df_filtered[df_filtered["time"] >= start_date]
            if end_date and "time" in df_filtered.columns:
                df_filtered = df_filtered[df_filtered["time"] <= end_date]

            if df_filtered.empty:
                continue

            # Calculate aggregate for this source
            source_agg = self._aggregate_source(df_filtered)
            source_sentiments[source] = source_agg

        # Weighted combination
        return self._combine_sources(source_sentiments)

    def aggregate_time_series(
        self,
        sentiment_data: dict[str, pd.DataFrame],
        ticker: str,
        freq: str = "1D",
    ) -> pd.DataFrame:
        """
        Create sentiment time series

        Args:
            sentiment_data: Source data
            ticker: Stock ticker
            freq: Frequency for resampling ('1D', '1H', etc.)

        Returns:
            DataFrame with time series
        """
        all_series = []

        for source, df in sentiment_data.items():
            if df is None or df.empty:
                continue

            # Filter by ticker
            if "ticker" in df.columns:
                df = df[df["ticker"] == ticker].copy()

            if df.empty or "time" not in df.columns:
                continue

            # Set time as index
            df["time"] = pd.to_datetime(df["time"])
            df = df.set_index("time")

            # Resample
            if "sentiment_score" in df.columns:
                series = df["sentiment_score"].resample(freq).mean()
                series.name = f"{source}_sentiment"
                all_series.append(series)

        if not all_series:
            return pd.DataFrame()

        # Combine all series
        result = pd.concat(all_series, axis=1)

        # Calculate weighted average
        weights = [
            self.source_weights.get(col.replace("_snetiment", ""), 1.0) for col in result.columns
        ]
        result["aggregate_sentiment"] = result.mul(weights).sum(axis=1) / sum(weights)

        return result

    def _aggregate_source(self, df: pd.DataFrame) -> dict[str, float]:
        """Aggregate sentiment for a single source"""
        if "sentiment_score" not in df.columns:
            return {"score": 0.0, "count": 0}

        scores = df["sentiment_score"].dropna()

        if len(scores) == 0:
            return {"score": 0.0, "count": 0}

        return {
            "score": float(scores.mean()),
            "std": float(scores.std()),
            "min": float(scores.min()),
            "max": float(scores.max()),
            "count": len(scores),
            "positive_pct": float((scores > 0.5).sum() / len(scores)),
            "negative_pct": float((scores < 0.5).sum() / len(scores)),
        }

    def _combine_sources(self, source_sentiments: dict[str, dict]) -> dict[str, Any]:
        """Combine sentiment from multiple sources"""
        if not source_sentiments:
            return self._empty_sentiment()

        # Weighted average
        total_weight = 0.0
        weighted_score = 0.0
        total_count = 0

        for source, sentiment in source_sentiments.items():
            weight = self.source_weights.get(source, 1.0)
            weighted_score += sentiment["score"] * weight
            total_weight += weight
            total_count += sentiment["count"]

        aggregate_score = weighted_score / total_weight if total_weight > 0 else 0.0

        # Categorize sentiment
        if aggregate_score > 0.6:
            sentiment_label = "positive"
        elif aggregate_score < 0.4:
            sentiment_label = ("negative",)
        else:
            sentiment_label = "neutral"

        return {
            "aggregate_score": aggregate_score,
            "sentiment": sentiment_label,
            "sources": source_sentiments,
            "total_mentions": total_count,
            "num_sources": len(source_sentiments),
        }

    def _empty_sentiment(self) -> dict[str, Any]:
        """Return empty sentiment"""
        return {
            "aggregate_score": 0.5,
            "sentiment": "neutral",
            "sources": {},
            "total_mentions": 0,
            "num_sources": 0,
        }

    def compare_tickers(
        self,
        sentiment_data: dict[str, pd.DataFrame],
        tickers: list[str],
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> pd.DataFrame:
        """
        Compare sentiment across multiple tickers

        Args:
            sentiment_data: Source data
            tickers: List of tickers to compare
            start_date: Start date
            end_date: End date

        Returns:
            DataFrame with comparison
        """
        results = []

        for ticker in tickers:
            agg = self.aggregate_by_ticker(sentiment_data, ticker, start_date, end_date)
            results.append({"ticker": ticker, **agg})

        df = pd.DataFrame(results)
        df = df.sort_values("aggregate_score", ascending=False)

        return df

