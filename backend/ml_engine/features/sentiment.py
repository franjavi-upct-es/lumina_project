# backend/ml_engine/features/sentiment.py
"""
Sentiment analysis features from news and social media
Market sentiment indicators for ML models
"""

import numpy as np
import pandas as pd
from loguru import logger


class SentimentFeatures:
    """
    Sentiment analysis features

    Sources:
    - News articles
    - Social media (Reddit, Twitter)
    - Analyst ratings
    - Market sentiment indicators
    """

    def __init__(self):
        self.feature_names = []

    def create_sentiment_features(
        self,
        sentiment_data: pd.DataFrame,
        aggregation_windows: list[int] = [1, 3, 7, 14, 30],
    ) -> pd.DataFrame:
        """
        Create sentiment features from raw sentiment data

        Args:
            sentiment_data: DataFrame with columns [time, sentiment_score, volume, source]
            aggregation_windows: Windows for rolling aggregations (days)

        Returns:
            DataFrame with sentiment features
        """
        df = sentiment_data.copy()

        if df.empty:
            logger.warning("Empty sentiment data provided")
            return df

        # Ensure time column is datetime
        if "time" in df.columns:
            df["time"] = pd.to_datetime(df["time"])
            df = df.sort_values("time")

        # Basic sentiment features
        df = self._add_basic_sentiment(df)

        # Rolling aggregations
        for window in aggregation_windows:
            df = self._add_rolling_sentiment(df, window)

        # Source-specific features
        if "source" in df.columns:
            df = self._add_source_features(df)

        # Sentiment momentum
        df = self._add_sentiment_momentum(df)

        # Volume-weighted sentiment
        if "volume" in df.columns:
            df = self._add_volume_weighted_sentiment(df)

        logger.info(f"Created {len(self.feature_names)} sentiment features")
        return df

    def _add_basic_sentiment(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add basic sentiment metrics"""
        if "sentiment_score" in df.columns:
            # Normalized sentiment (-1 to 1)
            df["sentiment_normalized"] = df["sentiment_score"].clip(-1, 1)

            # Binary snetiment
            df["sentiment_positive"] = (df["sentiment_score"] > 0).astype(int)
            df["sentiment_negative"] = (df["sentiment_score"] < 0).astype(int)
            df["sentiment_neutral"] = (df["sentiment_score"] == 0).astype(int)

            # Sentiment magnitude
            df["sentiment_magnitude"] = df["sentiment_score"].abs()

            self.feature_names.extend(
                [
                    "sentiment_normalized",
                    "sentiment_positive",
                    "sentiment_negative",
                    "sentiment_neutral",
                    "sentiment_magnitude",
                ]
            )

        return df

    def _add_rolling_sentiment(self, df: pd.DataFrame, window: int) -> pd.DataFrame:
        """Add rolling sentiment aggregations"""
        if "sentiment_score" not in df.columns:
            return df

        # Mean sentiment
        df[f"sentiment_mean_{window}d"] = df["sentiment_score"].rolling(window).mean()

        # Sentiment std
        df[f"sentiment_std_{window}d"] = df["sentiment_score"].rolling(window).std()

        # Sentiment trend
        df[f"sentiment_trend_{window}d"] = df["sentiment_score"] - df["sentiment_score"].shift(
            window
        )

        # Positive ratio
        if "sentiment_positive" in df.columns:
            df[f"positive_ratio_{window}d"] = df["sentiment_positive"].rolling(window).mean()

        self.feature_names.extend(
            [
                f"sentiment_mean_{window}d",
                f"sentiment_std_{window}d",
                f"sentiment_trend_{window}d",
                f"positive_ratio_{window}d",
            ]
        )

        return df

    def _add_source_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add Source-specific features"""
        sources = df["source"].unique()

        for source in sources:
            source_mask = df["source"] == source
            source_col = f"sentiment_{source}"

            df[source_col] = np.where(source_mask, df["sentiment_score"], np.nan)

            # Forward fill source sentiment
            df[f"{source_col}_ffill"] = df[source_col].fillna(method="ffill")

            self.feature_names.extend([source_col, f"{source_col}_ffill"])

        return df

    def _add_sentiment_momentum(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add sentiment momentum indicators"""
        if "sentiment_score" not in df.columns:
            return df

        # Rate of change
        for period in [3, 7, 24]:
            df[f"sentiment_roc_{period}d"] = df["sentiment_score"].pct_change(period) * 100
            self.feature_names.append(f"sentiment_roc_{period}d")

        # Sentiment acceleration
        df["sentiment_acceleration"] = df["sentiment_score"].diff().diff()
        self.feature_names.append("sentiment_acceleration")

        # Sentiment reversal indicator
        df["sentiment_reversal"] = (
            (df["sentiment_score"].shift(1) > 0) & (df["sentiment_score"] < 0)
            | (df["sentiment_score"].shift(1) < 0) & (df["sentiment_score"] > 0)
        ).astype(int)
        self.feature_names.append("sentiment_reversal")

        return df

    def _add_volume_weighted_sentiment(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volume-weighted sentiment"""
        if "volume" not in df.columns or "sentiment_score" not in df.columns:
            return df

        # Volume-weighted sentiment
        df["sentiment_volume_weighted"] = (df["sentiment_score"] * df["volume"]) / df[
            "volume"
        ].rolling(7).sum()

        # High volume sentiment
        volume_threshold = df["volume"].quantile(0.75)
        df["high_volume_sentiment"] = np.where(
            df["volume"] > volume_threshold, df["sentiment_score"], np.nan
        )

        self.feature_names.extend(["sentiment_volume_weighted", "high_volume_sentiment"])

        return df

    def create_social_sentiment_features(
        self,
        reddit_data: pd.DataFrame | None = None,
        twitter_data: pd.DataFrame | None = None,
    ) -> dict[str, float]:
        """
        Create social media sentiment features

        Args:
            reddit_data: Reddit sentiment data
            twitter_data: Twitter sentiment data

        Returns:
            Dictionary of social sentiment features
        """
        features = {}

        if reddit_data is not None and not reddit_data.empty:
            features.update(self._extract_reddit_features(reddit_data))

        if twitter_data is not None and not twitter_data.empty:
            features.update(self._extract_twitter_features(twitter_data))

        return features

    def _extract_reddit_features(self, reddit_data: pd.DataFrame) -> dict[str, float]:
        """Extract Reddit-specific features"""
        return {
            "reddit_sentiment_mean": float(reddit_data["sentiment_score"].mean()),
            "reddit_sentiment_std": float(reddit_data["sentiment_score"].std()),
            "reddit_volume": float(len(reddit_data)),
            "reddit_upvote_ratio": float(reddit_data.get("upvote_ratio", pd.Series([0.5])).mean()),
            "reddit_comment_volume": float(reddit_data.get("num_comments", pd.Series([0])).sum()),
        }

    def _extract_twitter_features(self, twitter_data: pd.DataFrame) -> dict[str, float]:
        """Extract Twitter-specific features"""
        return {
            "twitter_sentiment_mean": float(twitter_data["sentiment_score"].mean()),
            "twitter_sentiment_std": float(twitter_data["sentiment_score"].std()),
            "twitter_volume": float(len(twitter_data)),
            "twitter_retweet_ratio": float(
                twitter_data.get("retweet_count", pd.Series([0])).mean()
            ),
            "twitter_engagement": float(
                twitter_data.get("like_count", pd.Series([0])).sum()
                + twitter_data.get("retweet_count", pd.Series([0])).sum()
            ),
        }

    def calculate_fear_greed_index(
        self, sentiment_features: dict[str, float], market_data: dict[str, float] | None = None
    ) -> float:
        """
        Calculate Fear & Greed Index (0-100)

        Args:
            sentiment_features: Sentiment features
            market_data: Optional market momentum data

        Returns:
            Fear & Greed score (0=Extreme Fear, 100=Extreme Greed)
        """
        score = 50.0  # Neutral

        # Sentiment component (40% weight)
        sentiment = sentiment_features.get("sentiment_mean_7d", 0)
        sentiment_score = (sentiment + 1) * 20  # Convert -1, 1 to 0.40
        score += sentiment_score * 0.4

        # Volume component (20% weight)
        if "positive_ratio_7d" in sentiment_features:
            positive_ratio = sentiment_features["positive_ratio_7d"]
            score += positive_ratio * 20 * 0.2

        # Market momentum (40% weight)
        if market_data:
            momentum = market_data.get("momentum_7d", 0)
            momentum_score = (momentum + 1) * 20  # Convert -1,1 to 0.40
            score += momentum_score * 0.4

        return max(0, min(100, score))

    def detect_sentiment_extremes(
        self, df: pd.DataFrame, threshold_std: float = 2.0
    ) -> pd.DataFrame:
        """
        Detect extreme sentiment evnets

        Args:
            df: DataFrame with sentiment data
            threshold_std: Standard deviation threshold

        Returns:
            DataFrame with extreme event indicators
        """
        if "sentiment_score" not in df.columns:
            return df

        mean = df["sentiment_score"].rolling(30).mean()
        std = df["sentiment_score"].rolling(30).std()

        df["sentiment_extreme_positive"] = (
            df["sentiment_score"] > mean + threshold_std * std
        ).astype(int)

        df["sentiment_extreme_negative"] = (
            df["sentiment_score"] < mean - threshold_std * std
        ).astype(int)

        self.feature_names.extend(["sentiment_extreme_positive", "sentiment_extreme_negative"])

        return df

    def get_feature_names(self) -> list[str]:
        """Get list of all feature names"""
        return self.feature_names
