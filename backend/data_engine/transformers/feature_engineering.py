# backend/data_engine/transformers/feature_engineering.py
"""
Feature Engineering for V3
==========================

Creates technical indicators and derived features for the perception encoders.
Aligned with V3 feature taxonomy (Temporal, Semantic, Structural, Macro).

Version: 3.0.0
"""

import polars as pl
from loguru import logger


class FeatureEngineer:
    """
    Feature engineering pipeline for V3

    Creates features categorized by encoder target:
    - Temporal: Price/volume indicators for TFT
    - Momentum: Trend indicators for TFT
    - Volatility: Risk measures for TFT
    """

    def __init__(self):
        """Initialize feature engineer"""
        logger.info("FeatureEngineer initialized")

    def create_all_features(
        self,
        data: pl.DataFrame,
        include_temporal: bool = True,
        include_volume: bool = True,
        include_volatility: bool = True,
    ) -> pl.DataFrame:
        """
        Create all features for a dataset

        Args:
            data: DataFrame with OHLCV data
            include_temporal: Add temporal features
            include_volume: Add volume features
            include_volatility: Add volatility features

        Returns:
            DataFrame with added features
        """
        df = data.clone()

        try:
            if include_temporal:
                df = self.add_temporal_features(df)

            if include_volume:
                df = self.add_volume_features(df)

            if include_volatility:
                df = self.add_volatility_features(df)

            logger.success(f"Created features: {len(df.columns)} columns")
            return df

        except Exception as e:
            logger.error(f"Error creating features: {e}")
            return data

    def add_temporal_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Add temporal momentum/trend features"""
        try:
            # RSI
            df = df.with_columns(
                [
                    self._calculate_rsi(df["close"], 14).alias("rsi_14"),
                    self._calculate_rsi(df["close"], 28).alias("rsi_28"),
                ]
            )

            # Moving averages
            df = df.with_columns(
                [
                    pl.col("close").rolling_mean(window_size=20).alias("sma_20"),
                    pl.col("close").rolling_mean(window_size=50).alias("sma_50"),
                    pl.col("close").rolling_mean(window_size=200).alias("sma_200"),
                ]
            )

            # MACD
            ema_12 = pl.col("close").ewm_mean(span=12)
            ema_26 = pl.col("close").ewm_mean(span=26)
            macd = ema_12 - ema_26
            signal = macd.ewm_mean(span=9)

            df = df.with_columns(
                [
                    macd.alias("macd"),
                    signal.alias("macd_signal"),
                    (macd - signal).alias("macd_hist"),
                ]
            )

            logger.debug("Added temporal features")
            return df

        except Exception as e:
            logger.error(f"Error adding temporal features: {e}")
            return df

    def add_volume_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Add volume-based features"""
        try:
            df = df.with_columns(
                [
                    pl.col("volume").rolling_mean(window_size=20).alias("volume_sma_20"),
                    (pl.col("volume") / pl.col("volume").rolling_mean(window_size=20)).alias(
                        "volume_ratio"
                    ),
                    ((pl.col("close") - pl.col("open")) * pl.col("volume") / 1000000).alias(
                        "money_flow"
                    ),
                ]
            )

            logger.debug("Added volume features")
            return df

        except Exception as e:
            logger.error(f"Error adding volume features: {e}")
            return df

    def add_volatility_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Add volatility/risk features"""
        try:
            # ATR
            tr = pl.max_horizontal(
                [
                    pl.col("high") - pl.col("low"),
                    (pl.col("high") - pl.col("close").shift(1)).abs(),
                    (pl.col("low") - pl.col("close").shift(1)).abs(),
                ]
            )

            df = df.with_columns(
                [
                    tr.rolling_mean(window_size=14).alias("atr_14"),
                ]
            )

            # Bollinger Bands
            sma_20 = pl.col("close").rolling_mean(window_size=20)
            std_20 = pl.col("close").rolling_std(window_size=20)

            df = df.with_columns(
                [
                    (sma_20 + 2 * std_20).alias("bb_upper"),
                    (sma_20 - 2 * std_20).alias("bb_lower"),
                    ((pl.col("close") - (sma_20 - 2 * std_20)) / (4 * std_20)).alias("bb_width"),
                ]
            )

            # Historical volatility
            returns = pl.col("close").pct_change()
            df = df.with_columns(
                [
                    returns.rolling_std(window_size=20).alias("volatility_20d"),
                ]
            )

            logger.debug("Added volatility features")
            return df

        except Exception as e:
            logger.error(f"Error adding volatility features: {e}")
            return df

    @staticmethod
    def _calculate_rsi(series: pl.Series, period: int = 14) -> pl.Series:
        """Calculate RSI indicator"""
        delta = series.diff()

        gain = delta.clip(lower_bound=0)
        loss = (-delta).clip(lower_bound=0)

        avg_gain = gain.rolling_mean(window_size=period)
        avg_loss = loss.rolling_mean(window_size=period)

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        return rsi


# Global instance
_engineer: FeatureEngineer | None = None


def get_feature_engineer() -> FeatureEngineer:
    """Get global feature engineer"""
    global _engineer
    if _engineer is None:
        _engineer = FeatureEngineer()
    return _engineer
