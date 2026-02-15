# backend/data_engine/transformers/regime_detection.py
"""
Market Regime Detection for V3
==============================

Detects market regimes (Bull, Bear, High Vol, Low Vol, Crash, etc.)
Used as context features for the perception layer.

Version: 3.0.0
"""

from enum import Enum

import polars as pl
from loguru import logger


class MarketRegime(str, Enum):
    """Market regime classifications"""

    BULL_LOW_VOL = "bull_low_vol"
    BULL_HIGH_VOL = "bull_high_vol"
    BEAR_LOW_VOL = "bear_low_vol"
    BEAR_HIGH_VOL = "bear_high_vol"
    SIDEWAYS = "sideways"
    CRASH = "crash"
    RECOVERY = "recovery"


class RegimeDetector:
    """
    Market regime detection

    Classifies market state based on trend and volatility.
    """

    def __init__(
        self,
        trend_window: int = 50,
        vol_window: int = 20,
        crash_threshold: float = -0.10,  # -10% drop
    ):
        """
        Initialize regime detector

        Args:
            trend_window: Window for trend calculation
            vol_window: Window for volatility calculation
            crash_threshold: Threshold for crash detection
        """
        self.trend_window = trend_window
        self.vol_window = vol_window
        self.crash_threshold = crash_threshold

        logger.info("RegimeDetector initialized")

    def detect_regimes(self, data: pl.DataFrame) -> pl.DataFrame:
        """
        Detect market regimes

        Args:
            data: DataFrame with price data

        Returns:
            DataFrame with regime column
        """
        try:
            df = data.clone()

            # Calculate trend
            sma = pl.col("close").rolling_mean(window_size=self.trend_window)
            trend = pl.col("close") / sma - 1.0

            # Calculate volatility
            returns = pl.col("close").pct_change()
            volatility = returns.rolling_std(window_size=self.vol_window)
            vol_median = volatility.median()

            # Detect crash (large drawdown)
            max_close = pl.col("close").rolling_max(window_size=self.vol_window)
            drawdown = pl.col("close") / max_close - 1.0

            df = df.with_columns(
                [
                    trend.alias("_trend"),
                    volatility.alias("_volatility"),
                    drawdown.alias("_drawdown"),
                ]
            )

            # Classify regime
            def classify_regime(row):
                if row["_drawdown"] < self.crash_threshold:
                    return MarketRegime.CRASH.value
                elif row["_trend"] > 0.02:  # Uptrend
                    if row["_volatility"] > vol_median:
                        return MarketRegime.BULL_HIGH_VOL.value
                    else:
                        return MarketRegime.BULL_LOW_VOL.value
                elif row["_trend"] < -0.02:  # Downtrend
                    if row["_volatility"] > vol_median:
                        return MarketRegime.BEAR_HIGH_VOL.value
                    else:
                        return MarketRegime.BEAR_LOW_VOL.value
                else:
                    return MarketRegime.SIDEWAYS.value

            # Apply classification
            df = df.with_columns(
                [
                    pl.struct(["_trend", "_volatility", "_drawdown"])
                    .map_elements(classify_regime, return_dtype=pl.Utf8)
                    .alias("market_regime")
                ]
            )

            # Drop temporary columns
            df = df.drop(["_trend", "_volatility", "_drawdown"])

            logger.success("Detected market regimes")
            return df

        except Exception as e:
            logger.error(f"Error detecting regimes: {e}")
            return data


# Global instance
_detector: RegimeDetector | None = None


def get_regime_detector() -> RegimeDetector:
    """Get global regime detector"""
    global _detector
    if _detector is None:
        _detector = RegimeDetector()
    return _detector
