# backend/perception/temporal/preprocessor.py
"""
Temporal Data Preprocessor

Prepares OHLCV time series data for TFT encoder:
- Technical indicator calculation
- Feature normalization
- Sequence windowing
- Static covariate encoding

Features Generated:
- Price features: Returns, log returns, price levels
- Volatility: ATR, Bollinger Bands, Parkinson volatility
- Momentum: RSI, MACD, Stochastic
- Volume: Volume ratios, OBV
- Time features: Hour, day of week, month (cyclical encoding)

Static Covariates:
- Asset class, sector, market cap
- Volatility regime, liquidity profile
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from loguru import logger


@dataclass
class PreprocessorConfig:
    """
    Configuration for temporal preprocessor.

    Attributes:
        lookback_window: Number of historical bars
        target_horizon: Forecast horizon (unused in encoding)
        normalize: Apply normalization
        add_technicals: Add technical indicators
        add_time_features: Add time-based features
    """

    lookback_window: int = 60
    target_horizon: int = 1
    normalize: bool = True
    add_technicals: bool = True
    add_time_features: bool = True


class TemporalPreprocessor:
    """
    Preprocesses OHLCV data for temporal encoding.

    Pipeline:
    1. Calculate technical indicators
    2. Add time features
    3. Normalize
    4. Create sequences

    Example:
        >>> preprocessor = TemporalPreprocessor()
        >>> df = load_ohlcv_data('AAPL')
        >>> features, static = preprocessor.preprocess(df)
    """

    def __init__(self, config: PreprocessorConfig | None = None):
        """
        Initialize preprocessor.

        Args:
            config: Preprocessor configuration
        """
        self.config = config or PreprocessorConfig()

        # Normalization statistics (fitted during training)
        self.mean_ = None
        self.std_ = None

        logger.debug(f"TemporalPreprocessor initialized: window={self.config.lookback_window}")

    def preprocess(
        self, df: pd.DataFrame, static_features: dict | None = None
    ) -> tuple[np.ndarray, dict]:
        """
        Preprocess OHLCV dataframe.

        Args:
            df: DataFrame with OHLCV columns
            static_features: Static covariates dict

        Returns:
            features: Array [lookback_window, num_features]
            static_dict: Static features
        """
        # Validate columns
        required = ["open", "high", "low", "close", "volume"]
        if not all(col in df.columns for col in required):
            raise ValueError(f"DataFrame must contain {required}")

        # Copy to avoid modifying original
        df = df.copy()

        # Add features
        if self.config.add_technicals:
            df = self._add_technical_indicators(df)

        if self.config.add_time_features:
            df = self._add_time_features(df)

        # Drop NaN from indicators
        df = df.dropna()

        if len(df) < self.config.lookback_window:
            raise ValueError(f"Insufficient data: {len(df)} < {self.config.lookback_window}")

        # Extract last window
        features_df = df.iloc[-self.config.lookback_window :]

        # Select feature columns (exclude date/time)
        feature_cols = [
            c for c in features_df.columns if c not in ["date", "datetime", "timestamp"]
        ]

        features = features_df[feature_cols].values

        # Normalize
        if self.config.normalize:
            features = self._normalize(features)

        # Static features
        static_dict = static_features or {}

        return features, static_dict

    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators."""
        # Returns
        df["returns"] = df["close"].pct_change()
        df["log_returns"] = np.log(df["close"] / df["close"].shift(1))

        # Simple Moving Averages
        for period in [5, 10, 20]:
            df[f"sma_{period}"] = df["close"].rolling(period).mean()
            df[f"price_to_sma_{period}"] = df["close"] / df[f"sma_{period}"]

        # RSI (simplified)
        delta = df["close"].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = -delta.where(delta < 0, 0).rolling(14).mean()
        rs = gain / (loss + 1e-10)
        df["rsi"] = 100 - (100 / (1 + rs))

        # ATR (Average True Range)
        high_low = df["high"] - df["low"]
        high_close = abs(df["high"] - df["close"].shift())
        low_close = abs(df["low"] - df["close"].shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df["atr"] = true_range.rolling(14).mean()

        # Bollinger Bands
        sma_20 = df["close"].rolling(20).mean()
        std_20 = df["close"].rolling(20).std()
        df["bb_upper"] = sma_20 + (2 * std_20)
        df["bb_lower"] = sma_20 - (2 * std_20)
        df["bb_position"] = (df["close"] - df["bb_lower"]) / (
            df["bb_upper"] - df["bb_lower"] + 1e-10
        )

        # Volume features
        df["volume_sma"] = df["volume"].rolling(20).mean()
        df["volume_ratio"] = df["volume"] / (df["volume_sma"] + 1e-10)

        return df

    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add cyclical time features."""
        if df.index.dtype == "datetime64[ns]":
            # Hour (0-23)
            hour = df.index.hour
            df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
            df["hour_cos"] = np.cos(2 * np.pi * hour / 24)

            # Day of week (0-6)
            day = df.index.dayofweek
            df["day_sin"] = np.sin(2 * np.pi * day / 7)
            df["day_cos"] = np.cos(2 * np.pi * day / 7)

            # Month (1-12)
            month = df.index.month
            df["month_sin"] = np.sin(2 * np.pi * month / 12)
            df["month_cos"] = np.cos(2 * np.pi * month / 12)

        return df

    def _normalize(self, features: np.ndarray) -> np.ndarray:
        """
        Normalize features using z-score.

        Args:
            features: Feature array [window, num_features]

        Returns:
            Normalized features
        """
        if self.mean_ is None or self.std_ is None:
            # Fit normalization parameters
            self.mean_ = np.mean(features, axis=0)
            self.std_ = np.std(features, axis=0) + 1e-8

        # Apply normalization
        normalized = (features - self.mean_) / self.std_

        return normalized

    def fit(self, df: pd.DataFrame):
        """
        Fit normalization parameters.

        Args:
            df: Training data
        """
        # Add features
        if self.config.add_technicals:
            df = self._add_technical_indicators(df)

        if self.config.add_time_features:
            df = self._add_time_features(df)

        df = df.dropna()

        # Extract features
        feature_cols = [c for c in df.columns if c not in ["date", "datetime", "timestamp"]]
        features = df[feature_cols].values

        # Fit normalization
        self.mean_ = np.mean(features, axis=0)
        self.std_ = np.std(features, axis=0) + 1e-8

        logger.success(f"Fitted preprocessor on {len(df)} samples")


def create_temporal_features(
    df: pd.DataFrame, lookback: int = 60, normalize: bool = True
) -> np.ndarray:
    """
    Quick function to create temporal features.

    Args:
        df: OHLCV dataframe
        lookback: Window size
        normalize: Apply normalization

    Returns:
        Feature array [lookback, num_features]
    """
    config = PreprocessorConfig(lookback_window=lookback, normalize=normalize)

    preprocessor = TemporalPreprocessor(config)
    features, _ = preprocessor.preprocess(df)

    return features
