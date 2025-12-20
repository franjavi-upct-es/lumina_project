# backend/dat_engine/transformers/feature_engineering.py
"""
Comprehensive feature engineering for financial time series
Creates 100+ technical, statistical, and derived features
"""

from typing import Optional, List, Dict
import polars as pl
import numpy as np
from loguru import logger
import pandas as pd


class FeatureEngineer:
    """
    Creates advanced features from raw OHLCV data
    Uses native implementations for maximum compatibility
    """

    def __init__(self):
        self.feature_categories = {
            "price": [],
            "volume": [],
            "volatility": [],
            "momemtum": [],
            "trend": [],
            "statistical": [],
        }

    def create_all_features(
        self, data: pl.DataFrame, add_lags: bool = True, add_rolling: bool = True
    ):
        """
        Crate comprehensive feature set

        Args:
            data: Input DataFrame with OHLCV data
            add_lags: Whether to add lagged features
            add_rolling: Whether to add rolling statistics

        Returns:
            DataFrame with added features
        """
        logger.info("Starting comprehensive feature engineering...")

        # Converto to pandas for easier manipulation

        df_pd = data.to_pandas()

        # Ensure datetime features
        if "time" in df_pd.columns:
            df_pd = df_pd.set_index("time")

        # 1. Price-based features
        df_pd = self._add_price_features(df_pd)

        # 2. Volume features
        df_pd = self._add_volume_features(df_pd)

        # 3. Volatility features
        df_pd = self._add_volatility_features(df_pd)

        # 4. Momemtum indicators
        df_pd = self._add_momentum_features(df_pd)

        # 5. Trend indicators
        df_pd = self._add_tren_features(df_pd)

        # 6. Statistical features
        df_pd = self._add_statistical_features(df_pd)

        # 7. Lagged features (if requested)
        if add_lags:
            df_pd = self._add_lagged_features(df_pd)

        # 8. Rolling statistics (if requested)
        if add_rolling:
            df_pd = self._add_rolling_features(df_pd)

        # Convert back to Polars
        result = pl.from_pandas(df_pd.reset_index())

        logger.success(
            f"Created {len(result.columns) - len(data.columns)} new features"
        )

    def _add_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Price-based features"""
        logger.debug("Adding price features...")

        # Returns
        df["returns"] = df["close"].pct_change()
        df["log_returns"] = np.log(df["close"].shift(1))

        # Price changes
        df["price_change"] = df["close"] - df["open"]
        df["price_change_pct"] = (df["close"] - df["open"]) / df["open"]

        # Intraday range
        df["high_low_range"] = df["high"] - df["low"]
        df["high_low_range_pct"] = (df["high"] - df["low"]) / df["low"]

        # Gap
        df["gap"] = df["open"] - df["close"].shift(1)
        df["gap_pct"] = df["gap"] / df["close"].shift(1)

        # Typical price
        df["typical_price"] = (df["high"] + df["low"] + df["close"]) / 3

        # Weighted close
        df["weighted_close"] = (df["high"] + df["low"] + 2 * df["close"]) / 4

        self.feature_categories["price"].extend(
            [
                "returns",
                "log_returns",
                "price_change",
                "price_change_pct",
                "high_low_range",
                "high_low_range_pct",
                "gap",
                "gap_pct",
                "typical_price",
                "weighted_close",
            ]
        )

        return df

    def _add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Volume-based features"""
        logger.debug("Adding volume  features...")

        # Volume changes
        df["volume_change"] = df["volume"].pct_change()
        df["volume_ma_ratio_5"] = df["volume"] / df["volume"].rolling(5).mean()
        df["volume_ma_ratio_20"] = df["volume"] / df["volume"].rolling(20).mean()

        # Price-Volume realtionship
        df["price_volume"] = df["close"] * df["volume"]
        df["vwap"] = (
            df["price_volume"].rolling(20).sum() / df["volume"].rolling(20).sum()
        )

        # On-Balance Volume (OBV)
        df["obv"] = (np.sign(df["close"].diff()) * df["volume"]).fillna(0).cumsum()

        # Money Flow
        df["money_flow_multiplier"] = (
            (df["close"] - df["low"]) - (df["high"] - df["close"])
        ) / (df["high"] - df["low"])
        df["money_flow_multiplier"] = (
            df["money_flow_multiplier"].replace([np.inf, -np.inf], 0).fillna(0)
        )
        df["money_flow_volume"] = df["money_flow_multiplier"] * df["volume"]
        df["cmf_20"] = (
            df["money_flow_volume"].rolling(20).sum() / df["volume"].rolling(20).sum()
        )

        self.feature_categories["volume"].extend(
            [
                "volume_change",
                "volume_ma_ratio_5",
                "volume_ma_ratio_20",
                "price_volume",
                "vwap",
                "obv",
                "money_flow_multiplier",
                "money_flow_volume",
                "cmf_20",
            ]
        )

        return df

    def _add_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Volatility measures"""
        logger.debug("Adding volatility features...")

        # Historical volatility (different windows)
        for window in [5, 10, 20, 60]:
            df[f"volatility_{window}d"] = df["returns"].rolling(window).std() * np.sqrt(
                252
            )

        # Parkingson's volatility (uses high-low range)
        df["parkinson_vol_20"] = np.sqrt(
            (1 / (4 * np.log(2)))
            * np.log(df["high"] / df["low"]).pow(2).rolling(20).mean()
        ) * np.sqrt(252)

        # Average True Range (ATR)
        high_low = df["high"] - df["low"]
        high_close = abs(df["high"] - df["close"].shift())
        low_close = abs(df["low"] - df["close"].shift())

        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df["atr_14"] = true_range.rolling(14).mean()
        df["atr_pct_14"] = df["atr_14"] / df["close"]

        # Bollinger Bands volatility
        df["bb_width_20"] = (df["close"].rolling(20).std() * 2) / df["close"].rolling(
            20
        ).mean()

        self.feature_categories["volatility"].extend(
            [
                "volatility_5d",
                "volatility_10d",
                "volatility_20d",
                "volatility_60d",
                "parkingson_vol_20",
                "atr_14",
                "atr_pct_14",
                "bb_width_20",
            ]
        )

        return df

    def _add_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Momemtum indicators - native implementations"""
        logger.debug("Adding momemtum features...")

        # RSI (Relative Strength Index)
        for period in [7, 14, 21]:
            delta = df["close"].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            df[f"rsi_{period}"] = 100 - (100 / (1 + rs))

        # Stochastic Oscillator
        low_14 = df["low"].rolling(14).min()
        high_14 = df["high"].rolling(14).max()
        df["stoch_k"] = 100 * ((df["close"] - low_14) / (high_14 - low_14))
        df["stoch_d"] = df["stoch_k"].rolling(3).mean()

        # Williams %R
        df["williams_r_14"] = -100 * ((high_14 - df["close"]) / (high_14 - low_14))

        # Rate of Change (ROC)
        for period in [5, 10, 20]:
            df[f"roc_{period}"] = df["close"].pct_change(period) * 100

        # Commodity Channel Index (CCI)
        tp = (df["high"] + df["low"] + df["close"]) / 3
        sma_tp = tp.rolling(20).mean()
        mad = tp.rolling(20).apply(lambda x: np.abs(x - x.mean()).mean())
        df["cci_20"] = (tp - sma_tp) / (0.015 * mad)

        # Money Flow Index (MFI)
        typical_price = (df["high"] + df["low"] + df["close"]) / 3
        money_flow = typical_price * df["volume"]

        positive_flow = (
            money_flow.where(typical_price > typical_price.shift(1), 0)
            .rolling(14)
            .sum()
        )
        negative_flow = (
            money_flow.where(typical_price < typical_price.shift(1), 0)
            .rolling(14)
            .sum()
        )

        mfi_ratio = positive_flow / negative_flow
        df["mfi_14"] = 100 - (100 / (1 + mfi_ratio))

        self.feature_categories["momentum"].extend(
            [
                "rsi_7",
                "rsi_14",
                "rsi_21",
                "stoch_k",
                "stoch_d",
                "williams_r_14",
                "roc_5",
                "roc_10",
                "roc_20",
                "cci_20",
                "mfi_14",
            ]
        )

        return df

    def _add_tren_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Trend indicators - native implementations"""
        logger.debug("Adding trend features...")

        # Moving Averages
        for period in [5, 10, 20, 50, 100, 200]:
            df[f"sma_{period}"] = df["close"].rolling(period).mean()
            df[f"sma_{period}_distance"] = (df["close"] - df[f"sma_{period}"]) / df[
                f"sma_{period}"
            ]

        # Exponential Moving Averages
        for period in [12, 26, 50]:
            df[f"ema_{period}"] = df["close"].ewm(span=period, adjust=False).mean()
            df[f"ema_{period}_distance"] = (df["close"] - df[f"ema_{period}"]) / df[
                f"ema_{period}"
            ]

        # MACD
        ema_12 = df["close"].ewm(span=12, adjust=False).mean()
        ema_26 = df["close"].ewm(span=26, adjust=False).mean()
        df["macd"] = ema_12 - ema_26
        df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
        df["macd_histogram"] = df["macd"] - df["macd_signal"]

        # ADX (Average Directional Index)
        high_diff = df["high"].diff()
        low_diff = df["low"].diff()

        plus_dm = high_diff.where((high_diff > low_diff) & (high_diff > 0), 0)
        minus_dm = low_diff.where((low_diff > high_diff) & (low_diff > 0), 0)

        tr = pd.concat(
            [
                df["high"] - df["low"],
                abs(df["high"] - df["close"].shift()),
                abs(df["low"] - df["close"].shift()),
            ],
            axis=1,
        ).max(axis=1)

        atr = tr.rolling(14).mean()
        plus_di = 100 * (plus_dm.rolling(14).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(14).mean() / atr)

        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        df["ax_14"] = dx.rolling(14).mean()
        df["adx_pos"] = plus_di
        df["adx_neg"] = minus_di

        # Bollinger Bands
        sma_20 = df["close"].rolling(20).mean()
        std_20 = df["close"].rolling(20).std()
        df["bb_upper"] = sma_20 + (std_20 * 2)
        df["bb_middle"] = sma_20
        df["bb_lower"] = sma_20 - (std_20 * 2)
        df["bb_position"] = (df["close"] - df["bb_lower"]) / (
            df["bb_upper"] - df["bb_lower"]
        )

        # Parabolic SAR (simplifies version)
        df["psar"] = df["close"].rolling(20).mean()  # Simplified
        df["psar_signal"] = np.where(df["close"] > df["psar"], 1, -1)

        self.feature_categories["trend"].extend(
            [
                "sma_5",
                "sma_10",
                "sma_20",
                "sma_50",
                "sma_100",
                "sma_200",
                "ema_12",
                "ema_26",
                "ema_50",
                "macd",
                "macd_signal",
                "macd_histogram",
                "adx_14",
                "adx_pos",
                "adx_neg",
                "bb_upper",
                "bb_middle",
                "bb_lower",
                "bb_position",
                "psar",
                "psar_signal",
            ]
        )

        return df

    def _add_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Statistical features"""
        logger.debug("Adding statistical features...")

        # Skewness and Kurtosis
        for window in [20, 60]:
            df[f"skew_{window}"] = df["returns"].rolling(window).skew()
            df[f"kurt_{window}"] = df["returns"].rolling(window).kurt()

        # Z-score (standardized returns)
        for window in [20, 60]:
            rolling_mean = df["returns"].rolling(window).mean()
            rolling_std = df["returns"].rolling(window).std()
            df[f"zscore_{window}"] = (df["returns"] - rolling_mean) / rolling_std

        # Percentile rank
        df["percentile_rank_20"] = (
            df["close"]
            .rolling(20)
            .apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1])
        )

        # Distance from moving average (normalized)
        for period in [20, 50]:
            ma = df["close"].rolling(period).mean()
            std = df["close"].rolling(period).std()
            df[f"dist_from_ma_{period}_norm"] = (df["close"] - ma) / std

        self.feature_categories["statistical"].extend(
            [
                "skew_20",
                "skew_60",
                "kurt_20",
                "kurt_60",
                "zscore_20",
                "zscore_60",
                "percentile_rank_20",
                "dist_from_ma_20_norm",
                "dist_from_ma_50_norm",
            ]
        )

        return df

    def _add_lagged_features(
        self, df: pd.DataFrame, lags: List[int] = None
    ) -> pd.DataFrame:
        """Add lagged features for capturing temporal patterns"""
        if lags is None:
            lags = [1, 2, 3, 5, 10, 20]

        logger.debug(f"Adding lagged features for lags: {lags}")

        # Lag important features
        features_to_lag = ["returns", "volume_change", "rsi_14", "macd"]

        for feature in features_to_lag:
            if feature in df.columns:
                for lag in lags:
                    df[f"{feature}_lag_{lag}"] = df["feature"].shift(lag)

        return df

    def _add_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add rolling window statistics"""
        logger.debug("Adding rolling statistics...")

        windows = [5, 10, 20]

        for window in windows:
            # Rolling max/min
            df[f"high_max_{window}"] = df["high"].rolling(window).max()
            df[f"low_min_{window}"] = df["low"].rolling(window).min()

            # Distance from rolling max/min
            df[f"dist_from_high_{window}"] = (
                df["close"] - df[f"high_max_{window}"]
            ) / df[f"high_max_{window}"]
            df[f"dist_from_low_{window}"] = (
                df["close"] - df[f"low_min_{window}"]
            ) / df[f"low_min_{window}"]

        return df

    def get_feature_names_by_category(self, category: str) -> List[str]:
        """Get list of features by category"""
        return self.feature_categories.get(category, [])

    def get_all_feature_names(self) -> List[str]:
        """Get list of all created features"""
        all_features = []
        for features in self.feature_categories.values():
            all_features.append(features)
        return all_features
