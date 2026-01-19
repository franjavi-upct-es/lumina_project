# backend/ml_engine/features/technical.py
"""
Technical indicators feature engineering
Comprehensive set of technical analysis indicators
"""

import numpy as np
import pandas as pd
from loguru import logger


class TechnicalFeatures:
    """
    Technical analysis features for ML models

    Categories:
    - Trend indicators (MA, MACD, ADX)
    - Momentum indicators (RSI, Stochastic, ROC)
    - Volatility indicators (ATR, Bollinger Bands)
    - Volume indicators (OBV, VWAP, MFI)
    """

    def __init__(self):
        self.feature_names = []

    def create_all_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create all technical features

        Args:
            data: DataFrame with OHLCV columns

        Returns:
            DataFrame with technical features added
        """
        df = data.copy()

        # Trend indicators
        df = self._add_moving_averages(df)
        df = self._add_macd(df)
        df = self._add_adx(df)
        df = self._add_parabolic_sar(df)

        # Momentum indicators
        df = self._add_rsi(df)
        df = self._add_stochastic(df)
        df = self._add_roc(df)
        df = self._add_cci(df)
        df = self._add_williams_r(df)

        # Volatility indicators
        df = self._add_atr(df)
        df = self._add_bollinger_bands(df)
        df = self._add_keltner_channels(df)
        df = self._add_donchian_channels(df)

        # Volume indicators
        df = self._add_obv(df)
        df = self._add_vwap(df)
        df = self._add_mfi(df)
        df = self._add_cmf(df)

        logger.info(f"Created {len(self.feature_names)} technical features")
        return df

    def _add_moving_averages(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add various moving averages"""
        for period in [5, 10, 20, 50, 100, 200]:
            # Simple Moving Average
            df[f"sma_{period}"] = df["close"].rolling(period).mean()
            self.feature_names.append(f"sma_{period}")

            # Distance from SMA
            df[f"price_to_sma_{period}"] = (df["close"] - df[f"sma_{period}"]) / df[f"sma_{period}"]
            self.feature_names.append(f"price_to_sma_{period}")

        # Exponential Moving Averages
        for period in [12, 26, 50]:
            df[f"ema_{period}"] = df["close"].ewm(span=period, adjust=False).mean()
            self.feature_names.append(f"ema_{period}")

        # MA crossovers
        df["sma_cross_20_50"] = df["sma_20"] - df["sma_50"]
        df["sma_cross_50_200"] = df["sma_50"] - df["sma_200"]
        self.feature_names.extend(["sma_cross_20_50", "sma_cross_50_200"])

        return df

    def _add_macd(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add MACD indicator"""
        ema_12 = df["close"].ewm(span=12, adjust=False).mean()
        ema_26 = df["close"].ewm(span=26, adjust=False).mean()

        df["macd"] = ema_12 - ema_26
        df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
        df["macd_histogram"] = df["macd"] - df["macd_signal"]

        self.feature_names.extend(["macd", "macd_signal", "macd_histogram"])
        return df

    def _add_adx(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Add Average Directional Index"""
        high_diff = df["high"].diff()
        low_diff = df["low"].diff()

        plus_dm = high_diff.where((high_diff > low_diff) & (high_diff > 0), 0)
        minus_dm = low_diff.where((low_diff > high_diff) & (low_diff > 0), 0)

        tr = pd.concat(
            [
                df["high"] - df["low"],
                (df["high"] - df["close"].shift()).abs(),
                (df["low"] - df["close"].shift()).abs(),
            ],
            axis=1,
        ).max(axis=1)

        atr = tr.rolling(period).mean()
        plus_di = 100 * (plus_dm.rolling(period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(period).mean() / atr)

        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
        df["adx"] = dx.rolling(period).mean()
        df["adx_pos"] = plus_di
        df["adx_neg"] = minus_di

        self.feature_names.extend(["adx", "adx_pos", "adx_neg"])
        return df

    def _add_parabolic_sar(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add Parabolic SAR (simplified)"""
        df["psar"] = df["close"].rolling(20).mean()
        df["psar_signal"] = (df["close"] > df["psar"]).astype(int)

        self.feature_names.extend(["psar", "psar_signal"])
        return df

    def _add_rsi(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add RSI for multiple periods"""
        delta = df["close"].diff()

        for period in [7, 14, 21]:
            gain = delta.where(delta > 0, 0).rolling(period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(period).mean()

            rs = gain / loss
            df[f"rsi_{period}"] = 100 - (100 / (1 + rs))
            self.feature_names.append(f"rsi_{period}")

        return df

    def _add_stochastic(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Add Stochastic Oscillator"""
        low_min = df["low"].rolling(period).min()
        high_max = df["high"].rolling(period).max()

        df["stoch_k"] = 100 * ((df["close"] - low_min) / (high_max - low_min))
        df["stoch_d"] = df["stoch_k"].rolling(3).mean()

        self.feature_names.extend(["stoch_k", "stoch_d"])
        return df

    def _add_roc(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add Rate of Change"""
        for period in [5, 10, 20]:
            df[f"roc_{period}"] = df["close"].pct_change(period) * 100
            self.feature_names.append(f"roc_{period}")

        return df

    def _add_cci(self, df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        """Add Commodity Channel Index"""
        tp = (df["high"] + df["low"] + df["close"]) / 3
        sma_tp = tp.rolling(period).mean()
        mad = tp.rolling(period).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)

        df["cci"] = (tp - sma_tp) / (0.015 * mad)
        self.feature_names.append("cci")

        return df

    def _add_williams_r(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Add Williams %R"""
        high_max = df["high"].rolling(period).max()
        low_min = df["low"].rolling(period).min()

        df["williams_r"] = -100 * ((high_max - df["close"]) / (high_max - low_min))
        self.feature_names.append("williams_r")

        return df

    def _add_atr(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Add Average True Range"""
        high_low = df["high"] - df["low"]
        high_close = (df["high"] - df["close"].shift()).abs()
        low_close = (df["low"] - df["close"].shift()).abs()

        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df["atr"] = tr.rolling(period).mean()
        df["atr_pct"] = df["atr"] / df["close"]

        self.feature_names.extend(["atr", "atr_pct"])
        return df

    def _add_bollinger_bands(
        self, df: pd.DataFrame, period: int = 20, std: int = 2
    ) -> pd.DataFrame:
        """Add Bollinger Bands"""
        sma = df["close"].rolling(period).mean()
        rolling_std = df["close"].rolling(period).std()

        df["bb_upper"] = sma + (rolling_std * std)
        df["bb_middle"] = sma
        df["bb_lower"] = sma - (rolling_std * std)
        df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_middle"]
        df["bb_position"] = (df["close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])

        self.feature_names.extend(["bb_upper", "bb_middle", "bb_lower", "bb_width", "bb_position"])
        return df

    def _add_keltner_channels(self, df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        """Add Keltner Channels"""
        ema = df["close"].ewm(span=period, adjust=False).mean()
        atr = self._calculate_atr(df, period)

        df["keltner_upper"] = ema + (2 * atr)
        df["keltner_middle"] = ema
        df["keltner_lower"] = ema - (2 * atr)

        self.feature_names.extend(["keltner_upper", "keltner_middle", "keltner_lower"])
        return df

    def _add_donchian_channels(self, df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        """Add Donchian Channels"""
        df["donchian_upper"] = df["high"].rolling(period).max()
        df["donchian_lower"] = df["low"].rolling(period).min()
        df["donchian_middle"] = (df["donchian_upper"] + df["donchian_lower"]) / 2

        self.feature_names.extend(["donchian_upper", "donchian_lower", "donchian_middle"])
        return df

    def _add_obv(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add On-Balance Volume"""
        df["obv"] = (np.sign(df["close"].diff()) * df["volume"]).fillna(0).cumsum()
        self.feature_names.append("obv")

        return df

    def _add_vwap(self, df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        """Add Volume Weighted Average Price"""
        typical_price = (df["high"] + df["low"] + df["close"]) / 3
        df["vwap"] = (typical_price * df["volume"]).rolling(period).sum() / df["volume"].rolling(
            period
        ).sum()

        self.feature_names.append("vwap")
        return df

    def _add_mfi(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Add Money Flow Index"""
        typical_price = (df["high"] + df["low"] + df["close"]) / 3
        money_flow = typical_price * df["volume"]

        positive_flow = (
            money_flow.where(typical_price > typical_price.shift(1), 0).rolling(period).sum()
        )
        negative_flow = (
            money_flow.where(typical_price < typical_price.shift(1), 0).rolling(period).sum()
        )

        mfi_ratio = positive_flow / negative_flow
        df["mfi"] = 100 - (100 / (1 + mfi_ratio))

        self.feature_names.append("mfi")
        return df

    def _add_cmf(self, df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        """Add Chaikin Money Flow"""
        mf_multiplier = ((df["close"] - df["low"]) - (df["high"] - df["close"])) / (
            df["high"] - df["low"]
        )
        mf_volume = mf_multiplier * df["volume"]

        df["cmf"] = mf_volume.rolling(period).sum() / df["volume"].rolling(period).sum()
        self.feature_names.append("cmf")

        return df

    def _calculate_atr(self, df: pd.DataFrame, period: int) -> pd.Series:
        """Helper to calculate ATR"""
        high_low = df["high"] - df["low"]
        high_close = (df["high"] - df["close"].shift()).abs()
        low_close = (df["low"] - df["close"].shift()).abs()

        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return tr.rolling(period).mean()

    def get_feature_names(self) -> list:
        """Get list of all feature names"""
        return self.feature_names
