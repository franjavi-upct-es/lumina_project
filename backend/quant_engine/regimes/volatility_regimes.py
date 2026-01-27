# backend/quant_engine/regimes/volatility_regimes.py
"""
Volatility-Based Market Regime Detection

Detects market regimes based on volatility characteristics using various methods:
- GARCH model-based regimes
- Realized volatility thresholds
- Volatility percentile regimes
- ATR (Average True Range) based regimes
- Parkinson volatility estimator

Volatility regimes are useful for:
- Risk management (high volatility = risk-off)
- Position sizing adjustments
- Strategy selection (trend-following vs mean-reversion)
- Options trading strategies
"""

import numpy as np
import pandas as pd
import polars as pl
from arch import arch_model
from loguru import logger


class VolatilityRegimeDetector:
    """
    Detect market regimes based on volatility characteristics

    Provides multiple methods for volatility-based regime classification,
    from simple percentile-based approaches to sophisticated GARCH models.
    """

    def __init__(
        self,
        method: str = "percentile",
        n_regimes: int = 3,
    ):
        """
        Initialize volatility regime detector

        Args:
            method: Detection method
                - 'percentile': Classify by volatility percentiles
                - 'garch': Use GARCH model volatility forecasts
                - 'realized': Use realized volatility with thresholds
                - 'atr': Use Average True Range
                - 'parkinson': Use Parkinson high-low volatility estimator
                - 'adaptive': Adaptive threshold based on rolling statistics
            n_regimes: Number of volatility regimes (typically 2-4)
        """
        self.method = method
        self.n_regimes = n_regimes

        # Thresholds (will be calculated during fit)
        self.thresholds: list[float] = []

        # GARCH model (if using GARCH method)
        self.garch_model = None

        # Regime labels
        self.regime_labels = self._initialize_regime_labels()

        logger.info(
            f"Volatility regime detector initialized: method={method}, n_regimes={n_regimes}"
        )

    def _initialize_regime_labels(self) -> dict[int, str]:
        """Initialize regime labels based on number of regimes"""
        if self.n_regimes == 2:
            return {0: "low_vol", 1: "high_vol"}
        elif self.n_regimes == 3:
            return {0: "low_vol", 1: "medium_vol", 2: "high_vol"}
        elif self.n_regimes == 4:
            return {0: "very_low_vol", 1: "low_vol", 2: "high_vol", 3: "very_high_vol"}
        else:
            return {i: f"vol_regime_{i}" for i in range(self.n_regimes)}

    def calculate_realized_volatility(
        self,
        data: pl.DataFrame,
        window: int = 20,
        annualize: bool = True,
    ) -> pl.DataFrame:
        """
        Calculate realized volatility from returns

        Args:
            data: DataFrame with price data
            window: Rolling window for volatility calculation
            annualize: Whether to annualize volatility (multiply by sqrt(252))

        Returns:
            DataFrame with realized volatility column
        """
        result = data.clone()

        # Calculate returns
        result = result.with_columns([pl.col("close").pct_change().alias("returns")])

        # Calculate rolling standard deviation
        result = result.with_columns([pl.col("returns").rolling_std(window).alias("realized_vol")])

        # Annualize if requested
        if annualize:
            result = result.with_columns(
                [(pl.col("realized_vol") * np.sqrt(252)).alias("realized_vol")]
            )

        return result

    def calculate_parkinson_volatility(
        self,
        data: pl.DataFrame,
        window: int = 20,
        annualize: bool = True,
    ) -> pl.DataFrame:
        """
        Calculate Parkinson volatility estimator (uses high-low range)

        More efficient than close-to-close volatility.
        Formula: sigma = sqrt((1/(4*ln(2))) * (ln(High/Low))^2)

        Args:
            data: DataFrame with high and low prices
            window: Rolling window
            annualize: Whether to annualize

        Returns:
            DataFrame with Parkinson volatility
        """
        if "high" not in data.columns or "low" not in data.columns:
            raise ValueError("Data must contain 'high' and 'low' columns")

        result = data.clone()

        # Calculate log(High/Low)^2
        hl_squared = (pl.col("high") / pl.col("low")).log().pow(2)

        # Parkinson estimator
        parkinson_factor = 1 / (4 * np.log(2))

        result = result.with_columns(
            [(hl_squared.rolling_mean(window) * parkinson_factor).sqrt().alias("parkinson_vol")]
        )

        # Annualize
        if annualize:
            result = result.with_columns(
                [(pl.col("parkinson_vol") * np.sqrt(252)).alias("parkinson_vol")]
            )

        return result

    def calculate_atr(
        self,
        data: pl.DataFrame,
        window: int = 14,
    ) -> pl.DataFrame:
        """
        Calculate Average True Range (ATR)

        Args:
            data: DataFrame with OHLC data
            window: ATR period

        Returns:
            DataFrame with ATR column
        """
        if not all(col in data.columns for col in ["high", "low", "close"]):
            raise ValueError("Data must contain 'high', 'low', 'close' columns")

        result = data.clone()

        # True Range = max(high-low, abs(high-prev_close), abs(low-prev_close))
        prev_close = pl.col("close").shift(1)

        tr = pl.max_horizontal(
            [
                pl.col("high") - pl.col("low"),
                (pl.col("high") - prev_close).abs(),
                (pl.col("low") - prev_close).abs(),
            ]
        )

        result = result.with_columns([tr.alias("true_range")])

        # Calculate ATR as EMA of True Range
        result = result.with_columns(
            [pl.col("true_range").ewm_mean(span=window, adjust=False).alias("atr")]
        )

        # ATR percentage (normalized by price)
        result = result.with_columns([(pl.col("atr") / pl.col("close") * 100).alias("atr_pct")])

        return result

    def fit_garch(
        self,
        data: pl.DataFrame,
        p: int = 1,
        q: int = 1,
    ) -> "VolatilityRegimeDetector":
        """
        Fit GARCH model for volatility forecasting

        Args:
            data: DataFrame with returns
            p: GARCH lag order
            q: ARCH lag order

        Returns:
            self
        """
        logger.info(f"Fitting GARCH({p},{q}) model")

        # Prepare returns
        if "returns" not in data.columns:
            returns_data = data.with_columns([pl.col("close").pct_change().alias("returns")])
        else:
            returns_data = data

        returns = (
            returns_data.select("returns").to_series().drop_nulls().to_numpy() * 100
        )  # Scale to percentage

        # Fit GARCH model
        self.garch_model = arch_model(
            returns,
            vol="Garch",
            p=p,
            q=q,
            dist="normal",
        )

        self.garch_fit = self.garch_model.fit(disp="off", show_warning=False)

        logger.success(f"GARCH model fitted: AIC={self.garch_fit.aic:.2f}")

        return self

    def predict_garch_volatility(
        self,
        horizon: int = 1,
    ) -> float:
        """
        Forecast volatility using fitted GARCH model

        Args:
            horizon: Forecast horizon (days)

        Returns:
            Forecasted volatility
        """
        if self.garch_model is None:
            raise ValueError("GARCH model must be fitted first")

        forecast = self.garch_fit.forecast(horizon=horizon)
        volatility = np.sqrt(forecast.variance.values[-1, 0])

        return volatility

    def detect_percentile_regimes(
        self,
        data: pl.DataFrame,
        volatility_col: str = "realized_vol",
    ) -> pl.DataFrame:
        """
        Detect regimes using volatility percentiles

        Args:
            data: DataFrame with volatility column
            volatility_col: Name of volatility column

        Returns:
            DataFrame with regime classifications
        """
        result = data.clone()

        # Calculate percentile thresholds
        vol_values = result.select(volatility_col).to_series().drop_nulls()

        if self.n_regimes == 2:
            self.thresholds = [vol_values.quantile(0.5)]
        elif self.n_regimes == 3:
            self.thresholds = [vol_values.quantile(0.33), vol_values.quantile(0.67)]
        elif self.n_regimes == 4:
            self.thresholds = [
                vol_values.quantile(0.25),
                vol_values.quantile(0.50),
                vol_values.quantile(0.75),
            ]
        else:
            # Equal percentiles
            percentiles = [i / self.n_regimes for i in range(1, self.n_regimes)]
            self.thresholds = [vol_values.quantile(p) for p in percentiles]

        logger.info(f"Volatility thresholds: {self.thresholds}")

        # Classify regimes
        def classify_regime(vol):
            if vol is None:
                return None
            for i, threshold in enumerate(self.thresholds):
                if vol < threshold:
                    return i
            return len(self.thresholds)

        regimes = result.select(volatility_col).to_series().apply(classify_regime)
        regime_labels = regimes.apply(
            lambda r: self.regime_labels.get(r, f"regime_{r}") if r is not None else None
        )

        result = result.with_columns(
            [
                regimes.alias("regime"),
                regime_labels.alias("regime_label"),
            ]
        )

        return result

    def detect_adaptive_regimes(
        self,
        data: pl.DataFrame,
        volatility_col: str = "realized_vol",
        lookback: int = 252,
    ) -> pl.DataFrame:
        """
        Detect regimes using adaptive thresholds

        Thresholds are recalculated using rolling window, allowing
        regime detection to adapt to changing market conditions.

        Args:
            data: DataFrame with volatility column
            volatility_col: Name of volatility column
            lookback: Rolling window for threshold calculation

        Returns:
            DataFrame with regime classifications
        """
        result = data.clone()

        # Calculate rolling percentiles
        vol_series = pl.col(volatility_col)

        if self.n_regimes == 2:
            thresholds_df = result.select(
                [vol_series.rolling_quantile(0.5, window_size=lookback).alias("threshold_0")]
            )
        elif self.n_regimes == 3:
            thresholds_df = result.select(
                [
                    vol_series.rolling_quantile(0.33, window_size=lookback).alias("threshold_0"),
                    vol_series.rolling_quantile(0.67, window_size=lookback).alias("threshold_1"),
                ]
            )
        else:
            percentiles = [i / self.n_regimes for i in range(1, self.n_regimes)]
            threshold_cols = [
                vol_series.rolling_quantile(p, window_size=lookback).alias(f"threshold_{i}")
                for i, p in enumerate(percentiles)
            ]
            thresholds_df = result.select(threshold_cols)

        # Classify based on adaptive thresholds
        vol_values = result.select(volatility_col).to_series()
        threshold_arrays = [thresholds_df.select(col).to_series() for col in thresholds_df.columns]

        regimes = []
        for i in range(len(vol_values)):
            vol = vol_values[i]
            if vol is None:
                regimes.append(None)
                continue

            regime = self.n_regimes - 1  # Default to highest regime
            for j, threshold_array in enumerate(threshold_arrays):
                threshold = threshold_array[i]
                if threshold is not None and vol < threshold:
                    regime = j
                    break

            regimes.append(regime)

        regime_labels = [
            self.regime_labels.get(r, f"regime_{r}") if r is not None else None for r in regimes
        ]

        result = result.with_columns(
            [
                pl.Series("regime", regimes),
                pl.Series("regime_label", regime_labels),
            ]
        )

        return result

    def detect(
        self,
        data: pl.DataFrame,
        **kwargs,
    ) -> pl.DataFrame:
        """
        Detect volatility regimes using configured method

        Args:
            data: DataFrame with OHLCV data
            **kwargs: Method-specific parameters

        Returns:
            DataFrame with regime classifications
        """
        logger.info(f"Detecting volatility regimes using method: {self.method}")

        if self.method == "realized":
            window = kwargs.get("window", 20)
            result = self.calculate_realized_volatility(data, window=window)
            result = self.detect_percentile_regimes(result, "realized_vol")

        elif self.method == "percentile":
            window = kwargs.get("window", 20)
            result = self.calculate_realized_volatility(data, window=window)
            result = self.detect_percentile_regimes(result, "realized_vol")

        elif self.method == "parkinson":
            window = kwargs.get("window", 20)
            result = self.calculate_parkinson_volatility(data, window=window)
            result = self.detect_percentile_regimes(result, "parkinson_vol")

        elif self.method == "atr":
            window = kwargs.get("window", 14)
            result = self.calculate_atr(data, window=window)
            result = self.detect_percentile_regimes(result, "atr_pct")

        elif self.method == "adaptive":
            window = kwargs.get("window", 20)
            lookback = kwargs.get("lookback", 252)
            result = self.calculate_realized_volatility(data, window=window)
            result = self.detect_adaptive_regimes(result, "realized_vol", lookback=lookback)

        elif self.method == "garch":
            p = kwargs.get("p", 1)
            q = kwargs.get("q", 1)

            # Fit GARCH model
            self.fit_garch(data, p=p, q=q)

            # Get conditional volatility from model
            result = data.clone()

            if "returns" not in result.columns:
                result = result.with_columns([pl.col("close").pct_change().alias("returns")])

            # Extract conditional volatility
            cond_vol = self.garch_fit.conditional_volatility

            result = result.with_columns([pl.Series("garch_vol", cond_vol)])

            result = self.detect_percentile_regimes(result, "garch_vol")

        else:
            raise ValueError(f"Unknown method: {self.method}")

        logger.success(f"Volatility regime detection complete")

        return result

    def get_regime_statistics(
        self,
        data: pl.DataFrame,
    ) -> pd.DataFrame:
        """
        Calculate statistics for each volatility regime

        Args:
            data: DataFrame with regime predictions

        Returns:
            DataFrame with regime statistics
        """
        if "regime" not in data.columns:
            raise ValueError("Data must contain 'regime' column")

        stats = []

        # Calculate returns if not present
        data_with_returns = data
        if "returns" not in data.columns:
            data_with_returns = data.with_columns([pl.col("close").pct_change().alias("returns")])

        for regime_id in range(self.n_regimes):
            regime_data = data_with_returns.filter(pl.col("regime") == regime_id)

            if regime_data.height == 0:
                continue

            stat_dict = {
                "regime": regime_id,
                "regime_label": self.regime_labels.get(regime_id, f"regime_{regime_id}"),
                "n_samples": regime_data.height,
                "frequency": regime_data.height / data.height,
            }

            # Return statistics
            if "returns" in regime_data.columns:
                returns = regime_data.select("returns").to_series()
                stat_dict.update(
                    {
                        "mean_return": returns.mean(),
                        "std_return": returns.std(),
                        "sharpe": returns.mean() / returns.std() if returns.std() > 0 else 0,
                        "skewness": returns.skew(),
                        "kurtosis": returns.kurtosis(),
                    }
                )

            # Volatility statistics (find volatility column)
            vol_cols = [
                col for col in regime_data.columns if "vol" in col.lower() and col != "volume"
            ]
            if vol_cols:
                vol = regime_data.select(vol_cols[0]).to_series()
                stat_dict[f"mean_{vol_cols[0]}"] = vol.mean()
                stat_dict[f"median_{vol_cols[0]}"] = vol.median()

            stats.append(stat_dict)

        return pd.DataFrame(stats)


def detect_volatility_regimes(
    data: pl.DataFrame,
    method: str = "percentile",
    n_regimes: int = 3,
    **kwargs,
) -> tuple[pl.DataFrame, VolatilityRegimeDetector]:
    """
    Convenience function for volatility regime detection

    Args:
        data: DataFrame with OHLCV data
        method: Detection method
        n_regimes: Number of regimes
        **kwargs: Method-specific parameters

    Returns:
        Tuple of (data_with_regimes, fitted_detector)

    Example:
        >>> # Percentile-based detection
        >>> data_with_regimes, detector = detect_volatility_regimes(
        ...     ohlcv_data,
        ...     method='percentile',
        ...     n_regimes=3,
        ...     window=20
        ... )

        >>> # GARCH-based detection
        >>> data_with_regimes, detector = detect_volatility_regimes(
        ...     ohlcv_data,
        ...     method='garch',
        ...     n_regimes=3,
        ...     p=1,
        ...     q=1
        ... )

        >>> # Get statistics
        >>> stats = detector.get_regime_statistics(data_with_regimes)
    """
    # Initialize detector
    detector = VolatilityRegimeDetector(
        method=method,
        n_regimes=n_regimes,
    )

    # Detect regimes
    result = detector.detect(data, **kwargs)

    logger.success(f"Volatility regime detection complete using {method}")

    return result, detector
