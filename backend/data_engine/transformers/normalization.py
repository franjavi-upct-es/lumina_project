# backend/data_engine/transformers/normalization.py
"""
Data normalization and scaling transformers
Prepares data for machine learning models
"""

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from loguru import logger
from sklearn.preprocessing import (
    MinMaxScaler,
    PowerTransformer,
    QuantileTransformer,
    RobustScaler,
    StandardScaler,
)


class DataNormalizer:
    """
    Comprehensive data normalization for financial time series

    Methods:
    - Standard scaling (z-score)
    - Min-Max scaling
    - Robust scaling (median and IQR)
    - Quantile transformation
    - Power transformation (Box-Cox, Yeo-Johnson)
    - Log transformation
    - Percentage returns
    """

    def __init__(self):
        """Initialize normalizer"""
        self.scalers: dict[str, Any] = {}
        self.fitted_columns: list[str] = []
        self.method: str | None = None
        self.is_fitted = False

    def fit(
        self,
        data: pl.DataFrame,
        columns: list[str] | None = None,
        method: str = "standard",
        **kwargs,
    ) -> "DataNormalizer":
        """
        Fit normalizer to data

        Args:
            data: Input DataFrame
            columns: Columns to normalize (None = all numeric)
            method: Normalization method
                - 'standard': Z-score standardization
                - 'minmax': Min-max scaling to [0, 1]
                - 'robust': Robust scaling using median and IQR
                - 'quantile': Quantile transformation
                - 'power': Power transformation (Yeo-Johnson)
                - 'log': Natural logarithm
                - 'returns': Percentage returns
            **kwargs: Additional parameters for scalers

        Returns:
            self
        """
        logger.info(f"Fitting {method} normalizer")

        self.method = method

        # Convert to pandas for sklearn compatibility
        data_pd = data.to_pandas()

        # Select columns
        if columns is None:
            # Get all numeric columns
            columns = data_pd.select_dtypes(include=[np.number]).columns.tolist()
            # Exclude time column if present
            columns = [c for c in columns if c not in ["time", "date"]]

        self.fitted_columns = columns

        if not columns:
            logger.warning("No numeric columns found for normalization")
            return self

        # Fit based on method
        if method == "standard":
            for col in columns:
                scaler = StandardScaler()
                scaler.fit(data_pd[[col]])
                self.scalers[col] = scaler

        elif method == "minmax":
            feature_range = kwargs.get("feature_range", (0, 1))
            for col in columns:
                scaler = MinMaxScaler(feature_range=feature_range)
                scaler.fit(data_pd[[col]])
                self.scalers[col] = scaler

        elif method == "robust":
            for col in columns:
                scaler = RobustScaler()
                scaler.fit(data_pd[[col]])
                self.scalers[col] = scaler

        elif method == "quantile":
            n_quantiles = kwargs.get("n_quantiles", 1000)
            output_distribution = kwargs.get("output_distribution", "uniform")
            for col in columns:
                scaler = QuantileTransformer(
                    n_quantiles=n_quantiles,
                    output_distribution=output_distribution,
                )
                scaler.fit(data_pd[[col]])
                self.scalers[col] = scaler

        elif method == "power":
            for col in columns:
                scaler = PowerTransformer(method="yeo-johnson", standardize=True)
                scaler.fit(data_pd[[col]])
                self.scalers[col] = scaler

        elif method == "log":
            # No fitting needed for log transform
            for col in columns:
                self.scalers[col] = "log"

        elif method == "returns":
            # No fitting needed for returns
            for col in columns:
                self.scalers[col] = "returns"

        else:
            raise ValueError(f"Unknown normalization method: {method}")

        self.is_fitted = True
        logger.success(f"Fitted {method} normalizer on {len(columns)} columns")
        return self

    def transform(self, data: pl.DataFrame) -> pl.DataFrame:
        """
        Transform data using fitted normalizer

        Args:
            data: Input DataFrame

        Returns:
            Normalized DataFrame
        """
        if not self.is_fitted:
            raise ValueError("Normalizer must be fitted before transform")

        logger.info(f"Transforming data with {self.method} normalization")

        # Convert to pandas
        data_pd = data.to_pandas()

        # Transform each column
        for col in self.fitted_columns:
            if col not in data_pd.columns:
                logger.warning(f"Column {col} not found in data")
                continue

            scaler = self.scalers[col]

            if scaler == "log":
                # Log transformation
                # Add small epsilon to avoid log(0)
                epsilon = 1e-10
                data_pd[col] = np.log(data_pd[col] + epsilon)

            elif scaler == "returns":
                # Percentage returns
                data_pd[col] = data_pd[col].pct_change()

            else:
                # Sklearn scaler
                data_pd[col] = scaler.transform(data_pd[[col]])

        # Convert back to polars
        result = pl.from_pandas(data_pd)

        logger.success("Data transformation complete")
        return result

    def fit_transform(
        self,
        data: pl.DataFrame,
        columns: list[str] | None = None,
        method: str = "standard",
        **kwargs,
    ) -> pl.DataFrame:
        """
        Fit and transform in one step

        Args:
            data: Input DataFrame
            columns: Columns to normalize
            method: Normalization method
            **kwargs: Additional parameters

        Returns:
            Normalized DataFrame
        """
        self.fit(data, columns, method, **kwargs)
        return self.transform(data)

    def inverse_transform(self, data: pl.DataFrame) -> pl.DataFrame:
        """
        Reverse normalization transformation

        Args:
            data: Normalized DataFrame

        Returns:
            Original scale DataFrame
        """
        if not self.is_fitted:
            raise ValueError("Normalizer must be fitted before inverse transform")

        logger.info("Performing inverse transformation")

        # Convert to pandas
        data_pd = data.to_pandas()

        # Inverse transform each column
        for col in self.fitted_columns:
            if col not in data_pd.columns:
                continue

            scaler = self.scalers[col]

            if scaler == "log":
                # Inverse log
                data_pd[col] = np.exp(data_pd[col])

            elif scaler == "returns":
                # Cannot reverse returns without original prices
                logger.warning(f"Cannot inverse transform returns for {col}")

            else:
                # Sklearn scaler
                data_pd[col] = scaler.inverse_transform(data_pd[[col]])

        # Convert back to polars
        result = pl.from_pandas(data_pd)

        logger.success("Inverse transformation complete")
        return result

    def save(self, path: str) -> None:
        """
        Save normalizer to disk

        Args:
            path: Path to save file
        """
        state = {
            "scalers": self.scalers,
            "fitted_columns": self.fitted_columns,
            "method": self.method,
            "is_fitted": self.is_fitted,
        }

        Path(path).parent.mkdir(parents=True, exist_ok=True)

        with open(path, "wb") as f:
            pickle.dump(state, f)

        logger.info(f"Saved normalizer to {path}")

    def load(self, path: str) -> "DataNormalizer":
        """
        Load normalizer from disk

        Args:
            path: Path to load file

        Returns:
            self
        """
        with open(path, "rb") as f:
            state = pickle.load(f)

        self.scalers = state["scalers"]
        self.fitted_columns = state["fitted_columns"]
        self.method = state["method"]
        self.is_fitted = state["is_fitted"]

        logger.info(f"Loaded normalizer from {path}")
        return self


class TimeSeriesNormalizer:
    """
    Specialized normalizer for time series data

    Features:
    - Rolling window normalization
    - Expanding window normalization
    - Detrending
    - Seasonal adjustment
    """

    def __init__(self):
        """Initialize time series normalizer"""
        self.method: str | None = None
        self.window: int | None = None
        self.params: dict[str, Any] = {}

    def rolling_normalize(
        self,
        data: pl.DataFrame,
        columns: list[str],
        window: int = 20,
        method: str = "zscore",
    ) -> pl.DataFrame:
        """
        Normalize using rolling window statistics

        Args:
            data: Input DataFrame
            columns: Columns to normalize
            window: Rolling window size
            method: 'zscore' or 'minmax'

        Returns:
            Normalized DataFrame
        """
        logger.info(f"Applying rolling {method} normalization (window={window})")

        result = data.clone()

        for col in columns:
            if col not in data.columns:
                continue

            if method == "zscore":
                # Rolling z-score
                rolling_mean = data[col].rolling_mean(window)
                rolling_std = data[col].rolling_std(window)

                normalized = (data[col] - rolling_mean) / rolling_std
                result = result.with_columns(normalized.alias(f"{col}_norm"))

            elif method == "minmax":
                # Rolling min-max
                rolling_min = data[col].rolling_min(window)
                rolling_max = data[col].rolling_max(window)

                normalized = (data[col] - rolling_min) / (rolling_max - rolling_min)
                result = result.with_columns(normalized.alias(f"{col}_norm"))

        logger.success("Rolling normalization complete")
        return result

    def expanding_normalize(
        self,
        data: pl.DataFrame,
        columns: list[str],
        min_periods: int = 20,
        method: str = "zscore",
    ) -> pl.DataFrame:
        """
        Normalize using expanding window (all historical data)

        Args:
            data: Input DataFrame
            columns: Columns to normalize
            min_periods: Minimum periods for calculation
            method: 'zscore' or 'minmax'

        Returns:
            Normalized DataFrame
        """
        logger.info(f"Applying expanding {method} normalization")

        # Convert to pandas for expanding window operations
        data_pd = data.to_pandas()

        for col in columns:
            if col not in data_pd.columns:
                continue

            if method == "zscore":
                # Expanding z-score
                expanding_mean = data_pd[col].expanding(min_periods=min_periods).mean()
                expanding_std = data_pd[col].expanding(min_periods=min_periods).std()

                normalized = (data_pd[col] - expanding_mean) / expanding_std
                data_pd[f"{col}_norm"] = normalized

            elif method == "minmax":
                # Expanding min-max
                expanding_min = data_pd[col].expanding(min_periods=min_periods).min()
                expanding_max = data_pd[col].expanding(min_periods=min_periods).max()

                normalized = (data_pd[col] - expanding_min) / (expanding_max - expanding_min)
                data_pd[f"{col}_norm"] = normalized

        # Convert back to polars
        result = pl.from_pandas(data_pd)

        logger.success("Expanding normalization complete")
        return result

    def detrend(
        self,
        data: pl.DataFrame,
        column: str,
        method: str = "linear",
    ) -> pl.DataFrame:
        """
        Remove trend from time series

        Args:
            data: Input DataFrame
            column: Column to detrend
            method: 'linear' or 'polynomial'

        Returns:
            Detrended DataFrame
        """
        logger.info(f"Detrending {column} using {method} method")

        data_pd = data.to_pandas()

        if column not in data_pd.columns:
            logger.error(f"Column {column} not found")
            return data

        # Create time index
        time_index = np.arange(len(data_pd))

        if method == "linear":
            # Fit linear trend
            coeffs = np.polyfit(time_index, data_pd[column], 1)
            trend = np.polyval(coeffs, time_index)

        elif method == "polynomial":
            # Fit polynomial trend (degree 2)
            coeffs = np.polyfit(time_index, data_pd[column], 2)
            trend = np.polyval(coeffs, time_index)

        else:
            raise ValueError(f"Unknown detrend method: {method}")

        # Remove trend
        data_pd[f"{column}_detrended"] = data_pd[column] - trend
        data_pd[f"{column}_trend"] = trend

        result = pl.from_pandas(data_pd)

        logger.success("Detrending complete")
        return result

    def percentage_change(
        self,
        data: pl.DataFrame,
        columns: list[str],
        periods: int = 1,
    ) -> pl.DataFrame:
        """
        Calculate percentage change

        Args:
            data: Input DataFrame
            columns: Columns to transform
            periods: Periods to shift for calculating change

        Returns:
            DataFrame with percentage changes
        """
        logger.info(f"Calculating {periods}-period percentage changes")

        result = data.clone()

        for col in columns:
            if col not in data.columns:
                continue

            # Calculate percentage change
            pct_change = (data[col] - data[col].shift(periods)) / data[col].shift(periods)
            result = result.with_columns(pct_change.alias(f"{col}_pct_change"))

        logger.success("Percentage change calculation complete")
        return result


def normalize_features(
    data: pl.DataFrame,
    feature_columns: list[str],
    method: str = "standard",
    **kwargs,
) -> tuple[pl.DataFrame, DataNormalizer]:
    """
    Convenience function to normalize features

    Args:
        data: Input DataFrame
        feature_columns: Features to normalize
        method: Normalization method
        **kwargs: Additional parameters

    Returns:
        Tuple of (normalized_data, fitted_normalizer)
    """
    normalizer = DataNormalizer()
    normalized_data = normalizer.fit_transform(data, feature_columns, method, **kwargs)

    return normalized_data, normalizer


def create_rolling_features(
    data: pl.DataFrame,
    columns: list[str],
    windows: list[int] | None = None,
) -> pl.DataFrame:
    """
    Create rolling statistical features

    Args:
        data: Input DataFrame
        columns: Columns to create features from
        windows: Window sizes for rolling calculations

    Returns:
        DataFrame with rolling features
    """
    if windows is None:
        windows = [5, 10, 20, 60]

    logger.info(f"Creating rolling features for {len(columns)} columns")

    result = data.clone()

    for col in columns:
        if col not in data.columns:
            continue

        for window in windows:
            # Rolling mean
            result = result.with_columns(
                data[col].rolling_mean(window).alias(f"{col}_rolling_mean_{window}")
            )

            # Rolling std
            result = result.with_columns(
                data[col].rolling_std(window).alias(f"{col}_rolling_std_{window}")
            )

            # Rolling min/max
            result = result.with_columns(
                data[col].rolling_min(window).alias(f"{col}_rolling_min_{window}")
            )
            result = result.with_columns(
                data[col].rolling_max(window).alias(f"{col}_rolling_max_{window}")
            )

    logger.success(f"Created {(len(result.columns) - len(data.columns))} rolling features")
    return result
