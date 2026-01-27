# backend/quant_engine/regimes/clustering.py
"""
Clustering-Based Market Regime Detection

Uses unsupervised learning techniques to identify distinct market regimes
based on various market features. Supports multiple clustering algorithms
and feature sets for robust regime identification.

Methods:
- K-Means clustering
- Gaussian Mixture Models (GMM)
- DBSCAN for anomaly detection
- Agglomerative clustering
"""

from typing import Any

import numpy as np
import pandas as pd
import polars as pl
from loguru import logger
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler


class ClusteringRegimeDetector:
    """
    Detect market regimes using clustering algorithms

    Identifies distinct market states by clustering historical market data
    based on features like returns, volatility, volume, and momentum.
    """

    def __init__(
        self,
        n_regimes: int = 3,
        method: str = "kmeans",
        random_state: int = 42,
    ):
        """
        Initialize clustering regime detector

        Args:
            n_regimes: Number of market regimes to detect (typically 2-5)
            method: Clustering method ('kmeans', 'gmm', 'dbscan', 'hierarchical')
            random_state: Random seed for reproducibility
        """
        self.n_regimes = n_regimes
        self.method = method
        self.random_state = random_state

        # Model and preprocessing
        self.model = None
        self.scaler = StandardScaler()
        self.is_fitted = False

        # Feature storage
        self.feature_names: list[str] = []
        self.regime_centers: np.ndarray | None = None

        # Regime labels mapping
        self.regime_labels = self._initialize_regime_labels()

        logger.info(
            f"Clustering regime detector initialized: method={method}, n_regimes={n_regimes}"
        )

    def _initialize_regime_labels(self) -> dict[int, str]:
        """Initialize regime label names"""
        if self.n_regimes == 2:
            return {0: "bearish", 1: "bullish"}
        elif self.n_regimes == 3:
            return {0: "bear", 1: "neutral", 2: "bull"}
        elif self.n_regimes == 4:
            return {0: "crisis", 1: "bear", 2: "neutral", 3: "bull"}
        else:
            return {i: f"regime_{i}" for i in range(self.n_regimes)}

    def extract_features(
        self,
        data: pl.DataFrame,
        feature_config: dict[str, Any] | None = None,
    ) -> pl.DataFrame:
        """
        Extract features for regime detection

        Args:
            data: DataFrame with OHLCV data
            feature_config: Configuration for feature extraction
                - 'windows': List of lookback windows for features
                - 'include_volume': Whether to include volume features
                - 'include_momentum': Whether to include momentum features

        Returns:
            DataFrame with extracted features
        """
        feature_config = feature_config or {}
        windows = feature_config.get("windows", [5, 20, 60])
        include_volume = feature_config.get("include_volume", True)
        include_momentum = feature_config.get("include_momentum", True)

        logger.info(f"Extracting features for regime detection with windows: {windows}")

        result = data.clone()
        features = []

        # Returns features
        result = result.with_columns([pl.col("close").pct_change().alias("return_1d")])
        features.append("return_1d")

        for window in windows:
            # Rolling returns
            col_name = f"return_{window}d"
            result = result.with_columns([pl.col("close").pct_change(window).alias(col_name)])
            features.append(col_name)

            # Volatility (rolling std of returns)
            vol_name = f"volatility_{window}d"
            result = result.with_columns([pl.col("return_1d").rolling_std(window).alias(vol_name)])
            features.append(vol_name)

        # Volume features
        if include_volume and "volume" in data.columns:
            result = result.with_columns([pl.col("volume").pct_change().alias("volume_change")])
            features.append("volume_change")

            for window in windows:
                vol_ratio_name = f"volume_ratio_{window}d"
                result = result.with_columns(
                    [
                        (pl.col("volume") / pl.col("volume").rolling_mean(window)).alias(
                            vol_ratio_name
                        )
                    ]
                )
                features.append(vol_ratio_name)

        # Momentum features
        if include_momentum:
            for window in windows:
                # Price momentum (current vs SMA)
                mom_name = f"momentum_{window}d"
                result = result.with_columns(
                    [((pl.col("close") / pl.col("close").rolling_mean(window)) - 1).alias(mom_name)]
                )
                features.append(mom_name)

                # RSI-like feature
                rsi_name = f"rsi_{window}d"
                gains = pl.col("return_1d").clip_min(0).rolling_mean(window)
                losses = (-pl.col("return_1d").clip_max(0)).rolling_mean(window)
                result = result.with_columns([(gains / (gains + losses + 1e-10)).alias(rsi_name)])
                features.append(rsi_name)

        self.feature_names = features
        logger.success(f"Extracted {len(features)} features")

        return result

    def fit(
        self,
        data: pl.DataFrame,
        features: list[str] | None = None,
    ) -> "ClusteringRegimeDetector":
        """
        Fit clustering model to detect regimes

        Args:
            data: DataFrame with features
            features: List of feature column names (if None, use all numeric columns)

        Returns:
            self
        """
        # Select features
        if features is None:
            features = self.feature_names if self.feature_names else data.columns

        # Extract feature matrix
        feature_data = data.select(features).to_pandas()
        X = feature_data.dropna().values

        if len(X) == 0:
            raise ValueError("No valid data for clustering")

        logger.info(f"Fitting {self.method} clustering with {X.shape[0]} samples")

        # Standardize features
        X_scaled = self.scaler.fit_transform(X)

        # Fit clustering model
        if self.method == "kmeans":
            self.model = KMeans(
                n_clusters=self.n_regimes,
                n_init=10,
                random_state=self.random_state,
            )
            self.model.fit(X_scaled)
            self.regime_centers = self.model.cluster_centers_

        elif self.method == "gmm":
            self.model = GaussianMixture(
                n_components=self.n_regimes,
                covariance_type="full",
                random_state=self.random_state,
            )
            self.model.fit(X_scaled)
            self.regime_centers = self.model.means_

        elif self.method == "hierarchical":
            self.model = AgglomerativeClustering(
                n_clusters=self.n_regimes,
                linkage="ward",
            )
            self.model.fit(X_scaled)
            # Calculate centers manually for hierarchical
            labels = self.model.labels_
            self.regime_centers = np.array(
                [X_scaled[labels == i].mean(axis=0) for i in range(self.n_regimes)]
            )

        elif self.method == "dbscan":
            self.model = DBSCAN(
                eps=0.5,
                min_samples=10,
            )
            self.model.fit(X_scaled)
            # No fixed centers for DBSCAN
            self.regime_centers = None

        else:
            raise ValueError(f"Unknown clustering method: {self.method}")

        self.is_fitted = True
        logger.success(f"Clustering model fitted successfully")

        return self

    def predict(
        self,
        data: pl.DataFrame,
        features: list[str] | None = None,
    ) -> pl.DataFrame:
        """
        Predict market regimes for new data

        Args:
            data: DataFrame with features
            features: List of feature column names

        Returns:
            DataFrame with regime predictions
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        # Select features
        if features is None:
            features = self.feature_names

        # Extract feature matrix
        feature_data = data.select(features).to_pandas()
        X = feature_data.values

        # Scale features
        X_scaled = self.scaler.transform(X)

        # Predict regimes
        if self.method == "gmm":
            regimes = self.model.predict(X_scaled)
        else:
            regimes = (
                self.model.fit_predict(X_scaled)
                if self.method == "dbscan"
                else self.model.predict(X_scaled)
            )

        # Map regimes to ordered labels (sort by mean return or volatility)
        regimes = self._order_regimes(data, regimes, features)

        # Add predictions to dataframe
        regime_labels = [self.regime_labels.get(r, f"regime_{r}") for r in regimes]

        result = data.with_columns(
            [
                pl.Series("regime", regimes),
                pl.Series("regime_label", regime_labels),
            ]
        )

        logger.info(f"Predicted regimes for {len(data)} samples")

        return result

    def fit_predict(
        self,
        data: pl.DataFrame,
        features: list[str] | None = None,
    ) -> pl.DataFrame:
        """
        Fit model and predict regimes in one step

        Args:
            data: DataFrame with features
            features: List of feature column names

        Returns:
            DataFrame with regime predictions
        """
        self.fit(data, features)
        return self.predict(data, features)

    def _order_regimes(
        self,
        data: pl.DataFrame,
        regimes: np.ndarray,
        features: list[str],
    ) -> np.ndarray:
        """
        Order regimes by market characteristic (e.g., returns)

        Ensures regime 0 = bearish, regime n-1 = bullish

        Args:
            data: Original dataframe
            regimes: Predicted regime labels
            features: Feature names

        Returns:
            Re-ordered regime labels
        """
        if self.method == "dbscan":
            # DBSCAN doesn't guarantee fixed number of clusters
            return regimes

        # Calculate mean return for each regime
        regime_returns = {}

        for regime_id in range(self.n_regimes):
            mask = regimes == regime_id
            if mask.sum() > 0:
                # Use first return feature as proxy for regime characteristic
                return_feature = [f for f in features if "return" in f.lower()][0]
                regime_data = data.filter(pl.Series(mask)).select(return_feature)
                regime_returns[regime_id] = regime_data.mean().to_numpy()[0]
            else:
                regime_returns[regime_id] = 0

        # Sort regimes by mean return
        sorted_regimes = sorted(regime_returns.items(), key=lambda x: x[1])
        regime_mapping = {old: new for new, (old, _) in enumerate(sorted_regimes)}

        # Remap regimes
        ordered_regimes = np.array([regime_mapping.get(r, r) for r in regimes])

        return ordered_regimes

    def get_regime_statistics(
        self,
        data: pl.DataFrame,
    ) -> pd.DataFrame:
        """
        Calculate statistics for each detected regime

        Args:
            data: DataFrame with regime predictions

        Returns:
            DataFrame with regime statistics
        """
        if "regime" not in data.columns:
            raise ValueError("Data must contain 'regime' column")

        stats = []

        for regime_id in range(self.n_regimes):
            regime_data = data.filter(pl.col("regime") == regime_id)

            if regime_data.height == 0:
                continue

            stat_dict = {
                "regime": regime_id,
                "regime_label": self.regime_labels.get(regime_id, f"regime_{regime_id}"),
                "n_samples": regime_data.height,
                "frequency": regime_data.height / data.height,
            }

            # Calculate metrics if available
            if "return_1d" in regime_data.columns:
                returns = regime_data.select("return_1d").to_series()
                stat_dict.update(
                    {
                        "mean_return": returns.mean(),
                        "std_return": returns.std(),
                        "sharpe": returns.mean() / returns.std() if returns.std() > 0 else 0,
                    }
                )

            if "volatility_20d" in regime_data.columns:
                vol = regime_data.select("volatility_20d").to_series()
                stat_dict["mean_volatility"] = vol.mean()

            stats.append(stat_dict)

        stats_df = pd.DataFrame(stats)
        logger.info(f"Calculated statistics for {len(stats)} regimes")

        return stats_df

    def get_regime_transitions(
        self,
        data: pl.DataFrame,
    ) -> pd.DataFrame:
        """
        Calculate regime transition probabilities

        Args:
            data: DataFrame with regime predictions (must have 'regime' column)

        Returns:
            Transition probability matrix as DataFrame
        """
        if "regime" not in data.columns:
            raise ValueError("Data must contain 'regime' column")

        regimes = data.select("regime").to_series().to_numpy()

        # Initialize transition matrix
        transition_matrix = np.zeros((self.n_regimes, self.n_regimes))

        # Count transitions
        for i in range(len(regimes) - 1):
            from_regime = regimes[i]
            to_regime = regimes[i + 1]

            if from_regime >= 0 and to_regime >= 0:  # Ignore -1 (noise in DBSCAN)
                transition_matrix[int(from_regime), int(to_regime)] += 1

        # Normalize to probabilities
        row_sums = transition_matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        transition_probs = transition_matrix / row_sums

        # Convert to DataFrame
        regime_names = [self.regime_labels.get(i, f"regime_{i}") for i in range(self.n_regimes)]
        transition_df = pd.DataFrame(
            transition_probs,
            index=regime_names,
            columns=regime_names,
        )

        logger.info("Calculated regime transition probabilities")

        return transition_df


def detect_regimes_clustering(
    data: pl.DataFrame,
    n_regimes: int = 3,
    method: str = "kmeans",
    feature_config: dict[str, Any] | None = None,
) -> tuple[pl.DataFrame, ClusteringRegimeDetector]:
    """
    Convenience function for regime detection using clustering

    Args:
        data: DataFrame with OHLCV data
        n_regimes: Number of regimes to detect
        method: Clustering method ('kmeans', 'gmm', 'hierarchical', 'dbscan')
        feature_config: Feature extraction configuration

    Returns:
        Tuple of (data_with_regimes, fitted_detector)

    Example:
        >>> data_with_regimes, detector = detect_regimes_clustering(
        ...     ohlcv_data,
        ...     n_regimes=3,
        ...     method='kmeans',
        ...     feature_config={'windows': [5, 20, 60]}
        ... )
        >>> stats = detector.get_regime_statistics(data_with_regimes)
    """
    # Initialize detector
    detector = ClusteringRegimeDetector(
        n_regimes=n_regimes,
        method=method,
    )

    # Extract features
    data_with_features = detector.extract_features(data, feature_config)

    # Fit and predict
    result = detector.fit_predict(data_with_features)

    logger.success(f"Regime detection complete using {method}")

    return result, detector
