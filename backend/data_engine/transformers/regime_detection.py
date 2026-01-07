# backend/data_engine/transformers/regime_detection.py
"""
Market regime detection using various methods
Identifies bull, bear, and sideways market conditions
"""

from typing import Optional, List, Tuple
import polars as pl
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from hmmlearn import hmm
from loguru import logger
import pickle
from pathlib import Path


class RegimeDetector:
    """
    Detect market regimes using multiple methods

    Methods:
    - Hidden Markov Model (HMM)
    - K-Means clustering
    - Gaussian Mixture Model (GMM)
    - Rule-based (moving averages)
    - Volatility-based
    """

    def __init__(self, n_regimes: int = 3):
        """
        Initialize regime detector

        Args:
            n_regimes: Number of market regimes (typically 3: bull/bear/sideways)
        """
        self.n_regimes = n_regimes
        self.model = None
        self.method: Optional[str] = None
        self.is_fitted = False
        self.feature_names: List[str] = []

        # Regime labels
        self.regime_labels = {
            0: "bear",
            1: "sideways",
            2: "bull",
        }

    def fit_hmm(
        self,
        data: pl.DataFrame,
        feature_columns: List[str],
        n_iter: int = 100,
        covariance_type: str = "full",
    ) -> "RegimeDetector":
        """
        Fit Hidden Markov Model for regime detection

        Args:
            data: Input DataFrame with features
            feature_columns: Features to use for regime detection
            n_iter: Number of EM iterations
            covariance_type: Covariance type ('full', 'diag', 'tied', 'spherical')

        Returns:
            self
        """
        logger.info(f"Fitting HMM with {self.n_regimes} regimes")

        self.method = "hmm"
        self.feature_names = feature_columns

        # Convert to pandas and extract features
        data_pd = data.to_pandas()
        features = data_pd[feature_columns].dropna().values

        if len(features) == 0:
            raise ValueError("No valid data for HMM fitting")

        # Create and fit HMM
        self.model = hmm.GaussianHMM(
            n_components=self.n_regimes,
            covariance_type=covariance_type,
            n_iter=n_iter,
            random_state=42,
        )

        self.model.fit(features)

        self.is_fitted = True
        logger.success(f"HMM fitted with {self.n_regimes} states")
        return self

    def fit_kmeans(
        self,
        data: pl.DataFrame,
        feature_columns: List[str],
        n_init: int = 10,
    ) -> "RegimeDetector":
        """
        Fit K-Means clustering for regime detection

        Args:
            data: Input DataFrame
            feature_columns: Features to use
            n_init: Number of initializations

        Returns:
            self
        """
        logger.info(f"Fitting K-Means with {self.n_regimes} clusters")

        self.method = "kmeans"
        self.feature_names = feature_columns

        # Extract features
        data_pd = data.to_pandas()
        features = data_pd[feature_columns].dropna().values

        # Fit K-Means
        self.model = KMeans(
            n_clusters=self.n_regimes,
            n_init=n_init,
            random_state=42,
        )

        self.model.fit(features)

        self.is_fitted = True
        logger.success(f"K-Means fitted with {self.n_regimes} clusters")
        return self

    def fit_gmm(
        self,
        data: pl.DataFrame,
        feature_columns: List[str],
        covariance_type: str = "full",
    ) -> "RegimeDetector":
        """
        Fit Gaussian Mixture Model for regime detection

        Args:
            data: Input DataFrame
            feature_columns: Features to use
            covariance_type: Covariance type

        Returns:
            self
        """
        logger.info(f"Fitting GMM with {self.n_regimes} components")

        self.method = "gmm"
        self.feature_names = feature_columns

        # Extract features
        data_pd = data.to_pandas()
        features = data_pd[feature_columns].dropna().values

        # Fit GMM
        self.model = GaussianMixture(
            n_components=self.n_regimes,
            covariance_type=covariance_type,
            random_state=42,
        )

        self.model.fit(features)

        self.is_fitted = True
        logger.success(f"GMM fitted with {self.n_regimes} components")
        return self

    def predict(self, data: pl.DataFrame) -> pl.DataFrame:
        """
        Predict market regimes

        Args:
            data: Input DataFrame with features

        Returns:
            DataFrame with regime predictions
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        logger.info(f"Predicting regimes using {self.method}")

        # Extract features
        data_pd = data.to_pandas()
        features = data_pd[self.feature_names].values

        # Predict based on method
        if self.method == "hmm":
            regimes = self.model.predict(features)
            probabilities = self.model.predict_proba(features)

        elif self.method in ["kmeans", "gmm"]:
            regimes = self.model.predict(features)

            if self.method == "gmm":
                probabilities = self.model.predict_proba(features)
            else:
                # K-Means doesn't give probabilities, create dummy
                probabilities = np.zeros((len(regimes), self.n_regimes))
                probabilities[np.arange(len(regimes)), regimes] = 1.0

        else:
            raise ValueError(f"Unknown method: {self.method}")

        # Sort regimes by mean return (bear=0, sideways=1, bull=2)
        regimes = self._sort_regimes(data_pd, regimes)

        # Add to DataFrame
        result = data.clone()
        result = result.with_columns(pl.Series("regime", regimes))

        # Add regime label
        regime_labels = [self.regime_labels[r] for r in regimes]
        result = result.with_columns(pl.Series("regime_label", regime_labels))

        # Add probabilities
        for i in range(self.n_regimes):
            result = result.with_columns(pl.Series(f"regime_{i}_prob", probabilities[:, i]))

        logger.success("Regime prediction complete")
        return result

    def _sort_regimes(self, data: pd.DataFrame, regimes: np.ndarray) -> np.ndarray:
        """
        Sort regimes by mean return (bear < sideways < bull)

        Args:
            data: DataFrame with price data
            regimes: Raw regime predictions

        Returns:
            Sorted regime labels
        """
        if "returns" not in data.columns and "close" in data.columns:
            returns = data["close"].pct_change()
        elif "returns" in data.columns:
            returns = data["returns"]
        else:
            # Cannot sort, return as is
            return regimes

        # Calculate mean return for each regime
        regime_returns = {}
        for r in range(self.n_regimes):
            mask = regimes == r
            regime_returns[r] = returns[mask].mean()

        # Sort regimes by mean return
        sorted_regimes = sorted(regime_returns.items(), key=lambda x: x[1])

        # Create mapping: bear=0, sideways=1, bull=2
        regime_mapping = {old: new for new, (old, _) in enumerate(sorted_regimes)}

        # Apply mapping
        sorted_labels = np.array([regime_mapping[r] for r in regimes])

        return sorted_labels

    def fit_predict(
        self,
        data: pl.DataFrame,
        feature_columns: List[str],
        method: str = "hmm",
        **kwargs,
    ) -> pl.DataFrame:
        """
        Fit and predict in one step

        Args:
            data: Input DataFrame
            feature_columns: Features to use
            method: Detection method ('hmm', 'kmeans', 'gmm')
            **kwargs: Additional parameters for fitting

        Returns:
            DataFrame with regime predictions
        """
        if method == "hmm":
            self.fit_hmm(data, feature_columns, **kwargs)
        elif method == "kmeans":
            self.fit_kmeans(data, feature_columns, **kwargs)
        elif method == "gmm":
            self.fit_gmm(data, feature_columns, **kwargs)
        else:
            raise ValueError(f"Unknown method: {method}")

        return self.predict(data)

    def save(self, path: str) -> None:
        """
        Save detector to disk

        Args:
            path: Path to save file
        """
        state = {
            "model": self.model,
            "method": self.method,
            "n_regimes": self.n_regimes,
            "feature_names": self.feature_names,
            "is_fitted": self.is_fitted,
            "regime_labels": self.regime_labels,
        }

        Path(path).parent.mkdir(parents=True, exist_ok=True)

        with open(path, "wb") as f:
            pickle.dump(state, f)

        logger.info(f"Saved regime detector to {path}")

    def load(self, path: str) -> "RegimeDetector":
        """
        Load detector from disk

        Args:
            path: Path to load file

        Returns:
            self
        """
        with open(path, "rb") as f:
            state = pickle.load(f)

        self.model = state["model"]
        self.method = state["method"]
        self.n_regimes = state["n_regimes"]
        self.feature_names = state["feature_names"]
        self.is_fitted = state["is_fitted"]
        self.regime_labels = state["regime_labels"]

        logger.info(f"Loaded regime detector from {path}")
        return self


class RuleBasedRegimeDetector:
    """
    Simple rule-based regime detection using technical indicators

    No training required - uses predefined rules
    """

    def __init__(self):
        """Initialize rule-based detector"""
        pass

    def detect_ma_regime(
        self,
        data: pl.DataFrame,
        short_window: int = 50,
        long_window: int = 200,
    ) -> pl.DataFrame:
        """
        Detect regime using moving average crossover

        Args:
            data: DataFrame with 'close' column
            short_window: Short MA window
            long_window: Long MA window

        Returns:
            DataFrame with regime labels
        """
        logger.info(f"Detecting MA regime ({short_window}/{long_window})")

        if "close" not in data.columns:
            raise ValueError("Data must contain 'close' column")

        # Calculate moving averages
        result = data.with_columns(
            [
                pl.col("close").rolling_mean(short_window).alias("ma_short"),
                pl.col("close").rolling_mean(long_window).alias("ma_long"),
            ]
        )

        # Determine regime
        # Bull: short MA > long MA and price > long MA
        # Bear: short MA < long MA and price < long MA
        # Sideways: otherwise

        regime = []
        regime_label = []

        for row in result.iter_rows(named=True):
            price = row["close"]
            ma_short = row["ma_short"]
            ma_long = row["ma_long"]

            if ma_short is None or ma_long is None:
                regime.append(1)  # Sideways (default)
                regime_label.append("sideways")
                continue

            if ma_short > ma_long and price > ma_long:
                regime.append(2)  # Bull
                regime_label.append("bull")
            elif ma_short < ma_long and price < ma_long:
                regime.append(0)  # Bear
                regime_label.append("bear")
            else:
                regime.append(1)  # Sideways
                regime_label.append("sideways")

        result = result.with_columns(
            [
                pl.Series("regime", regime),
                pl.Series("regime_label", regime_label),
            ]
        )

        logger.success("MA regime detection complete")
        return result

    def detect_volatility_regime(
        self,
        data: pl.DataFrame,
        window: int = 20,
        high_vol_threshold: float = 0.02,
        low_vol_threshold: float = 0.01,
    ) -> pl.DataFrame:
        """
        Detect regime based on volatility levels

        Args:
            data: DataFrame with returns
            window: Window for volatility calculation
            high_vol_threshold: Threshold for high volatility
            low_vol_threshold: Threshold for low volatility

        Returns:
            DataFrame with volatility regime
        """
        logger.info("Detecting volatility regime")

        # Calculate returns if not present
        if "returns" not in data.columns and "close" in data.columns:
            result = data.with_columns(pl.col("close").pct_change().alias("returns"))
        else:
            result = data.clone()

        # Calculate rolling volatility
        result = result.with_columns(pl.col("returns").rolling_std(window).alias("volatility"))

        # Classify regime
        vol_regime = []
        vol_regime_label = []

        for vol in result["volatility"]:
            if vol is None:
                vol_regime.append(1)
                vol_regime_label.append("normal")
            elif vol > high_vol_threshold:
                vol_regime.append(2)
                vol_regime_label.append("high_volatility")
            elif vol < low_vol_threshold:
                vol_regime.append(0)
                vol_regime_label.append("low_volatility")
            else:
                vol_regime.append(1)
                vol_regime_label.append("normal")

        result = result.with_columns(
            [
                pl.Series("vol_regime", vol_regime),
                pl.Series("vol_regime_label", vol_regime_label),
            ]
        )

        logger.success("Volatility regime detection complete")
        return result

    def detect_trend_regime(
        self,
        data: pl.DataFrame,
        window: int = 50,
        trend_threshold: float = 0.15,
    ) -> pl.DataFrame:
        """
        Detect trend-based regime

        Args:
            data: DataFrame with 'close' column
            window: Lookback window
            trend_threshold: Threshold for trend classification

        Returns:
            DataFrame with trend regime
        """
        logger.info("Detecting trend regime")

        if "close" not in data.columns:
            raise ValueError("Data must contain 'close' column")

        # Calculate percentage change over window
        result = data.with_columns((pl.col("close").pct_change(window)).alias("trend"))

        # Classify regime
        trend_regime = []
        trend_label = []

        for trend in result["trend"]:
            if trend is None:
                trend_regime.append(1)
                trend_label.append("sideways")
            elif trend > trend_threshold:
                trend_regime.append(2)
                trend_label.append("bull")
            elif trend < -trend_threshold:
                trend_regime.append(0)
                trend_label.append("bear")
            else:
                trend_regime.append(1)
                trend_label.append("sideways")

        result = result.with_columns(
            [
                pl.Series("trend_regime", trend_regime),
                pl.Series("trend_regime_label", trend_label),
            ]
        )

        logger.success("Trend regime detection complete")
        return result


def detect_regimes(
    data: pl.DataFrame,
    method: str = "hmm",
    feature_columns: Optional[List[str]] = None,
    **kwargs,
) -> Tuple[pl.DataFrame, RegimeDetector]:
    """
    Convenience function to detect market regimes

    Args:
        data: Input DataFrame
        method: Detection method ('hmm', 'kmeans', 'gmm', 'ma', 'volatility', 'trend')
        feature_columns: Features to use (for ML methods)
        **kwargs: Additional parameters

    Returns:
        Tuple of (data_with_regimes, fitted_detector)
    """
    # Rule-based methods
    if method in ["ma", "volatility", "trend"]:
        detector = RuleBasedRegimeDetector()

        if method == "ma":
            result = detector.detect_ma_regime(data, **kwargs)
        elif method == "volatility":
            result = detector.detect_volatility_regime(data, **kwargs)
        elif method == "trend":
            result = detector.detect_trend_regime(data, **kwargs)

        return result, detector

    # ML-based methods
    if feature_columns is None:
        raise ValueError("feature_columns required for ML methods")

    detector = RegimeDetector(n_regimes=kwargs.get("n_regimes", 3))
    result = detector.fit_predict(data, feature_columns, method, **kwargs)

    return result, detector


def combine_regime_signals(
    data: pl.DataFrame,
    regime_columns: List[str],
    method: str = "majority",
) -> pl.DataFrame:
    """
    Combine multiple regime signals into consensus

    Args:
        data: DataFrame with multiple regime predictions
        regime_columns: Columns containing regime predictions
        method: Combination method ('majority', 'weighted', 'conservative')

    Returns:
        DataFrame with combined regime
    """
    logger.info(f"Combining {len(regime_columns)} regime signals")

    if method == "majority":
        # Majority vote
        regime_values = data.select(regime_columns).to_numpy()

        # Get most common regime for each row
        combined = []
        for row in regime_values:
            unique, counts = np.unique(row, return_counts=True)
            combined.append(unique[np.argmax(counts)])

        result = data.with_columns(pl.Series("combined_regime", combined))

    elif method == "conservative":
        # Conservative: only bull if all agree bull, only bear if all agree bear
        regime_values = data.select(regime_columns).to_numpy()

        combined = []
        for row in regime_values:
            if np.all(row == 2):
                combined.append(2)  # Bull
            elif np.all(row == 0):
                combined.append(0)  # Bear
            else:
                combined.append(1)  # Sideways

        result = data.with_columns(pl.Series("combined_regime", combined))

    else:
        raise ValueError(f"Unknown combination method: {method}")

    # Add label
    labels = {0: "bear", 1: "sideways", 2: "bull"}
    result = result.with_columns(
        pl.col("combined_regime").map_dict(labels).alias("combined_regime_label")
    )

    logger.success("Regime combination complete")
    return result
