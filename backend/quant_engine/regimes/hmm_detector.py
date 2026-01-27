# backend/quant_engine/regimes/hmm_detector.py
"""
Hidden Markov Model (HMM) for Market Regime Detection

Implements HMM-based regime detection for financial time series.
HMM is particularly effective for regime detection as it can model:
- Hidden states (regimes) with observable features
- Temporal dependencies between regimes
- Regime persistence and transition dynamics

The model assumes that markets exist in discrete hidden states (regimes)
and that observable features (returns, volatility) are generated from
regime-specific distributions.
"""

import numpy as np
import pandas as pd
import polars as pl
from hmmlearn import hmm
from loguru import logger
from sklearn.preprocessing import StandardScaler


class HMMRegimeDetector:
    """
    Hidden Markov Model for market regime detection

    Uses Gaussian HMM to detect latent market states based on observable
    features like returns and volatility. The model learns:
    1. Transition probabilities between regimes
    2. Initial regime probabilities
    3. Emission distributions for each regime
    """

    def __init__(
        self,
        n_regimes: int = 3,
        covariance_type: str = "full",
        n_iter: int = 100,
        random_state: int = 42,
    ):
        """
        Initialize HMM regime detector

        Args:
            n_regimes: Number of hidden states (market regimes)
            covariance_type: Type of covariance matrix
                - 'full': Each state has its own full covariance matrix
                - 'diag': Diagonal covariance (features independent)
                - 'tied': All states share same covariance
                - 'spherical': Single variance parameter per state
            n_iter: Maximum number of EM iterations
            random_state: Random seed for reproducibility
        """
        self.n_regimes = n_regimes
        self.covariance_type = covariance_type
        self.n_iter = n_iter
        self.random_state = random_state

        # Model components
        self.model: hmm.GaussianHMM | None = None
        self.scaler = StandardScaler()
        self.is_fitted = False

        # Feature tracking
        self.feature_names: list[str] = []

        # Results storage
        self.transition_matrix_: np.ndarray | None = None
        self.means_: np.ndarray | None = None
        self.covariances_: np.ndarray | None = None

        # Regime interpretation
        self.regime_labels = self._initialize_regime_labels()

        logger.info(
            f"HMM regime detector initialized: n_regimes={n_regimes}, covariance={covariance_type}"
        )

    def _initialize_regime_labels(self) -> dict[int, str]:
        """Initialize human-readable regime labels"""
        if self.n_regimes == 2:
            return {0: "low_volatility", 1: "high_volatility"}
        elif self.n_regimes == 3:
            return {0: "bear", 1: "neutral", 2: "bull"}
        elif self.n_regimes == 4:
            return {0: "crisis", 1: "bear", 2: "neutral", 3: "bull"}
        else:
            return {i: f"regime_{i}" for i in range(self.n_regimes)}

    def prepare_features(
        self,
        data: pl.DataFrame,
        feature_windows: list[int] | None = None,
    ) -> pl.DataFrame:
        """
        Prepare features for HMM

        Typical features for regime detection:
        - Returns at multiple horizons
        - Volatility at multiple horizons
        - Volume changes (if available)

        Args:
            data: DataFrame with OHLCV data
            feature_windows: List of lookback windows for feature calculation

        Returns:
            DataFrame with calculated features
        """
        feature_windows = feature_windows or [1, 5, 20]

        logger.info(f"Preparing HMM features with windows: {feature_windows}")

        result = data.clone()
        features = []

        # Calculate returns
        result = result.with_columns([pl.col("close").pct_change().alias("returns")])

        for window in feature_windows:
            # Log returns over window
            return_col = f"returns_{window}d"
            result = result.with_columns([pl.col("close").pct_change(window).alias(return_col)])
            features.append(return_col)

            # Realized volatility (std of returns)
            vol_col = f"volatility_{window}d"
            result = result.with_columns([pl.col("returns").rolling_std(window).alias(vol_col)])
            features.append(vol_col)

            # Range-based volatility (high-low)
            if "high" in data.columns and "low" in data.columns:
                range_vol_col = f"range_vol_{window}d"
                result = result.with_columns(
                    [
                        ((pl.col("high") - pl.col("low")) / pl.col("close"))
                        .rolling_mean(window)
                        .alias(range_vol_col)
                    ]
                )
                features.append(range_vol_col)

        # Volume features (if available)
        if "volume" in data.columns:
            result = result.with_columns([pl.col("volume").pct_change().alias("volume_change")])
            features.append("volume_change")

            # Volume z-score
            result = result.with_columns(
                [
                    (
                        (pl.col("volume") - pl.col("volume").rolling_mean(20))
                        / pl.col("volume").rolling_std(20)
                    ).alias("volume_zscore")
                ]
            )
            features.append("volume_zscore")

        self.feature_names = features
        logger.success(f"Prepared {len(features)} features for HMM")

        return result

    def fit(
        self,
        data: pl.DataFrame,
        features: list[str] | None = None,
    ) -> "HMMRegimeDetector":
        """
        Fit HMM to historical data

        Args:
            data: DataFrame with features
            features: List of feature column names (if None, use all prepared features)

        Returns:
            self
        """
        # Select features
        if features is None:
            if not self.feature_names:
                raise ValueError("No features available. Call prepare_features first.")
            features = self.feature_names

        # Extract feature matrix
        X_df = data.select(features).to_pandas()
        X = X_df.dropna().values

        if len(X) < 100:
            logger.warning(f"Small sample size for HMM: {len(X)} observations")

        logger.info(f"Fitting HMM with {len(X)} observations and {X.shape[1]} features")

        # Standardize features
        X_scaled = self.scaler.fit_transform(X)

        # Create and fit HMM
        self.model = hmm.GaussianHMM(
            n_components=self.n_regimes,
            covariance_type=self.covariance_type,
            n_iter=self.n_iter,
            random_state=self.random_state,
            verbose=False,
        )

        # Fit model
        self.model.fit(X_scaled)

        # Store learned parameters
        self.transition_matrix_ = self.model.transmat_
        self.means_ = self.model.means_
        self.covariances_ = self.model.covars_

        # Check convergence
        if self.model.monitor_.converged:
            logger.success(f"HMM converged after {len(self.model.monitor_.history)} iterations")
        else:
            logger.warning("HMM did not converge within iteration limit")

        self.is_fitted = True

        return self

    def predict(
        self,
        data: pl.DataFrame,
        features: list[str] | None = None,
        return_probabilities: bool = False,
    ) -> pl.DataFrame:
        """
        Predict regimes for new data

        Args:
            data: DataFrame with features
            features: List of feature column names
            return_probabilities: If True, also return regime probabilities

        Returns:
            DataFrame with regime predictions (and probabilities if requested)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        # Select features
        if features is None:
            features = self.feature_names

        # Extract features
        X_df = data.select(features).to_pandas()
        X = X_df.values

        # Scale features
        X_scaled = self.scaler.transform(X)

        # Predict most likely sequence of states (Viterbi algorithm)
        regimes = self.model.predict(X_scaled)

        # Order regimes by mean return
        regimes = self._order_regimes(data, regimes, features)

        # Create regime labels
        regime_labels = [self.regime_labels.get(r, f"regime_{r}") for r in regimes]

        # Build result dataframe
        result = data.with_columns(
            [
                pl.Series("regime", regimes),
                pl.Series("regime_label", regime_labels),
            ]
        )

        # Add probabilities if requested
        if return_probabilities:
            # Predict probabilities for each regime
            probs = self.model.predict_proba(X_scaled)

            for i in range(self.n_regimes):
                prob_col = f"regime_{i}_prob"
                result = result.with_columns([pl.Series(prob_col, probs[:, i])])

        logger.info(f"Predicted regimes for {len(data)} observations")

        return result

    def fit_predict(
        self,
        data: pl.DataFrame,
        features: list[str] | None = None,
        return_probabilities: bool = False,
    ) -> pl.DataFrame:
        """
        Fit HMM and predict regimes in one step

        Args:
            data: DataFrame with features
            features: List of feature column names
            return_probabilities: If True, return regime probabilities

        Returns:
            DataFrame with regime predictions
        """
        self.fit(data, features)
        return self.predict(data, features, return_probabilities)

    def _order_regimes(
        self,
        data: pl.DataFrame,
        regimes: np.ndarray,
        features: list[str],
    ) -> np.ndarray:
        """
        Order regimes by characteristic (typically mean return)

        Ensures regime 0 = bearish, regime n-1 = bullish

        Args:
            data: Original dataframe
            regimes: Predicted regime labels
            features: Feature names

        Returns:
            Re-ordered regime labels
        """
        # Calculate mean return for each regime
        regime_characteristics = {}

        # Find a return feature to use for ordering
        return_features = [f for f in features if "return" in f.lower()]
        if not return_features:
            # No return feature, use first feature
            characteristic_feature = features[0]
        else:
            # Use shortest-horizon return
            characteristic_feature = sorted(return_features, key=len)[0]

        for regime_id in range(self.n_regimes):
            mask = regimes == regime_id
            if mask.sum() > 0:
                regime_data = data.filter(pl.Series(mask)).select(characteristic_feature)
                mean_val = regime_data.mean().to_numpy()
                regime_characteristics[regime_id] = mean_val[0] if len(mean_val) > 0 else 0
            else:
                regime_characteristics[regime_id] = 0

        # Sort regimes by characteristic
        sorted_regimes = sorted(regime_characteristics.items(), key=lambda x: x[1])
        regime_mapping = {old: new for new, (old, _) in enumerate(sorted_regimes)}

        # Remap regimes
        ordered_regimes = np.array([regime_mapping[r] for r in regimes])

        # Update regime labels to match new ordering
        self.regime_labels = {
            regime_mapping[old_id]: label for old_id, label in self.regime_labels.items()
        }

        return ordered_regimes

    def get_transition_matrix(self) -> pd.DataFrame:
        """
        Get regime transition probability matrix

        Returns:
            DataFrame with transition probabilities
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")

        regime_names = [self.regime_labels.get(i, f"regime_{i}") for i in range(self.n_regimes)]

        transition_df = pd.DataFrame(
            self.transition_matrix_,
            index=regime_names,
            columns=regime_names,
        )

        return transition_df

    def get_regime_characteristics(self) -> pd.DataFrame:
        """
        Get characteristics of each regime (means and covariances)

        Returns:
            DataFrame with regime statistics
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")

        # Create stats dataframe
        stats = []

        for i in range(self.n_regimes):
            regime_stats = {
                "regime": i,
                "regime_label": self.regime_labels.get(i, f"regime_{i}"),
            }

            # Add feature means
            for j, feature_name in enumerate(self.feature_names):
                # Inverse transform to get original scale
                feature_mean_scaled = self.means_[i, j]
                feature_mean = (feature_mean_scaled * self.scaler.scale_[j]) + self.scaler.mean_[j]
                regime_stats[f"mean_{feature_name}"] = feature_mean

            # Add variance (for diagonal/spherical covariance)
            if self.covariance_type in ["diag", "spherical"]:
                for j, feature_name in enumerate(self.feature_names):
                    if self.covariance_type == "diag":
                        variance = self.covariances_[i, j]
                    else:  # spherical
                        variance = self.covariances_[i]

                    # Scale back
                    variance_original = variance * (self.scaler.scale_[j] ** 2)
                    regime_stats[f"var_{feature_name}"] = variance_original

            stats.append(regime_stats)

        return pd.DataFrame(stats)

    def calculate_regime_persistence(
        self,
        data: pl.DataFrame,
    ) -> pd.DataFrame:
        """
        Calculate average duration (persistence) of each regime

        Args:
            data: DataFrame with regime predictions

        Returns:
            DataFrame with regime persistence statistics
        """
        if "regime" not in data.columns:
            raise ValueError("Data must contain 'regime' column")

        regimes = data.select("regime").to_series().to_numpy()

        # Calculate run lengths for each regime
        regime_durations = {i: [] for i in range(self.n_regimes)}

        current_regime = regimes[0]
        current_duration = 1

        for i in range(1, len(regimes)):
            if regimes[i] == current_regime:
                current_duration += 1
            else:
                regime_durations[current_regime].append(current_duration)
                current_regime = regimes[i]
                current_duration = 1

        # Add last duration
        regime_durations[current_regime].append(current_duration)

        # Calculate statistics
        persistence_stats = []

        for regime_id in range(self.n_regimes):
            durations = regime_durations[regime_id]

            if durations:
                stats = {
                    "regime": regime_id,
                    "regime_label": self.regime_labels.get(regime_id, f"regime_{regime_id}"),
                    "n_episodes": len(durations),
                    "mean_duration": np.mean(durations),
                    "median_duration": np.median(durations),
                    "max_duration": np.max(durations),
                    "min_duration": np.min(durations),
                }
            else:
                stats = {
                    "regime": regime_id,
                    "regime_label": self.regime_labels.get(regime_id, f"regime_{regime_id}"),
                    "n_episodes": 0,
                    "mean_duration": 0,
                    "median_duration": 0,
                    "max_duration": 0,
                    "min_duration": 0,
                }

            persistence_stats.append(stats)

        return pd.DataFrame(persistence_stats)

    def score(self, data: pl.DataFrame, features: list[str] | None = None) -> float:
        """
        Calculate log-likelihood score of data under fitted model

        Args:
            data: DataFrame with features
            features: Feature column names

        Returns:
            Log-likelihood score
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")

        if features is None:
            features = self.feature_names

        X_df = data.select(features).to_pandas()
        X = X_df.dropna().values
        X_scaled = self.scaler.transform(X)

        return self.model.score(X_scaled)


def detect_regimes_hmm(
    data: pl.DataFrame,
    n_regimes: int = 3,
    feature_windows: list[int] | None = None,
    covariance_type: str = "full",
    n_iter: int = 100,
    return_probabilities: bool = False,
) -> tuple[pl.DataFrame, HMMRegimeDetector]:
    """
    Convenience function for HMM regime detection

    Args:
        data: DataFrame with OHLCV data
        n_regimes: Number of market regimes
        feature_windows: Lookback windows for feature calculation
        covariance_type: Covariance matrix type
        n_iter: Maximum EM iterations
        return_probabilities: Whether to return regime probabilities

    Returns:
        Tuple of (data_with_regimes, fitted_detector)

    Example:
        >>> data_with_regimes, detector = detect_regimes_hmm(
        ...     ohlcv_data,
        ...     n_regimes=3,
        ...     feature_windows=[1, 5, 20],
        ... )
        >>> transition_matrix = detector.get_transition_matrix()
        >>> persistence = detector.calculate_regime_persistence(data_with_regimes)
    """
    # Initialize detector
    detector = HMMRegimeDetector(
        n_regimes=n_regimes,
        covariance_type=covariance_type,
        n_iter=n_iter,
    )

    # Prepare features
    data_with_features = detector.prepare_features(data, feature_windows)

    # Fit and predict
    result = detector.fit_predict(data_with_features, return_probabilities=return_probabilities)

    logger.success(f"HMM regime detection complete: {n_regimes} regimes")

    return result, detector
