# backend/quant_engine/factors/pca_analysis.py
"""
Principal Component Analysis (PCA) for Factor Analysis

This module provides PCA-based factor analysis for financial data.
Useful for dimensionality reduction, identifying latent factors,
and understanding return covariance structure.

Key Features:
- Standard PCA with automatic scaling
- Incremental PCA for large datasets
- Factor interpretation and loading analysis
- Variance explained analysis
- Factor score calculation
- Rolling PCA for time-varying analysis

Applications in Finance:
- Identifying statistical factors in returns
- Risk decomposition
- Portfolio construction (PCA-based strategies)
- Dimensionality reduction for ML models
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.preprocessing import StandardScaler

# ============================================================================
# ENUMS AND DATA CLASSES
# ============================================================================


class ScalingMethod(Enum):
    """Data scaling methods"""

    NONE = "none"
    STANDARD = "standard"  # Z-score standardization
    DEMEAN = "demean"  # Only subtract mean


class SelectionMethod(Enum):
    """Methods for selecting number of components"""

    FIXED = "fixed"  # Fixed number of components
    VARIANCE = "variance"  # Explain X% of variance
    KAISER = "kaiser"  # Eigenvalue > 1 criterion
    SCREE = "scree"  # Elbow method (automatic)
    PARALLEL = "parallel"  # Parallel analysis


@dataclass
class PCAResults:
    """Results from PCA analysis"""

    # Number of components
    n_components: int
    n_features: int
    n_samples: int

    # Explained variance
    explained_variance: np.ndarray  # Variance explained by each component
    explained_variance_ratio: np.ndarray  # Proportion of variance
    cumulative_variance_ratio: np.ndarray  # Cumulative proportion

    # Components (loadings)
    components: np.ndarray  # Shape: (n_components, n_features)
    loadings: pd.DataFrame  # Components as DataFrame with feature names

    # Eigenvalues
    eigenvalues: np.ndarray

    # Factor scores
    scores: np.ndarray | None = None  # Shape: (n_samples, n_components)
    scores_df: pd.DataFrame | None = None

    # Feature names
    feature_names: list[str] = field(default_factory=list)

    # Selection info
    selection_method: SelectionMethod = SelectionMethod.FIXED

    # Reconstruction error
    reconstruction_error: float = 0.0


@dataclass
class FactorInterpretation:
    """Interpretation of a PCA factor"""

    factor_id: int
    eigenvalue: float
    variance_explained: float
    cumulative_variance: float

    # Top positive and negative loadings
    top_positive_loadings: list[tuple[str, float]]
    top_negative_loadings: list[tuple[str, float]]

    # Interpretation label (can be set manually or inferred)
    label: str = ""
    interpretation: str = ""


@dataclass
class RollingPCAResults:
    """Results from rolling PCA analysis"""

    dates: list[datetime]
    variance_explained: list[np.ndarray]  # Per-date variance ratios
    first_component_loadings: pd.DataFrame  # Time series of PC1 loadings
    eigenvalue_ratios: list[float]  # Ratio of first to second eigenvalue

    # Stability metrics
    loading_stability: float = 0.0  # How stable loadings are over time
    structure_stability: float = 0.0


# ============================================================================
# PCA ANALYZER CLASS
# ============================================================================


class PCAAnalyzer:
    """
    Principal Component Analysis for financial factor analysis

    This class provides comprehensive PCA functionality including:
    - Automatic component selection
    - Factor interpretation
    - Rolling window analysis
    - Reconstruction and projection

    Example:
        ```python
        analyzer = PCAAnalyzer()

        # Fit PCA on returns matrix
        results = await analyzer.fit(
            data=returns_df,
            n_components=5,
            scaling=ScalingMethod.STANDARD
        )

        # Get factor interpretations
        interpretations = analyzer.interpret_factors(results, top_n=5)

        # Project new data
        scores = analyzer.transform(new_data)
        ```
    """

    def __init__(
        self,
        random_state: int = 42,
    ):
        """
        Initialize PCA analyzer

        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state

        # Fitted model storage
        self._pca: PCA | None = None
        self._scaler: StandardScaler | None = None
        self._feature_names: list[str] = []
        self._is_fitted = False

    @property
    def is_fitted(self) -> bool:
        """Check if model is fitted"""
        return self._is_fitted

    # ========================================================================
    # FITTING
    # ========================================================================

    async def fit(
        self,
        data: pd.DataFrame | np.ndarray,
        n_components: int | float | None = None,
        scaling: ScalingMethod = ScalingMethod.STANDARD,
        selection_method: SelectionMethod = SelectionMethod.FIXED,
        variance_threshold: float = 0.95,
    ) -> PCAResults:
        """
        Fit PCA model on data

        Args:
            data: Input data matrix (samples x features)
            n_components: Number of components to keep
                - int: Fixed number
                - float between 0-1: Proportion of variance to explain
                - None: Use selection_method
            scaling: Scaling method to apply
            selection_method: Method for automatic component selection
            variance_threshold: Variance threshold for automatic selection

        Returns:
            PCAResults object with analysis results
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self._fit_sync(
                data, n_components, scaling, selection_method, variance_threshold
            ),
        )

    def _fit_sync(
        self,
        data: pd.DataFrame | np.ndarray,
        n_components: int | float | None,
        scaling: ScalingMethod,
        selection_method: SelectionMethod,
        variance_threshold: float,
    ) -> PCAResults:
        """
        Fit PCA synchronously
        """
        # Extract feature names
        if isinstance(data, pd.DataFrame):
            self._feature_names = data.columns.tolist()
            X = data.values.astype(np.float64)
        else:
            self._feature_names = [f"feature_{i}" for i in range(data.shape[1])]
            X = data.astype(np.float64)

        # Handle missing values
        X = np.nan_to_num(X, nan=0.0)

        # Scale data
        if scaling == ScalingMethod.STANDARD:
            self._scaler = StandardScaler()
            X_scaled = self._scaler.fit_transform(X)
        elif scaling == ScalingMethod.DEMEAN:
            X_scaled = X - X.mean(axis=0)
        else:
            X_scaled = X

        # Determine number of components
        if n_components is None:
            n_components = self._select_components(X_scaled, selection_method, variance_threshold)
        elif isinstance(n_components, float) and n_components < 1:
            # Interpret as variance ratio
            n_components = self._select_by_variance(X_scaled, n_components)

        n_components = min(n_components, X_scaled.shape[1], X_scaled.shape[0])

        logger.info(f"Fitting PCA with {n_components} components on {X_scaled.shape} data")

        # Fit PCA
        self._pca = PCA(
            n_components=n_components,
            random_state=self.random_state,
        )

        scores = self._pca.fit_transform(X_scaled)

        self._is_fitted = True

        # Build results
        results = self._build_results(X_scaled, scores, selection_method)

        return results

    def _select_components(
        self,
        X: np.ndarray,
        method: SelectionMethod,
        variance_threshold: float,
    ) -> int:
        """
        Automatically select number of components
        """
        # Fit full PCA to get eigenvalues
        full_pca = PCA(random_state=self.random_state)
        full_pca.fit(X)

        eigenvalues = full_pca.explained_variance_
        variance_ratios = full_pca.explained_variance_ratio_

        if method == SelectionMethod.VARIANCE:
            return self._select_by_variance(X, variance_threshold, variance_ratios)

        elif method == SelectionMethod.KAISER:
            # Kaiser criterion: eigenvalue > 1 (for standardized data)
            n = np.sum(eigenvalues > 1)
            return max(1, n)

        elif method == SelectionMethod.SCREE:
            return self._select_by_scree(eigenvalues)

        elif method == SelectionMethod.PARALLEL:
            return self._select_by_parallel_analysis(X, eigenvalues)

        else:
            # Default: explain 95% variance
            return self._select_by_variance(X, 0.95, variance_ratios)

    def _select_by_variance(
        self,
        X: np.ndarray,
        threshold: float,
        variance_ratios: np.ndarray | None = None,
    ) -> int:
        """
        Select components to explain given variance proportion
        """
        if variance_ratios is None:
            full_pca = PCA(random_state=self.random_state)
            full_pca.fit(X)
            variance_ratios = full_pca.explained_variance_ratio_

        cumsum = np.cumsum(variance_ratios)
        n = np.argmax(cumsum >= threshold) + 1
        return max(1, n)

    def _select_by_scree(self, eigenvalues: np.ndarray) -> int:
        """
        Select components using scree plot elbow method

        Uses second derivative to find elbow point
        """
        if len(eigenvalues) < 3:
            return 1

        # Calculate second differences
        first_diff = np.diff(eigenvalues)
        second_diff = np.diff(first_diff)

        # Find point of maximum curvature (elbow)
        # This is where second derivative changes most
        elbow = np.argmax(np.abs(second_diff)) + 1

        return max(1, elbow)

    def _select_by_parallel_analysis(
        self,
        X: np.ndarray,
        real_eigenvalues: np.ndarray,
        n_iterations: int = 100,
    ) -> int:
        """
        Select components using parallel analysis

        Compares real eigenvalues to eigenvalues from random data
        """
        n_samples, n_features = X.shape

        # Generate random eigenvalues
        random_eigenvalues = np.zeros((n_iterations, n_features))

        for i in range(n_iterations):
            random_data = np.random.randn(n_samples, n_features)
            pca = PCA()
            pca.fit(random_data)
            random_eigenvalues[i] = pca.explained_variance_

        # 95th percentile of random eigenvalues
        threshold = np.percentile(random_eigenvalues, 95, axis=0)

        # Count components where real > random
        n = np.sum(real_eigenvalues > threshold)

        return max(1, n)

    def _build_results(
        self,
        X: np.ndarray,
        scores: np.ndarray,
        selection_method: SelectionMethod,
    ) -> PCAResults:
        """
        Build PCAResults object from fitted model
        """
        # Create loadings DataFrame
        loadings_df = pd.DataFrame(
            self._pca.components_.T,
            index=self._feature_names,
            columns=[f"PC{i + 1}" for i in range(self._pca.n_components_)],
        )

        # Create scores DataFrame
        if isinstance(scores, np.ndarray):
            scores_df = pd.DataFrame(
                scores,
                columns=[f"PC{i + 1}" for i in range(self._pca.n_components_)],
            )
        else:
            scores_df = None

        # Calculate reconstruction error
        reconstructed = self._pca.inverse_transform(scores)
        reconstruction_error = np.mean((X - reconstructed) ** 2)

        return PCAResults(
            n_components=self._pca.n_components_,
            n_features=X.shape[1],
            n_samples=X.shape[0],
            explained_variance=self._pca.explained_variance_,
            explained_variance_ratio=self._pca.explained_variance_ratio_,
            cumulative_variance_ratio=np.cumsum(self._pca.explained_variance_ratio_),
            components=self._pca.components_,
            loadings=loadings_df,
            eigenvalues=self._pca.explained_variance_,
            scores=scores,
            scores_df=scores_df,
            feature_names=self._feature_names,
            selection_method=selection_method,
            reconstruction_error=reconstruction_error,
        )

    # ========================================================================
    # TRANSFORMATION
    # ========================================================================

    def transform(
        self,
        data: pd.DataFrame | np.ndarray,
    ) -> np.ndarray:
        """
        Transform data using fitted PCA

        Args:
            data: Input data (samples x features)

        Returns:
            Factor scores (samples x n_components)
        """
        if not self._is_fitted:
            raise ValueError("PCA not fitted. Call fit() first.")

        # Convert to numpy
        if isinstance(data, pd.DataFrame):
            X = data.values.astype(np.float64)
        else:
            X = data.astype(np.float64)

        # Handle NaN
        X = np.nan_to_num(X, nan=0.0)

        # Scale
        if self._scaler is not None:
            X = self._scaler.transform(X)

        # Transform
        return self._pca.transform(X)

    def inverse_transform(
        self,
        scores: np.ndarray,
    ) -> np.ndarray:
        """
        Reconstruct data from factor scores

        Args:
            scores: Factor scores (samples x n_components)

        Returns:
            Reconstructed data (samples x features)
        """
        if not self._is_fitted:
            raise ValueError("PCA not fitted. Call fit() first.")

        # Inverse transform
        X_reconstructed = self._pca.inverse_transform(scores)

        # Inverse scale
        if self._scaler is not None:
            X_reconstructed = self._scaler.inverse_transform(X_reconstructed)

        return X_reconstructed

    # ========================================================================
    # INTERPRETATION
    # ========================================================================

    def interpret_factors(
        self,
        results: PCAResults,
        top_n: int = 5,
    ) -> list[FactorInterpretation]:
        """
        Generate interpretations for each factor

        Args:
            results: PCA results to interpret
            top_n: Number of top loadings to show

        Returns:
            List of FactorInterpretation objects
        """
        interpretations = []

        cumsum = 0.0

        for i in range(results.n_components):
            loadings = results.loadings.iloc[:, i]

            # Sort by absolute value
            abs_loadings = loadings.abs().sort_values(ascending=False)

            # Get top positive loadings
            positive_loadings = loadings[loadings > 0].sort_values(ascending=False)
            top_positive = [(name, loadings[name]) for name in positive_loadings.head(top_n).index]

            # Get top negative loadings
            negative_loadings = loadings[loadings < 0].sort_values()
            top_negative = [(name, loadings[name]) for name in negative_loadings.head(top_n).index]

            cumsum += results.explained_variance_ratio[i]

            interpretation = FactorInterpretation(
                factor_id=i + 1,
                eigenvalue=results.eigenvalues[i],
                variance_explained=results.explained_variance_ratio[i],
                cumulative_variance=cumsum,
                top_positive_loadings=top_positive,
                top_negative_loadings=top_negative,
            )

            # Auto-generate interpretation based on loadings
            interpretation.interpretation = self._generate_interpretation(
                top_positive, top_negative
            )

            interpretations.append(interpretation)

        return interpretations

    def _generate_interpretation(
        self,
        positive: list[tuple[str, float]],
        negative: list[tuple[str, float]],
    ) -> str:
        """
        Generate text interpretation of factor
        """
        if not positive and not negative:
            return "No clear interpretation"

        parts = []

        if positive:
            pos_names = [name for name, _ in positive[:3]]
            parts.append(f"Positively loaded on: {', '.join(pos_names)}")

        if negative:
            neg_names = [name for name, _ in negative[:3]]
            parts.append(f"Negatively loaded on: {', '.join(neg_names)}")

        return ". ".join(parts)

    # ========================================================================
    # ROLLING PCA
    # ========================================================================

    async def rolling_pca(
        self,
        data: pd.DataFrame,
        window: int = 60,
        n_components: int = 3,
        step: int = 1,
        scaling: ScalingMethod = ScalingMethod.STANDARD,
    ) -> RollingPCAResults:
        """
        Perform rolling window PCA analysis

        Useful for tracking how factor structure changes over time.

        Args:
            data: Input data with datetime index
            window: Rolling window size
            n_components: Number of components to extract
            step: Step size between windows
            scaling: Scaling method

        Returns:
            RollingPCAResults with time-varying analysis
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, lambda: self._rolling_pca_sync(data, window, n_components, step, scaling)
        )

    def _rolling_pca_sync(
        self,
        data: pd.DataFrame,
        window: int,
        n_components: int,
        step: int,
        scaling: ScalingMethod,
    ) -> RollingPCAResults:
        """
        Rolling PCA synchronously
        """
        dates = []
        variance_explained_list = []
        pc1_loadings_list = []
        eigenvalue_ratios = []

        feature_names = data.columns.tolist()

        for i in range(window, len(data) + 1, step):
            window_data = data.iloc[i - window : i]

            # Scale
            if scaling == ScalingMethod.STANDARD:
                scaler = StandardScaler()
                X = scaler.fit_transform(window_data.values)
            else:
                X = window_data.values - window_data.values.mean(axis=0)

            # Fit PCA
            try:
                pca = PCA(n_components=n_components, random_state=self.random_state)
                pca.fit(X)

                dates.append(data.index[i - 1])
                variance_explained_list.append(pca.explained_variance_ratio_)
                pc1_loadings_list.append(pca.components_[0])

                # Eigenvalue ratio (measure of dominant factor strength)
                if len(pca.explained_variance_) > 1:
                    eigenvalue_ratios.append(
                        pca.explained_variance_[0] / pca.explained_variance_[1]
                    )
                else:
                    eigenvalue_ratios.append(float("inf"))

            except Exception as e:
                logger.warning(f"Rolling PCA failed at index {i}: {e}")
                continue

        # Create PC1 loadings DataFrame
        pc1_loadings_df = pd.DataFrame(
            pc1_loadings_list,
            index=dates,
            columns=feature_names,
        )

        # Calculate loading stability (correlation of loadings over time)
        if len(pc1_loadings_list) > 1:
            loading_correlations = []
            for i in range(1, len(pc1_loadings_list)):
                corr = np.corrcoef(pc1_loadings_list[i - 1], pc1_loadings_list[i])[0, 1]
                loading_correlations.append(abs(corr))
            loading_stability = np.mean(loading_correlations)
        else:
            loading_stability = 1.0

        return RollingPCAResults(
            dates=dates,
            variance_explained=variance_explained_list,
            first_component_loadings=pc1_loadings_df,
            eigenvalue_ratios=eigenvalue_ratios,
            loading_stability=loading_stability,
        )

    # ========================================================================
    # INCREMENTAL PCA
    # ========================================================================

    def fit_incremental(
        self,
        data: pd.DataFrame | np.ndarray,
        n_components: int,
        batch_size: int = 100,
        scaling: ScalingMethod = ScalingMethod.STANDARD,
    ) -> PCAResults:
        """
        Fit PCA incrementally for large datasets

        Uses mini-batches to fit PCA without loading entire dataset into memory.

        Args:
            data: Input data (can be large)
            n_components: Number of components
            batch_size: Batch size for incremental fitting
            scaling: Scaling method

        Returns:
            PCAResults object
        """
        # Extract feature names
        if isinstance(data, pd.DataFrame):
            self._feature_names = data.columns.tolist()
            X = data.values.astype(np.float64)
        else:
            self._feature_names = [f"feature_{i}" for i in range(data.shape[1])]
            X = data.astype(np.float64)

        # Handle NaN
        X = np.nan_to_num(X, nan=0.0)

        # Scale
        if scaling == ScalingMethod.STANDARD:
            self._scaler = StandardScaler()
            X = self._scaler.fit_transform(X)
        elif scaling == ScalingMethod.DEMEAN:
            X = X - X.mean(axis=0)

        # Fit incrementally
        ipca = IncrementalPCA(n_components=n_components, batch_size=batch_size)
        scores = ipca.fit_transform(X)

        # Store as regular PCA for compatibility
        self._pca = PCA(n_components=n_components)
        self._pca.components_ = ipca.components_
        self._pca.explained_variance_ = ipca.explained_variance_
        self._pca.explained_variance_ratio_ = ipca.explained_variance_ratio_
        self._pca.mean_ = ipca.mean_
        self._pca.n_components_ = n_components

        self._is_fitted = True

        return self._build_results(X, scores, SelectionMethod.FIXED)

    # ========================================================================
    # ANALYSIS UTILITIES
    # ========================================================================

    def biplot_data(
        self,
        results: PCAResults,
        pc_x: int = 1,
        pc_y: int = 2,
    ) -> dict[str, Any]:
        """
        Get data for creating a biplot

        Args:
            results: PCA results
            pc_x: Component for x-axis (1-indexed)
            pc_y: Component for y-axis (1-indexed)

        Returns:
            Dictionary with scores and loadings for plotting
        """
        # Convert to 0-indexed
        idx_x = pc_x - 1
        idx_y = pc_y - 1

        if idx_x >= results.n_components or idx_y >= results.n_components:
            raise ValueError(f"Component indices out of range (max: {results.n_components})")

        return {
            "scores_x": results.scores[:, idx_x].tolist() if results.scores is not None else [],
            "scores_y": results.scores[:, idx_y].tolist() if results.scores is not None else [],
            "loadings_x": results.components[idx_x].tolist(),
            "loadings_y": results.components[idx_y].tolist(),
            "feature_names": results.feature_names,
            "variance_x": results.explained_variance_ratio[idx_x],
            "variance_y": results.explained_variance_ratio[idx_y],
        }

    def correlation_with_factors(
        self,
        results: PCAResults,
        original_data: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Calculate correlation between original features and PC scores

        Args:
            results: PCA results
            original_data: Original data used for PCA

        Returns:
            Correlation matrix (features x components)
        """
        if results.scores is None:
            raise ValueError("No scores available in results")

        correlations = np.zeros((len(results.feature_names), results.n_components))

        for i, feature in enumerate(results.feature_names):
            for j in range(results.n_components):
                corr = np.corrcoef(original_data[feature].values, results.scores[:, j])[0, 1]
                correlations[i, j] = corr

        return pd.DataFrame(
            correlations,
            index=results.feature_names,
            columns=[f"PC{i + 1}" for i in range(results.n_components)],
        )


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================


async def fit_pca(
    data: pd.DataFrame,
    n_components: int | None = None,
    variance_threshold: float = 0.95,
) -> PCAResults:
    """
    Quick function to fit PCA on data

    Args:
        data: Input data
        n_components: Number of components (None for auto)
        variance_threshold: Variance to explain if n_components is None

    Returns:
        PCAResults object
    """
    analyzer = PCAAnalyzer()

    if n_components is None:
        return await analyzer.fit(
            data=data,
            scaling=ScalingMethod.STANDARD,
            selection_method=SelectionMethod.VARIANCE,
            variance_threshold=variance_threshold,
        )
    else:
        return await analyzer.fit(
            data=data,
            n_components=n_components,
            scaling=ScalingMethod.STANDARD,
        )


def get_factor_scores(
    data: pd.DataFrame,
    n_components: int = 5,
) -> pd.DataFrame:
    """
    Quick function to get factor scores from data

    Args:
        data: Input data
        n_components: Number of components

    Returns:
        DataFrame with factor scores
    """
    analyzer = PCAAnalyzer()

    # Fit synchronously for simple usage
    scaler = StandardScaler()
    X = scaler.fit_transform(data.values)

    pca = PCA(n_components=n_components)
    scores = pca.fit_transform(X)

    return pd.DataFrame(
        scores,
        index=data.index,
        columns=[f"PC{i + 1}" for i in range(n_components)],
    )


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================


if __name__ == "__main__":
    """Example usage of PCA analyzer"""

    async def main():
        # Create sample data (simulated returns)
        np.random.seed(42)
        n_samples = 500
        n_assets = 20

        # Simulate correlated returns
        factor_returns = np.random.randn(n_samples, 3)  # 3 latent factors
        loadings = np.random.randn(3, n_assets)
        idiosyncratic = np.random.randn(n_samples, n_assets) * 0.3

        returns = factor_returns @ loadings + idiosyncratic

        dates = pd.date_range("2020-01-01", periods=n_samples, freq="D")
        asset_names = [f"Asset_{i}" for i in range(n_assets)]

        returns_df = pd.DataFrame(returns, index=dates, columns=asset_names)

        # Initialize analyzer
        analyzer = PCAAnalyzer()

        # Example 1: Fit PCA with automatic component selection
        print("\n=== PCA Analysis ===")
        results = await analyzer.fit(
            data=returns_df,
            scaling=ScalingMethod.STANDARD,
            selection_method=SelectionMethod.KAISER,
        )

        print(f"Components selected: {results.n_components}")
        print(f"Variance explained: {results.explained_variance_ratio[:5].round(3)}")
        print(f"Cumulative variance: {results.cumulative_variance_ratio[:5].round(3)}")

        # Example 2: Interpret factors
        print("\n=== Factor Interpretations ===")
        interpretations = analyzer.interpret_factors(results, top_n=3)

        for interp in interpretations[:3]:
            print(f"\nPC{interp.factor_id}:")
            print(f"  Variance explained: {interp.variance_explained:.1%}")
            print(f"  Top positive: {interp.top_positive_loadings[:2]}")
            print(f"  Top negative: {interp.top_negative_loadings[:2]}")

        # Example 3: Rolling PCA
        print("\n=== Rolling PCA ===")
        rolling_results = await analyzer.rolling_pca(
            data=returns_df,
            window=60,
            n_components=3,
            step=20,
        )

        print(f"Windows analyzed: {len(rolling_results.dates)}")
        print(f"Loading stability: {rolling_results.loading_stability:.3f}")
        print(f"Eigenvalue ratio (PC1/PC2): {np.mean(rolling_results.eigenvalue_ratios):.2f}")

    asyncio.run(main())
