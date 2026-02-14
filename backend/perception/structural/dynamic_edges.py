# backend/perception/structural/dynamic_edges.py
"""
Dynamic Edge Computer

Computes dynamic edges for the market graph based on rolling correlations.

Edges represent relationships that change over time:
- Rolling 30-day price correlation (updated daily)
- High correlation → Strong edge weight
- Correlation breakdown → Weak or negative edge weight

Mathematical Formulation:
    ρ(X,Y) = Cov(X,Y) / (σ_X * σ_Y)

    Where:
    - ρ ∈ [-1, 1] is the correlation coefficient
    - Cov(X,Y) is the covariance
    - σ_X, σ_Y are standard deviations

Edge weights are typically:
    w_ij = |ρ_ij| for undirected graphs
    or w_ij = max(0, ρ_ij) to ignore negative correlations

Usage:
    >>> computer = DynamicEdgeComputer(window=30)
    >>> edges = computer.compute_edges(price_data)
    >>> # edges is adjacency matrix with correlation weights
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from loguru import logger


@dataclass
class CorrelationConfig:
    """
    Configuration for correlation-based edge computation.

    Attributes:
        window: Rolling window size (e.g., 30 days)
        min_periods: Minimum periods required for correlation
        method: Correlation method ('pearson', 'spearman', 'kendall')
        threshold: Minimum correlation to create edge
        use_absolute: Use absolute correlation values
    """

    window: int = 30
    min_periods: int = 20
    method: str = "pearson"
    threshold: float = 0.1
    use_absolute: bool = True


class DynamicEdgeComputer:
    """
    Computes dynamic edges based on rolling price correlations.

    This creates time-varying graph structures where edges represent
    the current correlation strength between assets.

    Example:
        >>> computer = DynamicEdgeComputer(config=CorrelationConfig(window=30))
        >>> price_df = pd.DataFrame({
        >>>     'AAPL': aapl_prices,
        >>>     'MSFT': msft_prices,
        >>>     'NVDA': nvda_prices
        >>> })
        >>> edges, weights = computer.compute_edges(price_df)
    """

    def __init__(self, config: CorrelationConfig | None = None):
        """
        Initialize dynamic edge computer.

        Args:
            config: Correlation configuration
        """
        self.config = config or CorrelationConfig()
        logger.debug(
            f"DynamicEdgeComputer initialized: window={self.config.window}, "
            f"method={self.config.method}"
        )

    def compute_edges(
        self, prices: pd.DataFrame, timestamp: pd.Timestamp | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute dynamic edges from price data.

        Args:
            prices: DataFrame with asset prices [timestamp, asset_name]
            timestamp: Specific timestamp to compute edges for (latest if None)

        Returns:
            edge_index: Edge list [2, num_edges] (source, target pairs)
            edge_weights: Edge weights [num_edges]
        """
        # Use latest timestamp if not specified
        if timestamp is None:
            timestamp = prices.index[-1]

        # Get rolling window
        end_idx = prices.index.get_loc(timestamp)
        start_idx = max(0, end_idx - self.config.window + 1)

        window_data = prices.iloc[start_idx : end_idx + 1]

        # Compute correlation matrix
        corr_matrix = self._compute_correlation_matrix(window_data)

        # Extract edges and weights
        edge_index, edge_weights = self._matrix_to_edges(corr_matrix)

        return edge_index, edge_weights

    def compute_rolling_edges(
        self, prices: pd.DataFrame, return_dataframe: bool = False
    ) -> dict[pd.Timestamp, tuple[np.ndarray, np.ndarray]]:
        """
        Compute edges for all timestamps in rolling fashion.

        Args:
            prices: DataFrame with asset prices
            return_dataframe: If True, return as DataFrame

        Returns:
            Dictionary mapping timestamps to (edge_index, edge_weights)
        """
        edges_dict = {}

        # Skip initial period without enough data
        start_idx = self.config.window - 1

        for i in range(start_idx, len(prices)):
            timestamp = prices.index[i]
            edge_index, edge_weights = self.compute_edges(prices, timestamp)
            edges_dict[timestamp] = (edge_index, edge_weights)

        logger.info(f"Computed rolling edges for {len(edges_dict)} timestamps")

        return edges_dict

    def _compute_correlation_matrix(self, data: pd.DataFrame) -> np.ndarray:
        """
        Compute correlation matrix from price data.

        Args:
            data: Price data

        Returns:
            Correlation matrix [n_assets, n_assets]
        """
        if len(data) < self.config.min_periods:
            logger.warning(f"Insufficient data: {len(data)} < {self.config.min_periods}")
            # Return identity matrix (no correlations)
            return np.eye(len(data.columns))

        # Compute returns
        returns = data.pct_change().dropna()

        if len(returns) == 0:
            return np.eye(len(data.columns))

        # Compute correlation
        if self.config.method == "pearson":
            corr_matrix = returns.corr(method="pearson").values
        elif self.config.method == "spearman":
            corr_matrix = returns.corr(method="spearman").values
        elif self.config.method == "kendall":
            corr_matrix = returns.corr(method="kendall").values
        else:
            raise ValueError(f"Unknown correlation method: {self.config.method}")

        # Handle NaN values
        corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)

        # Use absolute values if configured
        if self.config.use_absolute:
            corr_matrix = np.abs(corr_matrix)

        return corr_matrix

    def _matrix_to_edges(self, corr_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Convert correlation matrix to edge list.

        Args:
            corr_matrix: Correlation matrix [n_assets, n_assets]

        Returns:
            edge_index: [2, num_edges]
            edge_weights: [num_edges]
        """
        n_assets = corr_matrix.shape[0]

        # Get upper triangle (avoid duplicates and self-loops)
        edge_list = []
        weight_list = []

        for i in range(n_assets):
            for j in range(i + 1, n_assets):
                corr = corr_matrix[i, j]

                # Only create edge if above threshold
                if abs(corr) >= self.config.threshold:
                    # Add both directions for undirected graph
                    edge_list.append([i, j])
                    edge_list.append([j, i])
                    weight_list.append(corr)
                    weight_list.append(corr)

        if len(edge_list) == 0:
            # No edges - return empty arrays
            return np.array([[], []], dtype=np.int64), np.array([])

        edge_index = np.array(edge_list, dtype=np.int64).T
        edge_weights = np.array(weight_list, dtype=np.float32)

        return edge_index, edge_weights

    def get_adjacency_matrix(
        self, prices: pd.DataFrame, timestamp: pd.Timestamp | None = None
    ) -> np.ndarray:
        """
        Get adjacency matrix representation of edges.

        Args:
            prices: Price data
            timestamp: Timestamp for edges

        Returns:
            Adjacency matrix [n_assets, n_assets]
        """
        edge_index, edge_weights = self.compute_edges(prices, timestamp)

        n_assets = len(prices.columns)
        adj_matrix = np.zeros((n_assets, n_assets))

        if edge_index.shape[1] > 0:
            for k in range(edge_index.shape[1]):
                i, j = edge_index[:, k]
                adj_matrix[i, j] = edge_weights[k]

        return adj_matrix

    def detect_correlation_breakdown(
        self,
        prices: pd.DataFrame,
        historical_window: int = 90,
        current_window: int = 30,
        threshold_change: float = 0.5,
    ) -> list[tuple[str, str, float]]:
        """
        Detect correlation breakdowns (pairs that were correlated, now aren't).

        This is useful for adversarial scenario detection.

        Args:
            prices: Price data
            historical_window: Window for historical correlation
            current_window: Window for current correlation
            threshold_change: Minimum correlation change to flag

        Returns:
            List of (asset1, asset2, correlation_change) tuples
        """
        if len(prices) < historical_window:
            return []

        # Historical correlation
        hist_data = prices.iloc[-historical_window:-current_window]
        hist_corr = self._compute_correlation_matrix(hist_data)

        # Current correlation
        curr_data = prices.iloc[-current_window:]
        curr_corr = self._compute_correlation_matrix(curr_data)

        # Find breakdowns
        breakdowns = []
        asset_names = prices.columns.tolist()
        n_assets = len(asset_names)

        for i in range(n_assets):
            for j in range(i + 1, n_assets):
                hist = hist_corr[i, j]
                curr = curr_corr[i, j]
                change = abs(curr - hist)

                if change >= threshold_change:
                    breakdowns.append((asset_names[i], asset_names[j], change))

        # Sort by magnitude of change
        breakdowns.sort(key=lambda x: x[2], reverse=True)

        if breakdowns:
            logger.warning(f"Detected {len(breakdowns)} correlation breakdowns")

        return breakdowns


def compute_rolling_correlation(
    series1: pd.Series, series2: pd.Series, window: int = 30, min_periods: int = 20
) -> pd.Series:
    """
    Compute rolling correlation between two time series.

    Utility function for simple pairwise correlation.

    Args:
        series1: First time series
        series2: Second time series
        window: Rolling window size
        min_periods: Minimum periods required

    Returns:
        Rolling correlation series
    """
    # Align series
    aligned = pd.DataFrame({"series1": series1, "series2": series2}).dropna()

    # Compute rolling correlation
    rolling_corr = (
        aligned["series1"].rolling(window=window, min_periods=min_periods).corr(aligned["series2"])
    )

    return rolling_corr


def create_sector_edges(
    assets: list[str], sector_mapping: dict[str, str], weight: float = 1.0
) -> tuple[np.ndarray, np.ndarray]:
    """
    Create static edges based on sector membership.

    Assets in the same sector are connected.

    Args:
        assets: List of asset symbols
        sector_mapping: Dictionary mapping asset to sector
        weight: Edge weight for same-sector connections

    Returns:
        edge_index: [2, num_edges]
        edge_weights: [num_edges]
    """
    n_assets = len(assets)
    asset_to_idx = {asset: i for i, asset in enumerate(assets)}

    edges = []
    weights = []

    for i, asset_i in enumerate(assets):
        sector_i = sector_mapping.get(asset_i)
        if sector_i is None:
            continue

        for j, asset_j in enumerate(assets):
            if i >= j:  # Skip self and duplicates
                continue

            sector_j = sector_mapping.get(asset_j)
            if sector_j is None:
                continue

            # Connect if same sector
            if sector_i == sector_j:
                edges.append([i, j])
                edges.append([j, i])
                weights.append(weight)
                weights.append(weight)

    if len(edges) == 0:
        return np.array([[], []], dtype=np.int64), np.array([])

    edge_index = np.array(edges, dtype=np.int64).T
    edge_weights = np.array(weights, dtype=np.float32)

    return edge_index, edge_weights
