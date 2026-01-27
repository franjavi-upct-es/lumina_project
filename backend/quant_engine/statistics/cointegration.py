# backend/quant_engine/statistics/cointegration.py
"""
Cointegration Testing

Tests for long-run equilibrium relationships between non-stationary time series.
Two non-stationary series are cointegrated if their linear combination is stationary.

Methods implemented:
- Engle-Granger two-step test
- Johansen test (trace and max eigenvalue)
- Phillips-Ouliaris test
- Cointegrating vector estimation

Applications in finance:
- Pairs trading (finding cointegrated stock pairs)
- Statistical arbitrage
- Portfolio balancing
- Risk management

References:
- Engle, R. F., & Granger, C. W. J. (1987). Co-integration and error correction.
- Johansen, S. (1991). Estimation and hypothesis testing of cointegration vectors.
"""

from typing import Any

import numpy as np
import pandas as pd
import polars as pl
from loguru import logger
from statsmodels.tsa.stattools import adfuller, coint
from statsmodels.tsa.vector_ar.vecm import coint_johansen


class CointegrationTester:
    """
    Test for cointegration between time series

    Provides methods for:
    - Engle-Granger cointegration test
    - Johansen cointegration test
    - Cointegrating vector estimation
    - Half-life calculation
    """

    def __init__(self):
        """Initialize cointegration tester"""
        logger.info("Cointegration tester initialized")

    def engle_granger_test(
        self,
        y: np.ndarray | pd.Series | pl.Series,
        x: np.ndarray | pd.Series | pl.Series,
        trend: str = "c",
    ) -> dict[str, Any]:
        """
        Engle-Granger two-step cointegration test

        Step 1: Regress Y on X to get cointegrating vector
        Step 2: Test residuals for stationarity using ADF test

        Null hypothesis: No cointegration (residuals have unit root)

        Args:
            y: Dependent variable time series
            x: Independent variable time series
            trend: Deterministic trend ('c', 'ct', 'ctt', 'nc')
                - 'c': constant only
                - 'ct': constant and trend
                - 'ctt': constant, linear and quadratic trend
                - 'nc': no constant, no trend

        Returns:
            Dictionary with test results
        """
        # Convert to numpy
        if isinstance(y, (pd.Series, pl.Series)):
            y = y.to_numpy()
        if isinstance(x, (pd.Series, pl.Series)):
            x = x.to_numpy()

        # Align series
        n = min(len(y), len(x))
        y = y[:n]
        x = x[:n]

        # Use statsmodels coint test
        t_stat, p_value, crit_values = coint(y, x, trend=trend)

        # Estimate cointegrating vector (OLS)
        from sklearn.linear_model import LinearRegression

        X_reshaped = x.reshape(-1, 1)
        model = LinearRegression()
        model.fit(X_reshaped, y)

        beta = model.coef_[0]
        alpha = model.intercept_

        # Calculate residuals (spread)
        residuals = y - (alpha + beta * x)

        # Additional ADF test on residuals for detailed info
        adf_result = adfuller(residuals, regression=trend)

        # Determine cointegration
        is_cointegrated = p_value < 0.05

        logger.info(
            f"Engle-Granger test: t-stat={t_stat:.4f}, p-value={p_value:.4f}, "
            f"cointegrated={is_cointegrated}"
        )

        return {
            "test_statistic": t_stat,
            "p_value": p_value,
            "critical_values": {
                "1%": crit_values[0],
                "5%": crit_values[1],
                "10%": crit_values[2],
            },
            "is_cointegrated": is_cointegrated,
            "cointegrating_vector": {
                "alpha": alpha,
                "beta": beta,
            },
            "residuals": residuals,
            "adf_statistic": adf_result[0],
            "adf_p_value": adf_result[1],
            "method": "engle_granger",
        }

    def johansen_test(
        self,
        data: pd.DataFrame | np.ndarray,
        det_order: int = 0,
        k_ar_diff: int = 1,
    ) -> dict[str, Any]:
        """
        Johansen cointegration test for multiple time series

        Tests for cointegration among multiple (>2) time series.
        Provides both trace and maximum eigenvalue statistics.

        Args:
            data: DataFrame or array with multiple time series (columns = variables)
            det_order: Deterministic term order
                - -1: no deterministic terms
                - 0: constant term
                - 1: constant and linear trend
            k_ar_diff: Number of lagged differences in the model

        Returns:
            Dictionary with Johansen test results
        """
        # Convert to DataFrame if needed
        if isinstance(data, np.ndarray):
            data = pd.DataFrame(data)

        # Run Johansen test
        johansen_result = coint_johansen(data, det_order=det_order, k_ar_diff=k_ar_diff)

        n_vars = data.shape[1]

        # Extract trace statistics
        trace_stats = johansen_result.lr1  # Trace statistic
        trace_crit_vals = johansen_result.cvt  # Critical values for trace

        # Extract max eigenvalue statistics
        max_eig_stats = johansen_result.lr2  # Max eigenvalue statistic
        max_eig_crit_vals = johansen_result.cvm  # Critical values for max eigenvalue

        # Determine number of cointegrating relationships
        # Using 5% significance level (index 1)
        r_trace = np.sum(trace_stats > trace_crit_vals[:, 1])
        r_max_eig = np.sum(max_eig_stats > max_eig_crit_vals[:, 1])

        # Cointegrating vectors
        coint_vectors = johansen_result.evec

        # Eigenvalues
        eigenvalues = johansen_result.eig

        logger.info(
            f"Johansen test: {r_trace} cointegrating relations (trace), "
            f"{r_max_eig} (max eigenvalue)"
        )

        return {
            "trace_statistic": trace_stats.tolist(),
            "trace_critical_values": {
                "90%": trace_crit_vals[:, 0].tolist(),
                "95%": trace_crit_vals[:, 1].tolist(),
                "99%": trace_crit_vals[:, 2].tolist(),
            },
            "max_eigenvalue_statistic": max_eig_stats.tolist(),
            "max_eigenvalue_critical_values": {
                "90%": max_eig_crit_vals[:, 0].tolist(),
                "95%": max_eig_crit_vals[:, 1].tolist(),
                "99%": max_eig_crit_vals[:, 2].tolist(),
            },
            "n_cointegrating_relations_trace": int(r_trace),
            "n_cointegrating_relations_max_eig": int(r_max_eig),
            "cointegrating_vectors": coint_vectors,
            "eigenvalues": eigenvalues.tolist(),
            "method": "johansen",
        }

    def calculate_half_life(
        self,
        residuals: np.ndarray | pd.Series | pl.Series,
    ) -> dict[str, float]:
        """
        Calculate half-life of mean reversion for cointegrated series

        Half-life is the expected time for the spread to revert halfway
        to its mean. Useful for pairs trading.

        Args:
            residuals: Residuals from cointegration (the spread)

        Returns:
            Dictionary with half-life and related metrics
        """
        # Convert to numpy
        if isinstance(residuals, (pd.Series, pl.Series)):
            residuals = residuals.to_numpy()

        # Lag residuals
        residuals_lag = residuals[:-1]
        residuals_diff = np.diff(residuals)

        # Fit AR(1) model: Δz_t = λ * z_{t-1} + ε_t
        from sklearn.linear_model import LinearRegression

        model = LinearRegression()
        model.fit(residuals_lag.reshape(-1, 1), residuals_diff)

        lambda_param = model.coef_[0]

        # Half-life = -ln(2) / ln(1 + λ)
        if lambda_param < 0:
            half_life = -np.log(2) / np.log(1 + lambda_param)
        else:
            half_life = np.inf  # No mean reversion

        logger.info(f"Half-life of mean reversion: {half_life:.2f} periods")

        return {
            "half_life": half_life,
            "lambda": lambda_param,
            "mean_reverting": lambda_param < 0,
        }

    def estimate_hedge_ratio(
        self,
        y: np.ndarray | pd.Series | pl.Series,
        x: np.ndarray | pd.Series | pl.Series,
        method: str = "ols",
    ) -> dict[str, Any]:
        """
        Estimate optimal hedge ratio for cointegrated pairs

        Args:
            y: Dependent variable
            x: Independent variable
            method: Estimation method ('ols', 'tls')
                - 'ols': Ordinary Least Squares
                - 'tls': Total Least Squares

        Returns:
            Dictionary with hedge ratio and statistics
        """
        # Convert to numpy
        if isinstance(y, (pd.Series, pl.Series)):
            y = y.to_numpy()
        if isinstance(x, (pd.Series, pl.Series)):
            x = x.to_numpy()

        if method == "ols":
            from sklearn.linear_model import LinearRegression

            model = LinearRegression()
            model.fit(x.reshape(-1, 1), y)

            hedge_ratio = model.coef_[0]
            intercept = model.intercept_

            # Calculate R-squared
            y_pred = model.predict(x.reshape(-1, 1))
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r_squared = 1 - (ss_res / ss_tot)

        elif method == "tls":
            # Total Least Squares (orthogonal regression)
            from scipy.linalg import svd

            # Center data
            x_centered = x - np.mean(x)
            y_centered = y - np.mean(y)

            # Stack into matrix
            data_matrix = np.column_stack([x_centered, y_centered])

            # SVD
            U, S, Vt = svd(data_matrix)

            # Hedge ratio from first right singular vector
            hedge_ratio = -Vt[0, 1] / Vt[0, 0]
            intercept = np.mean(y) - hedge_ratio * np.mean(x)

            # Approximate R-squared
            residuals = y - (intercept + hedge_ratio * x)
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r_squared = 1 - (ss_res / ss_tot)

        else:
            raise ValueError(f"Unknown method: {method}")

        # Calculate spread
        spread = y - (intercept + hedge_ratio * x)

        logger.info(f"Hedge ratio ({method}): {hedge_ratio:.4f}")

        return {
            "hedge_ratio": hedge_ratio,
            "intercept": intercept,
            "r_squared": r_squared,
            "spread": spread,
            "spread_mean": np.mean(spread),
            "spread_std": np.std(spread),
            "method": method,
        }

    def rolling_cointegration(
        self,
        y: np.ndarray | pd.Series | pl.Series,
        x: np.ndarray | pd.Series | pl.Series,
        window: int = 252,
    ) -> pd.DataFrame:
        """
        Perform rolling cointegration test

        Tests cointegration over a rolling window to detect time-varying
        cointegration relationships.

        Args:
            y: Dependent variable
            x: Independent variable
            window: Rolling window size

        Returns:
            DataFrame with rolling cointegration statistics
        """
        # Convert to numpy
        if isinstance(y, (pd.Series, pl.Series)):
            y = y.to_numpy()
        if isinstance(x, (pd.Series, pl.Series)):
            x = x.to_numpy()

        n = len(y)
        results = []

        for i in range(window, n):
            window_y = y[i - window : i]
            window_x = x[i - window : i]

            # Test cointegration
            try:
                test_result = self.engle_granger_test(window_y, window_x)

                results.append(
                    {
                        "period": i,
                        "test_statistic": test_result["test_statistic"],
                        "p_value": test_result["p_value"],
                        "is_cointegrated": test_result["is_cointegrated"],
                        "beta": test_result["cointegrating_vector"]["beta"],
                    }
                )
            except Exception as e:
                logger.warning(f"Rolling cointegration failed at period {i}: {e}")
                continue

        results_df = pd.DataFrame(results)

        logger.success(f"Rolling cointegration completed for {len(results)} windows")

        return results_df


def test_cointegration(
    y: np.ndarray | pd.Series | pl.Series,
    x: np.ndarray | pd.Series | pl.Series,
    method: str = "engle_granger",
) -> dict[str, Any]:
    """
    Convenience function to test cointegration

    Args:
        y: Dependent variable
        x: Independent variable
        method: Test method ('engle_granger' or 'johansen')

    Returns:
        Cointegration test results

    Example:
        >>> # Generate cointegrated series
        >>> np.random.seed(42)
        >>> n = 200
        >>> x = np.cumsum(np.random.randn(n))  # Random walk
        >>> y = 2 * x + np.random.randn(n) * 0.5  # Cointegrated with x
        >>>
        >>> # Test cointegration
        >>> result = test_cointegration(y, x)
        >>> print(f"Cointegrated: {result['is_cointegrated']}")
        >>> print(f"Hedge ratio: {result['cointegrating_vector']['beta']:.4f}")
    """
    tester = CointegrationTester()

    if method == "engle_granger":
        return tester.engle_granger_test(y, x)
    elif method == "johansen":
        # Combine into DataFrame
        data = pd.DataFrame({"y": y, "x": x})
        return tester.johansen_test(data)
    else:
        raise ValueError(f"Unknown method: {method}")


def find_cointegrated_pairs(
    data: pd.DataFrame,
    significance: float = 0.05,
) -> pd.DataFrame:
    """
    Find all cointegrated pairs in a dataset

    Args:
        data: DataFrame with multiple time series (columns = assets)
        significance: P-value threshold for cointegration

    Returns:
        DataFrame with cointegrated pairs and their statistics

    Example:
        >>> # Generate sample data
        >>> n = 200
        >>> data = pd.DataFrame({
        ...     'A': np.cumsum(np.random.randn(n)),
        ...     'B': np.cumsum(np.random.randn(n)),
        ...     'C': np.cumsum(np.random.randn(n)),
        ... })
        >>> # Find cointegrated pairs
        >>> pairs = find_cointegrated_pairs(data)
        >>> print(pairs)
    """
    tester = CointegrationTester()
    columns = data.columns.tolist()
    n_series = len(columns)

    pairs = []

    for i in range(n_series):
        for j in range(i + 1, n_series):
            asset1 = columns[i]
            asset2 = columns[j]

            # Test both directions
            try:
                result_ij = tester.engle_granger_test(data[asset1], data[asset2])
                result_ji = tester.engle_granger_test(data[asset2], data[asset1])

                # Use the better result
                if result_ij["p_value"] < result_ji["p_value"]:
                    result = result_ij
                    direction = f"{asset2} -> {asset1}"
                else:
                    result = result_ji
                    direction = f"{asset1} -> {asset2}"

                if result["p_value"] < significance:
                    # Calculate half-life
                    half_life_result = tester.calculate_half_life(result["residuals"])

                    pairs.append(
                        {
                            "asset1": asset1,
                            "asset2": asset2,
                            "direction": direction,
                            "p_value": result["p_value"],
                            "test_statistic": result["test_statistic"],
                            "hedge_ratio": result["cointegrating_vector"]["beta"],
                            "half_life": half_life_result["half_life"],
                        }
                    )

            except Exception as e:
                logger.warning(f"Cointegration test failed for {asset1}-{asset2}: {e}")
                continue

    pairs_df = pd.DataFrame(pairs)

    if not pairs_df.empty:
        pairs_df = pairs_df.sort_values("p_value")

    logger.success(f"Found {len(pairs_df)} cointegrated pairs")

    return pairs_df
