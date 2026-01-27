# backend/quant_engine/statistics/causality.py
"""
Granger Causality Testing

Tests whether one time series is useful in forecasting another.
X "Granger-causes" Y if past values of X help predict Y better than
past values of Y alone.

Important notes:
- Granger causality is about predictive causality, not true causation
- Requires stationary time series (test/difference first)
- Sensitive to lag selection

Applications in finance:
- Lead-lag relationships between assets
- Information flow between markets
- Market microstructure analysis
- Trading signal development

References:
- Granger, C. W. J. (1969). Investigating causal relations by econometric models.
- Hamilton, J. D. (1994). Time series analysis.
"""

from typing import Any

import numpy as np
import pandas as pd
import polars as pl
from loguru import logger
from statsmodels.tsa.stattools import grangercausalitytests


class CausalityTester:
    """
    Test for Granger causality between time series

    Provides methods for:
    - Granger causality tests
    - Optimal lag selection
    - Bidirectional causality testing
    - Multiple series causality analysis
    """

    def __init__(self, max_lag: int = 10):
        """
        Initialize causality tester

        Args:
            max_lag: Maximum number of lags to test
        """
        self.max_lag = max_lag
        logger.info(f"Causality tester initialized: max_lag={max_lag}")

    def granger_causality_test(
        self,
        y: np.ndarray | pd.Series | pl.Series,
        x: np.ndarray | pd.Series | pl.Series,
        max_lag: int | None = None,
        verbose: bool = False,
    ) -> dict[str, Any]:
        """
        Test if X Granger-causes Y

        Null hypothesis: X does NOT Granger-cause Y

        Args:
            y: Dependent variable time series
            x: Independent variable time series
            max_lag: Maximum lag to test (None = use default)
            verbose: Print detailed results

        Returns:
            Dictionary with test results for each lag
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

        # Create dataframe for statsmodels
        data = pd.DataFrame({"y": y, "x": x})

        max_lag = max_lag or self.max_lag

        # Run Granger causality test
        try:
            gc_results = grangercausalitytests(data[["y", "x"]], max_lag, verbose=verbose)
        except Exception as e:
            logger.error(f"Granger causality test failed: {e}")
            return {"error": str(e)}

        # Extract results
        results = {}

        for lag in range(1, max_lag + 1):
            lag_results = gc_results[lag][0]

            # F-test
            f_test = lag_results["ssr_ftest"]
            f_stat = f_test[0]
            f_pvalue = f_test[1]

            # Chi-square test
            chi_test = lag_results["ssr_chi2test"]
            chi_stat = chi_test[0]
            chi_pvalue = chi_test[1]

            # Likelihood ratio test
            lr_test = lag_results["lrtest"]
            lr_stat = lr_test[0]
            lr_pvalue = lr_test[1]

            # Parameter F-test
            param_ftest = lag_results["params_ftest"]
            param_f_stat = param_ftest[0]
            param_f_pvalue = param_ftest[1]

            results[lag] = {
                "lag": lag,
                "f_stat": f_stat,
                "f_pvalue": f_pvalue,
                "chi2_stat": chi_stat,
                "chi2_pvalue": chi_pvalue,
                "lr_stat": lr_stat,
                "lr_pvalue": lr_pvalue,
                "param_f_stat": param_f_stat,
                "param_f_pvalue": param_f_pvalue,
                "significant_5pct": f_pvalue < 0.05,
                "significant_1pct": f_pvalue < 0.01,
            }

        logger.info(f"Granger causality test completed for lags 1-{max_lag}")

        return results

    def optimal_lag_selection(
        self,
        y: np.ndarray | pd.Series | pl.Series,
        x: np.ndarray | pd.Series | pl.Series,
        criterion: str = "aic",
    ) -> dict[str, Any]:
        """
        Select optimal lag using information criterion

        Args:
            y: Dependent variable
            x: Independent variable
            criterion: 'aic', 'bic', or 'hqic'

        Returns:
            Dictionary with optimal lag and information criteria
        """
        from statsmodels.tsa.api import VAR

        # Convert to numpy
        if isinstance(y, (pd.Series, pl.Series)):
            y = y.to_numpy()
        if isinstance(x, (pd.Series, pl.Series)):
            x = x.to_numpy()

        # Create dataframe
        data = pd.DataFrame({"y": y, "x": x})

        # Fit VAR model
        model = VAR(data)

        # Select lag order
        lag_order_results = model.select_order(maxlags=self.max_lag)

        # Get optimal lag for each criterion
        optimal_lags = {
            "aic": lag_order_results.aic,
            "bic": lag_order_results.bic,
            "hqic": lag_order_results.hqic,
            "fpe": lag_order_results.fpe,
        }

        selected_lag = optimal_lags[criterion]

        logger.info(f"Optimal lag selected: {selected_lag} (criterion: {criterion})")

        return {
            "optimal_lag": selected_lag,
            "criterion": criterion,
            "all_criteria": optimal_lags,
        }

    def bidirectional_causality(
        self,
        series1: np.ndarray | pd.Series | pl.Series,
        series2: np.ndarray | pd.Series | pl.Series,
        max_lag: int | None = None,
        alpha: float = 0.05,
    ) -> dict[str, Any]:
        """
        Test for bidirectional Granger causality

        Tests both:
        - Does series1 -> series2?
        - Does series2 -> series1?

        Args:
            series1: First time series
            series2: Second time series
            max_lag: Maximum lag to test
            alpha: Significance level

        Returns:
            Dictionary with bidirectional causality results
        """
        max_lag = max_lag or self.max_lag

        # Test series1 -> series2
        results_1_to_2 = self.granger_causality_test(series2, series1, max_lag)

        # Test series2 -> series1
        results_2_to_1 = self.granger_causality_test(series1, series2, max_lag)

        # Determine causality at each lag
        causality_summary = []

        for lag in range(1, max_lag + 1):
            if "error" in results_1_to_2 or "error" in results_2_to_1:
                continue

            s1_causes_s2 = results_1_to_2[lag]["f_pvalue"] < alpha
            s2_causes_s1 = results_2_to_1[lag]["f_pvalue"] < alpha

            if s1_causes_s2 and s2_causes_s1:
                relationship = "bidirectional"
            elif s1_causes_s2:
                relationship = "series1 -> series2"
            elif s2_causes_s1:
                relationship = "series2 -> series1"
            else:
                relationship = "no causality"

            causality_summary.append(
                {
                    "lag": lag,
                    "relationship": relationship,
                    "s1_to_s2_pvalue": results_1_to_2[lag]["f_pvalue"],
                    "s2_to_s1_pvalue": results_2_to_1[lag]["f_pvalue"],
                }
            )

        # Overall conclusion (using most significant lag)
        if causality_summary:
            # Find lag with strongest evidence
            min_pvalue_1_to_2 = min(r["s1_to_s2_pvalue"] for r in causality_summary)
            min_pvalue_2_to_1 = min(r["s2_to_s1_pvalue"] for r in causality_summary)

            if min_pvalue_1_to_2 < alpha and min_pvalue_2_to_1 < alpha:
                overall_relationship = "bidirectional"
            elif min_pvalue_1_to_2 < alpha:
                overall_relationship = "series1 -> series2"
            elif min_pvalue_2_to_1 < alpha:
                overall_relationship = "series2 -> series1"
            else:
                overall_relationship = "no significant causality"
        else:
            overall_relationship = "error"

        logger.info(f"Bidirectional causality: {overall_relationship}")

        return {
            "overall_relationship": overall_relationship,
            "series1_to_series2": results_1_to_2,
            "series2_to_series1": results_2_to_1,
            "lag_summary": causality_summary,
        }

    def pairwise_causality_matrix(
        self,
        data: pd.DataFrame,
        max_lag: int | None = None,
        alpha: float = 0.05,
    ) -> pd.DataFrame:
        """
        Test pairwise Granger causality for multiple series

        Args:
            data: DataFrame with multiple time series (columns = series)
            max_lag: Maximum lag to test
            alpha: Significance level

        Returns:
            DataFrame with pairwise causality results
        """
        max_lag = max_lag or self.max_lag
        columns = data.columns.tolist()
        n_series = len(columns)

        # Initialize results matrix
        causality_matrix = np.zeros((n_series, n_series))
        p_value_matrix = np.ones((n_series, n_series))

        for i, cause_var in enumerate(columns):
            for j, effect_var in enumerate(columns):
                if i == j:
                    continue

                # Test if cause_var -> effect_var
                results = self.granger_causality_test(
                    data[effect_var], data[cause_var], max_lag, verbose=False
                )

                if "error" not in results:
                    # Use minimum p-value across lags
                    min_pvalue = min(results[lag]["f_pvalue"] for lag in results.keys())
                    p_value_matrix[i, j] = min_pvalue

                    if min_pvalue < alpha:
                        causality_matrix[i, j] = 1

        # Create DataFrames
        causality_df = pd.DataFrame(causality_matrix, index=columns, columns=columns)

        pvalue_df = pd.DataFrame(p_value_matrix, index=columns, columns=columns)

        logger.success(f"Pairwise causality matrix computed for {n_series} series")

        return {
            "causality_matrix": causality_df,
            "p_value_matrix": pvalue_df,
        }

    def instantaneous_causality(
        self,
        series1: np.ndarray | pd.Series | pl.Series,
        series2: np.ndarray | pd.Series | pl.Series,
    ) -> dict[str, float]:
        """
        Test for instantaneous (contemporaneous) causality

        Tests if series1 and series2 have contemporaneous correlation
        after accounting for their own lags.

        Args:
            series1: First time series
            series2: Second time series

        Returns:
            Dictionary with instantaneous causality test results
        """
        from statsmodels.tsa.api import VAR

        # Convert to numpy
        if isinstance(series1, (pd.Series, pl.Series)):
            series1 = series1.to_numpy()
        if isinstance(series2, (pd.Series, pl.Series)):
            series2 = series2.to_numpy()

        # Create dataframe
        data = pd.DataFrame({"s1": series1, "s2": series2})

        # Fit VAR model
        model = VAR(data)
        results = model.fit(maxlags=self.max_lag, ic="aic")

        # Test instantaneous causality
        test_result = results.test_inst_causality(causing="s1")

        logger.info(f"Instantaneous causality test: chi2={test_result.test_statistic:.4f}")

        return {
            "chi2_statistic": test_result.test_statistic,
            "p_value": test_result.pvalue,
            "degrees_of_freedom": test_result.df,
            "significant": test_result.pvalue < 0.05,
        }


def test_granger_causality(
    y: np.ndarray | pd.Series | pl.Series,
    x: np.ndarray | pd.Series | pl.Series,
    max_lag: int = 10,
) -> dict[str, Any]:
    """
    Convenience function to test Granger causality

    Args:
        y: Dependent variable (effect)
        x: Independent variable (cause)
        max_lag: Maximum number of lags to test

    Returns:
        Dictionary with Granger causality test results

    Example:
        >>> # Generate sample data
        >>> np.random.seed(42)
        >>> x = np.random.randn(200)
        >>> y = np.zeros(200)
        >>> for t in range(2, 200):
        ...     y[t] = 0.5 * y[t-1] + 0.3 * x[t-1] + np.random.randn()
        >>>
        >>> # Test causality
        >>> results = test_granger_causality(y, x, max_lag=5)
        >>> for lag, result in results.items():
        ...     print(f"Lag {lag}: p-value = {result['f_pvalue']:.4f}")
    """
    tester = CausalityTester(max_lag=max_lag)
    return tester.granger_causality_test(y, x, max_lag)


def test_bidirectional_causality(
    series1: np.ndarray | pd.Series | pl.Series,
    series2: np.ndarray | pd.Series | pl.Series,
    max_lag: int = 10,
) -> dict[str, Any]:
    """
    Test for bidirectional Granger causality

    Args:
        series1: First time series
        series2: Second time series
        max_lag: Maximum lags to test

    Returns:
        Dictionary with bidirectional causality results

    Example:
        >>> import numpy as np
        >>> series1 = np.random.randn(200)
        >>> series2 = np.random.randn(200)
        >>> results = test_bidirectional_causality(series1, series2)
        >>> print(f"Relationship: {results['overall_relationship']}")
    """
    tester = CausalityTester(max_lag=max_lag)
    return tester.bidirectional_causality(series1, series2, max_lag)
