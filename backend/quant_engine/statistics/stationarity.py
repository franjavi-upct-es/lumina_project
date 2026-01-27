# backend/quant_engine/statistics/stationarity.py
"""
Stationarity Testing

Tests whether a time series is stationary (constant mean, variance, and autocorrelation).
Stationarity is a key assumption for many time series models.

Tests implemented:
- Augmented Dickey-Fuller (ADF) test
- KPSS (Kwiatkowski-Phillips-Schmidt-Shin) test
- Phillips-Perron (PP) test
- Variance Ratio test

A series is considered stationary if:
- Mean is constant over time
- Variance is constant over time
- Autocovariance depends only on lag, not on time

Applications:
- Model selection (ARIMA requires stationary series)
- Differencing requirements
- Risk modeling
- Pairs trading (cointegration requires non-stationary series)

References:
- Dickey, D. A., & Fuller, W. A. (1979). Distribution of the estimators.
- Kwiatkowski, D., et al. (1992). Testing the null hypothesis of stationarity.
- Phillips, P. C., & Perron, P. (1988). Testing for a unit root.
"""

from typing import Any

import numpy as np
import pandas as pd
import polars as pl
from arch.unitroot import PhillipsPerron, VarianceRatio
from loguru import logger
from statsmodels.tsa.stattools import adfuller, kpss


class StationarityTester:
    """
    Test time series for stationarity using multiple methods

    Provides comprehensive stationarity testing and differencing recommendations.
    """

    def __init__(self):
        """Initialize stationarity tester"""
        logger.info("Stationarity tester initialized")

    def adf_test(
        self,
        series: np.ndarray | pd.Series | pl.Series,
        regression: str = "c",
        maxlag: int | None = None,
        autolag: str = "AIC",
    ) -> dict[str, Any]:
        """
        Augmented Dickey-Fuller test for unit root

        Null hypothesis: Series has a unit root (non-stationary)
        Alternative: Series is stationary

        Args:
            series: Time series to test
            regression: Trend component
                - 'c': constant only (default)
                - 'ct': constant and trend
                - 'ctt': constant, linear and quadratic trend
                - 'nc': no constant, no trend
            maxlag: Maximum lag to use
            autolag: Method for automatic lag selection ('AIC', 'BIC', 't-stat')

        Returns:
            Dictionary with ADF test results
        """
        # Convert to numpy
        if isinstance(series, (pd.Series, pl.Series)):
            series = series.to_numpy()

        # Remove NaNs
        series = series[~np.isnan(series)]

        # Perform ADF test
        adf_result = adfuller(series, maxlag=maxlag, regression=regression, autolag=autolag)

        adf_stat = adf_result[0]
        p_value = adf_result[1]
        used_lag = adf_result[2]
        n_obs = adf_result[3]
        critical_values = adf_result[4]

        # Determine stationarity
        is_stationary = p_value < 0.05

        logger.info(
            f"ADF test: statistic={adf_stat:.4f}, p-value={p_value:.4f}, stationary={is_stationary}"
        )

        return {
            "test_statistic": adf_stat,
            "p_value": p_value,
            "used_lag": used_lag,
            "n_observations": n_obs,
            "critical_values": critical_values,
            "is_stationary": is_stationary,
            "method": "adf",
            "regression": regression,
        }

    def kpss_test(
        self,
        series: np.ndarray | pd.Series | pl.Series,
        regression: str = "c",
        nlags: str = "auto",
    ) -> dict[str, Any]:
        """
        KPSS test for stationarity

        Null hypothesis: Series is stationary (opposite of ADF!)
        Alternative: Series has a unit root

        Args:
            series: Time series to test
            regression: Trend component
                - 'c': constant (level stationary)
                - 'ct': constant and trend (trend stationary)
            nlags: Number of lags ('auto' or integer)

        Returns:
            Dictionary with KPSS test results
        """
        # Convert to numpy
        if isinstance(series, (pd.Series, pl.Series)):
            series = series.to_numpy()

        # Remove NaNs
        series = series[~np.isnan(series)]

        # Perform KPSS test
        kpss_result = kpss(series, regression=regression, nlags=nlags)

        kpss_stat = kpss_result[0]
        p_value = kpss_result[1]
        used_lag = kpss_result[2]
        critical_values = kpss_result[3]

        # Determine stationarity (opposite interpretation of ADF!)
        is_stationary = p_value > 0.05

        logger.info(
            f"KPSS test: statistic={kpss_stat:.4f}, p-value={p_value:.4f}, "
            f"stationary={is_stationary}"
        )

        return {
            "test_statistic": kpss_stat,
            "p_value": p_value,
            "used_lag": used_lag,
            "critical_values": critical_values,
            "is_stationary": is_stationary,
            "method": "kpss",
            "regression": regression,
        }

    def phillips_perron_test(
        self,
        series: np.ndarray | pd.Series | pl.Series,
        regression: str = "c",
        lags: int | None = None,
    ) -> dict[str, Any]:
        """
        Phillips-Perron test for unit root

        Similar to ADF but uses non-parametric corrections for serial correlation.

        Null hypothesis: Series has a unit root (non-stationary)
        Alternative: Series is stationary

        Args:
            series: Time series to test
            regression: Trend component ('c', 'ct', 'ctt', 'nc')
            lags: Number of lags for Newey-West correction

        Returns:
            Dictionary with PP test results
        """
        # Convert to numpy/pandas
        if isinstance(series, (np.ndarray, pl.Series)):
            series = pd.Series(series if isinstance(series, np.ndarray) else series.to_numpy())

        # Remove NaNs
        series = series.dropna()

        # Perform PP test
        pp = PhillipsPerron(series, trend=regression, lags=lags)

        pp_stat = pp.stat
        p_value = pp.pvalue

        # Determine stationarity
        is_stationary = p_value < 0.05

        logger.info(
            f"Phillips-Perron test: statistic={pp_stat:.4f}, p-value={p_value:.4f}, "
            f"stationary={is_stationary}"
        )

        return {
            "test_statistic": pp_stat,
            "p_value": p_value,
            "critical_values": pp.critical_values,
            "is_stationary": is_stationary,
            "method": "phillips_perron",
            "regression": regression,
        }

    def variance_ratio_test(
        self,
        series: np.ndarray | pd.Series | pl.Series,
        lags: int | list[int] = 2,
        robust: bool = True,
    ) -> dict[str, Any]:
        """
        Variance Ratio test for random walk hypothesis

        Tests if series follows a random walk (non-stationary).
        VR = Var(k-period return) / (k * Var(1-period return))

        Under random walk: VR should equal 1

        Args:
            series: Time series to test
            lags: Lag or list of lags to test
            robust: Use heteroskedasticity-robust standard errors

        Returns:
            Dictionary with VR test results
        """
        # Convert to pandas
        if isinstance(series, (np.ndarray, pl.Series)):
            series = pd.Series(series if isinstance(series, np.ndarray) else series.to_numpy())

        # Remove NaNs
        series = series.dropna()

        # Ensure lags is a list
        if isinstance(lags, int):
            lags = [lags]

        # Perform VR test
        vr = VarianceRatio(series, lags=lags, robust=robust)

        vr_stat = vr.stat
        p_value = vr.pvalue

        # VR significantly different from 1 indicates non-random walk (possibly stationary)
        is_random_walk = p_value > 0.05
        is_stationary = not is_random_walk

        logger.info(
            f"Variance Ratio test: VR={vr.vr:.4f}, p-value={p_value:.4f}, "
            f"random_walk={is_random_walk}"
        )

        return {
            "variance_ratio": vr.vr,
            "test_statistic": vr_stat,
            "p_value": p_value,
            "is_random_walk": is_random_walk,
            "is_stationary": is_stationary,
            "method": "variance_ratio",
            "lags": lags,
            "robust": robust,
        }

    def comprehensive_test(
        self,
        series: np.ndarray | pd.Series | pl.Series,
    ) -> dict[str, Any]:
        """
        Run all stationarity tests and provide overall conclusion

        Args:
            series: Time series to test

        Returns:
            Dictionary with results from all tests and overall conclusion
        """
        # Run all tests
        adf_result = self.adf_test(series)
        kpss_result = self.kpss_test(series)
        pp_result = self.phillips_perron_test(series)
        vr_result = self.variance_ratio_test(series)

        # Collect stationarity indicators
        tests_stationary = [
            adf_result["is_stationary"],
            kpss_result["is_stationary"],
            pp_result["is_stationary"],
            vr_result["is_stationary"],
        ]

        # Majority vote
        n_stationary = sum(tests_stationary)

        if n_stationary >= 3:
            overall_conclusion = "stationary"
        elif n_stationary <= 1:
            overall_conclusion = "non-stationary"
        else:
            overall_conclusion = "mixed_evidence"

        # Determine action
        if overall_conclusion == "stationary":
            recommendation = "Series appears stationary. No differencing needed."
        elif overall_conclusion == "non-stationary":
            recommendation = "Series appears non-stationary. Consider differencing."
        else:
            recommendation = "Mixed evidence. Recommend manual inspection and domain knowledge."

        logger.success(f"Comprehensive test: {overall_conclusion}")

        return {
            "adf": adf_result,
            "kpss": kpss_result,
            "phillips_perron": pp_result,
            "variance_ratio": vr_result,
            "overall_conclusion": overall_conclusion,
            "n_tests_stationary": n_stationary,
            "recommendation": recommendation,
        }

    def determine_differencing_order(
        self,
        series: np.ndarray | pd.Series | pl.Series,
        max_order: int = 2,
        alpha: float = 0.05,
    ) -> dict[str, Any]:
        """
        Determine optimal differencing order to achieve stationarity

        Args:
            series: Time series
            max_order: Maximum differencing order to test
            alpha: Significance level for stationarity tests

        Returns:
            Dictionary with differencing order and differenced series
        """
        # Convert to numpy
        if isinstance(series, (pd.Series, pl.Series)):
            series = series.to_numpy()

        current_series = series.copy()

        for d in range(max_order + 1):
            # Test current series
            adf_result = self.adf_test(current_series)

            if adf_result["p_value"] < alpha:
                # Series is stationary
                logger.info(f"Series is stationary after {d} differencing")

                return {
                    "differencing_order": d,
                    "is_stationary": True,
                    "adf_pvalue": adf_result["p_value"],
                    "differenced_series": current_series,
                }

            # Apply differencing for next iteration
            if d < max_order:
                current_series = np.diff(current_series)

        # If still not stationary after max_order differences
        logger.warning(f"Series still non-stationary after {max_order} differences")

        return {
            "differencing_order": max_order,
            "is_stationary": False,
            "adf_pvalue": adf_result["p_value"],
            "differenced_series": current_series,
            "warning": f"Still non-stationary after {max_order} differences",
        }

    def seasonal_differencing_test(
        self,
        series: np.ndarray | pd.Series | pl.Series,
        seasonal_period: int = 12,
    ) -> dict[str, Any]:
        """
        Test if seasonal differencing achieves stationarity

        Args:
            series: Time series
            seasonal_period: Seasonal period (e.g., 12 for monthly data)

        Returns:
            Dictionary with seasonal differencing results
        """
        # Convert to numpy
        if isinstance(series, (pd.Series, pl.Series)):
            series = series.to_numpy()

        # Apply seasonal differencing
        seasonal_diff = series[seasonal_period:] - series[:-seasonal_period]

        # Test original series
        original_test = self.adf_test(series)

        # Test seasonally differenced series
        seasonal_test = self.adf_test(seasonal_diff)

        improvement = seasonal_test["p_value"] < original_test["p_value"]

        logger.info(f"Seasonal differencing (period={seasonal_period}): improved={improvement}")

        return {
            "seasonal_period": seasonal_period,
            "original_pvalue": original_test["p_value"],
            "seasonal_diff_pvalue": seasonal_test["p_value"],
            "improvement": improvement,
            "is_stationary_after_seasonal_diff": seasonal_test["is_stationary"],
            "seasonal_differenced_series": seasonal_diff,
        }


def test_stationarity(
    series: np.ndarray | pd.Series | pl.Series, method: str = "adf", **kwargs
) -> dict[str, Any]:
    """
    Convenience function to test stationarity

    Args:
        series: Time series to test
        method: Test method
            - 'adf': Augmented Dickey-Fuller
            - 'kpss': KPSS test
            - 'pp': Phillips-Perron
            - 'vr': Variance Ratio
            - 'comprehensive': All tests
        **kwargs: Method-specific parameters

    Returns:
        Stationarity test results

    Example:
        >>> # Generate non-stationary series (random walk)
        >>> np.random.seed(42)
        >>> series = np.cumsum(np.random.randn(200))
        >>>
        >>> # Test stationarity
        >>> result = test_stationarity(series, method='adf')
        >>> print(f"Stationary: {result['is_stationary']}")
        >>> print(f"P-value: {result['p_value']:.4f}")

        >>> # Comprehensive test
        >>> result = test_stationarity(series, method='comprehensive')
        >>> print(f"Overall: {result['overall_conclusion']}")
        >>> print(f"Recommendation: {result['recommendation']}")
    """
    tester = StationarityTester()

    if method == "adf":
        return tester.adf_test(series, **kwargs)
    elif method == "kpss":
        return tester.kpss_test(series, **kwargs)
    elif method == "pp" or method == "phillips_perron":
        return tester.phillips_perron_test(series, **kwargs)
    elif method == "vr" or method == "variance_ratio":
        return tester.variance_ratio_test(series, **kwargs)
    elif method == "comprehensive":
        return tester.comprehensive_test(series)
    else:
        raise ValueError(f"Unknown method: {method}")


def make_stationary(
    series: np.ndarray | pd.Series | pl.Series,
    max_order: int = 2,
) -> dict[str, Any]:
    """
    Automatically make a series stationary through differencing

    Args:
        series: Time series to make stationary
        max_order: Maximum differencing order

    Returns:
        Dictionary with stationary series and differencing order

    Example:
        >>> # Generate non-stationary series
        >>> series = np.cumsum(np.random.randn(200))
        >>>
        >>> # Make stationary
        >>> result = make_stationary(series)
        >>> print(f"Differencing order: {result['differencing_order']}")
        >>> stationary_series = result['differenced_series']
    """
    tester = StationarityTester()
    return tester.determine_differencing_order(series, max_order)


def detect_stationarity_breakdown(
    series: np.ndarray | pd.Series | pl.Series,
    window: int = 100,
) -> pd.DataFrame:
    """
    Detect periods where stationarity breaks down using rolling tests

    Args:
        series: Time series
        window: Rolling window size

    Returns:
        DataFrame with rolling stationarity test results

    Example:
        >>> series = np.random.randn(500)
        >>> # Introduce non-stationarity in middle
        >>> series[200:300] = np.cumsum(series[200:300])
        >>>
        >>> # Detect breakdown
        >>> breakdown = detect_stationarity_breakdown(series, window=100)
        >>> print(breakdown[breakdown['is_stationary'] == False])
    """
    # Convert to numpy
    if isinstance(series, (pd.Series, pl.Series)):
        series = series.to_numpy()

    tester = StationarityTester()
    n = len(series)

    results = []

    for i in range(window, n):
        window_data = series[i - window : i]

        try:
            adf_result = tester.adf_test(window_data)

            results.append(
                {
                    "period": i,
                    "adf_statistic": adf_result["test_statistic"],
                    "p_value": adf_result["p_value"],
                    "is_stationary": adf_result["is_stationary"],
                }
            )
        except Exception as e:
            logger.warning(f"Rolling test failed at period {i}: {e}")
            continue

    results_df = pd.DataFrame(results)

    logger.success(f"Stationarity breakdown detection complete: {len(results)} windows")

    return results_df
