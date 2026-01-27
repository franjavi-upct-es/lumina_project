# backend/quant_engine/risk/var_calculator.py
"""
Value at Risk (VaR) Calculator

Calculates VaR using multiple methodologies:
- Historical Simulation
- Parametric (Variance-Covariance)
- Modified VaR (Cornish-Fisher)
- Monte Carlo Simulation
- Filtered Historical Simulation (GARCH)

VaR represents the maximum expected loss over a given time horizon
at a specified confidence level.

References:
- Jorion, P. (2006). Value at risk: the new benchmark for managing financial risk.
- McNeil, A. J., Frey, R., & Embrechts, P. (2015). Quantitative risk management.
"""

from typing import Any

import numpy as np
import pandas as pd
import polars as pl
from arch import arch_model
from loguru import logger
from scipy import stats


class VaRCalculator:
    """
    Calculate Value at Risk using multiple methods

    VaR answers: "What is the maximum loss I can expect with X% confidence?"
    """

    def __init__(
        self,
        confidence_level: float = 0.95,
        holding_period: int = 1,
    ):
        """
        Initialize VaR calculator

        Args:
            confidence_level: Confidence level (e.g., 0.95 for 95%)
            holding_period: Holding period in days
        """
        self.confidence_level = confidence_level
        self.holding_period = holding_period
        self.alpha = 1 - confidence_level

        logger.info(
            f"VaR calculator initialized: CL={confidence_level}, horizon={holding_period} day(s)"
        )

    def historical_var(
        self,
        returns: np.ndarray,
        weights: np.ndarray | None = None,
    ) -> dict[str, float]:
        """
        Calculate VaR using historical simulation

        Most widely used non-parametric method. Uses actual historical
        returns without assumptions about distribution.

        Args:
            returns: Historical returns (1D array or 2D for multi-asset)
            weights: Portfolio weights (if multi-asset)

        Returns:
            Dictionary with VaR and related metrics
        """
        # Calculate portfolio returns if needed
        if weights is not None and returns.ndim == 2:
            portfolio_returns = returns.T @ weights
        else:
            portfolio_returns = returns

        # Adjust for holding period
        if self.holding_period > 1:
            # Aggregate returns over holding period
            n_periods = len(portfolio_returns) // self.holding_period
            aggregated_returns = []

            for i in range(n_periods):
                period_return = (
                    np.prod(
                        1
                        + portfolio_returns[i * self.holding_period : (i + 1) * self.holding_period]
                    )
                    - 1
                )
                aggregated_returns.append(period_return)

            portfolio_returns = np.array(aggregated_returns)

        # Calculate VaR as percentile
        var = -np.percentile(portfolio_returns, self.alpha * 100)

        # Additional statistics
        mean_return = np.mean(portfolio_returns)
        std_return = np.std(portfolio_returns)
        min_return = np.min(portfolio_returns)

        logger.debug(f"Historical VaR: {var:.4f}")

        return {
            "var": var,
            "var_pct": var * 100,
            "mean_return": mean_return,
            "std_return": std_return,
            "worst_loss": -min_return,
            "method": "historical",
        }

    def parametric_var(
        self,
        returns: np.ndarray,
        weights: np.ndarray | None = None,
        distribution: str = "normal",
    ) -> dict[str, float]:
        """
        Calculate VaR using parametric method (variance-covariance)

        Assumes returns follow a specific distribution (usually normal).
        Fast and easy to calculate, but relies on distributional assumptions.

        Args:
            returns: Historical returns
            weights: Portfolio weights
            distribution: 'normal' or 't' (Student's t)

        Returns:
            Dictionary with VaR metrics
        """
        # Calculate portfolio returns
        if weights is not None and returns.ndim == 2:
            portfolio_returns = returns.T @ weights
        else:
            portfolio_returns = returns

        # Calculate parameters
        mu = np.mean(portfolio_returns)
        sigma = np.std(portfolio_returns)

        # Adjust for holding period
        mu_adj = mu * self.holding_period
        sigma_adj = sigma * np.sqrt(self.holding_period)

        # Calculate VaR based on distribution
        if distribution == "normal":
            z_alpha = stats.norm.ppf(self.alpha)
            var = -(mu_adj + z_alpha * sigma_adj)

        elif distribution == "t":
            # Fit Student's t distribution
            df, loc, scale = stats.t.fit(portfolio_returns)
            t_alpha = stats.t.ppf(self.alpha, df)
            var = -(mu_adj + t_alpha * sigma_adj)

        else:
            raise ValueError(f"Unknown distribution: {distribution}")

        logger.debug(f"Parametric VaR ({distribution}): {var:.4f}")

        return {
            "var": var,
            "var_pct": var * 100,
            "mean": mu,
            "std": sigma,
            "distribution": distribution,
            "method": "parametric",
        }

    def modified_var(
        self,
        returns: np.ndarray,
        weights: np.ndarray | None = None,
    ) -> dict[str, float]:
        """
        Calculate modified VaR using Cornish-Fisher expansion

        Adjusts parametric VaR for skewness and kurtosis.
        Better for non-normal distributions.

        Args:
            returns: Historical returns
            weights: Portfolio weights

        Returns:
            Dictionary with VaR metrics
        """
        # Calculate portfolio returns
        if weights is not None and returns.ndim == 2:
            portfolio_returns = returns.T @ weights
        else:
            portfolio_returns = returns

        # Calculate moments
        mu = np.mean(portfolio_returns)
        sigma = np.std(portfolio_returns)
        skew = stats.skew(portfolio_returns)
        kurt = stats.kurtosis(portfolio_returns)  # Excess kurtosis

        # Adjust for holding period
        mu_adj = mu * self.holding_period
        sigma_adj = sigma * np.sqrt(self.holding_period)

        # Cornish-Fisher expansion
        z_alpha = stats.norm.ppf(self.alpha)

        z_cf = (
            z_alpha
            + (z_alpha**2 - 1) * skew / 6
            + (z_alpha**3 - 3 * z_alpha) * kurt / 24
            - (2 * z_alpha**3 - 5 * z_alpha) * skew**2 / 36
        )

        var = -(mu_adj + z_cf * sigma_adj)

        logger.debug(f"Modified VaR: {var:.4f} (skew={skew:.3f}, kurt={kurt:.3f})")

        return {
            "var": var,
            "var_pct": var * 100,
            "mean": mu,
            "std": sigma,
            "skewness": skew,
            "kurtosis": kurt,
            "method": "modified",
        }

    def monte_carlo_var(
        self,
        returns: np.ndarray,
        weights: np.ndarray | None = None,
        n_simulations: int = 10000,
        method: str = "normal",
        random_state: int | None = None,
    ) -> dict[str, float]:
        """
        Calculate VaR using Monte Carlo simulation

        Args:
            returns: Historical returns
            weights: Portfolio weights
            n_simulations: Number of simulations
            method: 'normal', 'bootstrap', or 'historical'
            random_state: Random seed

        Returns:
            Dictionary with VaR metrics
        """
        if random_state is not None:
            np.random.seed(random_state)

        # Calculate portfolio returns
        if weights is not None and returns.ndim == 2:
            portfolio_returns = returns.T @ weights
        else:
            portfolio_returns = returns

        # Generate simulations based on method
        if method == "normal":
            mu = np.mean(portfolio_returns)
            sigma = np.std(portfolio_returns)
            simulated_returns = np.random.normal(mu, sigma, n_simulations)

        elif method == "bootstrap":
            # Resample from historical returns
            simulated_returns = np.random.choice(
                portfolio_returns, size=n_simulations, replace=True
            )

        elif method == "historical":
            # Use historical distribution directly
            simulated_returns = portfolio_returns
            n_simulations = len(portfolio_returns)

        else:
            raise ValueError(f"Unknown method: {method}")

        # Adjust for holding period
        if self.holding_period > 1:
            adjusted_returns = []
            for _ in range(n_simulations):
                period_returns = np.random.choice(simulated_returns, self.holding_period)
                period_return = np.prod(1 + period_returns) - 1
                adjusted_returns.append(period_return)
            simulated_returns = np.array(adjusted_returns)

        # Calculate VaR
        var = -np.percentile(simulated_returns, self.alpha * 100)

        logger.debug(f"Monte Carlo VaR ({method}): {var:.4f}")

        return {
            "var": var,
            "var_pct": var * 100,
            "n_simulations": n_simulations,
            "mean_simulated": np.mean(simulated_returns),
            "std_simulated": np.std(simulated_returns),
            "method": f"monte_carlo_{method}",
        }

    def garch_var(
        self,
        returns: np.ndarray,
        weights: np.ndarray | None = None,
        p: int = 1,
        q: int = 1,
    ) -> dict[str, float]:
        """
        Calculate VaR using GARCH model for volatility forecasting

        Filtered Historical Simulation - uses GARCH to model time-varying volatility

        Args:
            returns: Historical returns
            weights: Portfolio weights
            p: GARCH lag order
            q: ARCH lag order

        Returns:
            Dictionary with VaR metrics
        """
        # Calculate portfolio returns
        if weights is not None and returns.ndim == 2:
            portfolio_returns = returns.T @ weights
        else:
            portfolio_returns = returns

        # Scale to percentage for GARCH
        returns_pct = portfolio_returns * 100

        # Fit GARCH model
        model = arch_model(returns_pct, vol="Garch", p=p, q=q, dist="normal")
        fitted_model = model.fit(disp="off", show_warning=False)

        # Forecast volatility for next period
        forecast = fitted_model.forecast(horizon=self.holding_period)
        forecasted_variance = forecast.variance.values[-1, 0]
        forecasted_vol = np.sqrt(forecasted_variance)

        # Calculate VaR using forecasted volatility
        mu = np.mean(returns_pct)
        z_alpha = stats.norm.ppf(self.alpha)

        var = -(mu * self.holding_period + z_alpha * forecasted_vol)
        var = var / 100  # Convert back to decimal

        logger.debug(f"GARCH VaR: {var:.4f} (forecasted vol: {forecasted_vol:.2f}%)")

        return {
            "var": var,
            "var_pct": var * 100,
            "forecasted_volatility": forecasted_vol,
            "mean": mu / 100,
            "garch_params": {"p": p, "q": q},
            "method": "garch",
        }

    def calculate(
        self,
        returns: np.ndarray | pd.Series | pl.Series,
        weights: np.ndarray | None = None,
        method: str = "historical",
        **kwargs,
    ) -> dict[str, float]:
        """
        Calculate VaR using specified method

        Args:
            returns: Return data
            weights: Portfolio weights
            method: VaR calculation method
                - 'historical': Historical simulation
                - 'parametric': Variance-covariance
                - 'modified': Cornish-Fisher VaR
                - 'monte_carlo': Monte Carlo simulation
                - 'garch': GARCH-based VaR
            **kwargs: Method-specific parameters

        Returns:
            Dictionary with VaR and related metrics
        """
        # Convert to numpy
        if isinstance(returns, (pd.Series, pl.Series)):
            returns = returns.to_numpy()

        # Route to appropriate method
        if method == "historical":
            return self.historical_var(returns, weights)

        elif method == "parametric":
            distribution = kwargs.get("distribution", "normal")
            return self.parametric_var(returns, weights, distribution)

        elif method == "modified":
            return self.modified_var(returns, weights)

        elif method == "monte_carlo":
            n_sims = kwargs.get("n_simulations", 10000)
            mc_method = kwargs.get("mc_method", "normal")
            random_state = kwargs.get("random_state", None)
            return self.monte_carlo_var(returns, weights, n_sims, mc_method, random_state)

        elif method == "garch":
            p = kwargs.get("p", 1)
            q = kwargs.get("q", 1)
            return self.garch_var(returns, weights, p, q)

        else:
            raise ValueError(f"Unknown method: {method}")

    def backtest_var(
        self,
        returns: np.ndarray,
        var_forecasts: np.ndarray,
    ) -> dict[str, Any]:
        """
        Backtest VaR forecasts against actual returns

        Args:
            returns: Actual returns
            var_forecasts: VaR forecasts (positive values)

        Returns:
            Dictionary with backtest results
        """
        # Count violations (actual loss exceeds VaR)
        violations = returns < -var_forecasts
        n_violations = np.sum(violations)
        violation_rate = n_violations / len(returns)

        # Expected violation rate
        expected_violation_rate = self.alpha

        # Traffic light test (Basel)
        if violation_rate <= expected_violation_rate:
            zone = "green"
        elif violation_rate <= expected_violation_rate * 1.5:
            zone = "yellow"
        else:
            zone = "red"

        # Kupiec test (unconditional coverage)
        # H0: Actual violation rate = Expected violation rate
        if n_violations > 0:
            lr_uc = -2 * np.log(
                (self.alpha**n_violations) * ((1 - self.alpha) ** (len(returns) - n_violations))
            ) + 2 * np.log(
                (violation_rate**n_violations)
                * ((1 - violation_rate) ** (len(returns) - n_violations))
            )
            p_value_uc = 1 - stats.chi2.cdf(lr_uc, df=1)
        else:
            lr_uc = np.nan
            p_value_uc = 1.0

        logger.info(
            f"VaR Backtest: {n_violations}/{len(returns)} violations "
            f"({violation_rate:.2%}), Zone: {zone}"
        )

        return {
            "n_violations": n_violations,
            "violation_rate": violation_rate,
            "expected_violation_rate": expected_violation_rate,
            "traffic_light_zone": zone,
            "kupiec_lr": lr_uc,
            "kupiec_p_value": p_value_uc,
            "kupiec_reject": p_value_uc < 0.05 if not np.isnan(p_value_uc) else False,
        }


def calculate_var(
    returns: np.ndarray | pd.Series | pl.Series,
    confidence_level: float = 0.95,
    holding_period: int = 1,
    method: str = "historical",
    weights: np.ndarray | None = None,
    **kwargs,
) -> dict[str, float]:
    """
    Convenience function to calculate VaR

    Args:
        returns: Return data
        confidence_level: Confidence level (0 to 1)
        holding_period: Holding period in days
        method: Calculation method
        weights: Portfolio weights
        **kwargs: Method-specific parameters

    Returns:
        Dictionary with VaR metrics

    Example:
        >>> returns = np.random.normal(0.001, 0.02, 1000)
        >>> var_metrics = calculate_var(returns, confidence_level=0.95, method='historical')
        >>> print(f"VaR (95%): {var_metrics['var_pct']:.2f}%")

        >>> # Portfolio VaR
        >>> returns_matrix = np.random.normal(0.001, 0.02, (3, 1000))
        >>> weights = np.array([0.4, 0.4, 0.2])
        >>> portfolio_var = calculate_var(returns_matrix, weights=weights, method='parametric')
    """
    calculator = VaRCalculator(confidence_level, holding_period)
    return calculator.calculate(returns, weights, method, **kwargs)


def compare_var_methods(
    returns: np.ndarray,
    confidence_level: float = 0.95,
    weights: np.ndarray | None = None,
) -> pd.DataFrame:
    """
    Compare VaR across multiple methods

    Args:
        returns: Return data
        confidence_level: Confidence level
        weights: Portfolio weights

    Returns:
        DataFrame comparing different VaR methods

    Example:
        >>> returns = np.random.normal(0.001, 0.02, 1000)
        >>> comparison = compare_var_methods(returns)
        >>> print(comparison)
    """
    calculator = VaRCalculator(confidence_level)

    methods = ["historical", "parametric", "modified", "monte_carlo"]
    results = []

    for method in methods:
        try:
            result = calculator.calculate(returns, weights, method)
            results.append(
                {
                    "method": method,
                    "var": result["var"],
                    "var_pct": result["var_pct"],
                }
            )
        except Exception as e:
            logger.warning(f"Could not calculate VaR using {method}: {e}")

    comparison_df = pd.DataFrame(results)
    comparison_df = comparison_df.sort_values("var", ascending=False)

    logger.success(f"Compared {len(results)} VaR methods")

    return comparison_df
