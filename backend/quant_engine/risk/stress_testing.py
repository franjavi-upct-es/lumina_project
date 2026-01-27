# backend/quant_engine/risk/stress_testing.py
"""
Portfolio Stress Testing Module

Evaluates portfolio performance under extreme market scenarios including:
- Historical stress events (2008 crisis, COVID-19, etc.)
- Hypothetical scenarios
- Factor stress tests
- Sensitivity analysis
- Reverse stress testing

Stress testing helps identify vulnerabilities and potential losses
in tail risk scenarios that may not be captured by VaR/CVaR.

References:
- Basel Committee on Banking Supervision. (2009). Principles for sound stress testing practices.
- Breuer, T., et al. (2009). How to find plausible, severe, and useful stress scenarios.
"""

from typing import Any

import numpy as np
import pandas as pd
from loguru import logger


class StressTester:
    """
    Portfolio stress testing framework

    Implements various stress testing methodologies to evaluate
    portfolio resilience under adverse market conditions.
    """

    def __init__(self):
        """Initialize stress tester"""
        self.historical_scenarios = self._initialize_historical_scenarios()
        logger.info("Stress tester initialized")

    def _initialize_historical_scenarios(self) -> dict[str, dict[str, float]]:
        """
        Define historical stress scenarios

        Returns:
            Dictionary of historical shock scenarios
        """
        return {
            "black_monday_1987": {
                "equity": -0.20,  # -20% in equities
                "volatility_multiplier": 3.0,
                "description": "Black Monday - October 19, 1987",
            },
            "dotcom_crash_2000": {
                "equity": -0.45,  # Tech stocks -45%
                "bonds": 0.08,
                "volatility_multiplier": 2.5,
                "description": "Dot-com bubble burst - 2000-2002",
            },
            "financial_crisis_2008": {
                "equity": -0.50,  # -50% in equities
                "bonds": 0.05,  # Flight to quality
                "credit_spread": 0.06,  # Credit spreads widen
                "volatility_multiplier": 4.0,
                "description": "Global Financial Crisis - 2008",
            },
            "flash_crash_2010": {
                "equity": -0.09,
                "volatility_multiplier": 5.0,
                "duration": "intraday",
                "description": "Flash Crash - May 6, 2010",
            },
            "eu_debt_crisis_2011": {
                "equity": -0.25,
                "bonds": -0.15,  # European bonds affected
                "credit_spread": 0.04,
                "description": "European Debt Crisis - 2011",
            },
            "brexit_2016": {
                "equity": -0.08,
                "fx_gbp": -0.10,  # GBP depreciation
                "volatility_multiplier": 2.0,
                "description": "Brexit Vote - June 24, 2016",
            },
            "covid_crash_2020": {
                "equity": -0.34,  # -34% peak to trough
                "bonds": 0.10,  # Flight to safety
                "credit_spread": 0.05,
                "volatility_multiplier": 5.0,
                "description": "COVID-19 Pandemic - March 2020",
            },
            "interest_rate_shock_2022": {
                "equity": -0.25,
                "bonds": -0.15,  # Bond bear market
                "rates": 0.03,  # 300 bps rate increase
                "description": "Interest Rate Shock - 2022",
            },
        }

    def apply_scenario_shock(
        self,
        returns_matrix: np.ndarray,
        weights: np.ndarray,
        scenario: dict[str, float],
        asset_classes: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Apply scenario shock to portfolio

        Args:
            returns_matrix: Historical returns (n_assets x n_periods)
            weights: Current portfolio weights
            scenario: Shock scenario dictionary
            asset_classes: Asset class labels for each asset

        Returns:
            Dictionary with stressed portfolio metrics
        """
        n_assets = len(weights)

        # Default asset classes if not provided
        if asset_classes is None:
            asset_classes = ["equity"] * n_assets

        # Apply shocks to each asset
        shocked_returns = np.zeros(n_assets)

        for i, asset_class in enumerate(asset_classes):
            if asset_class in scenario:
                shocked_returns[i] = scenario[asset_class]
            elif "equity" in scenario and asset_class == "equity":
                shocked_returns[i] = scenario["equity"]
            elif "bonds" in scenario and asset_class in ["bond", "bonds", "fixed_income"]:
                shocked_returns[i] = scenario["bonds"]
            else:
                # No shock for this asset class
                shocked_returns[i] = 0

        # Portfolio impact
        portfolio_shock = weights @ shocked_returns

        # Volatility adjustment if specified
        if "volatility_multiplier" in scenario:
            vol_mult = scenario["volatility_multiplier"]
        else:
            vol_mult = 1.0

        # Calculate stressed portfolio value
        initial_value = 1.0
        stressed_value = initial_value * (1 + portfolio_shock)
        loss = initial_value - stressed_value
        loss_pct = -portfolio_shock

        logger.info(f"Scenario: {scenario.get('description', 'Custom')}, Loss: {loss_pct:.2%}")

        return {
            "scenario_name": scenario.get("description", "Custom Scenario"),
            "portfolio_shock": portfolio_shock,
            "loss": loss,
            "loss_pct": loss_pct,
            "stressed_value": stressed_value,
            "volatility_multiplier": vol_mult,
            "asset_shocks": dict(zip(asset_classes, shocked_returns)),
        }

    def historical_stress_test(
        self,
        returns_matrix: np.ndarray,
        weights: np.ndarray,
        scenario_name: str,
        asset_classes: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Run stress test using historical scenario

        Args:
            returns_matrix: Asset returns matrix
            weights: Portfolio weights
            scenario_name: Name of historical scenario
            asset_classes: Asset class labels

        Returns:
            Stress test results
        """
        if scenario_name not in self.historical_scenarios:
            available = list(self.historical_scenarios.keys())
            raise ValueError(f"Unknown scenario: {scenario_name}. Available: {available}")

        scenario = self.historical_scenarios[scenario_name]

        return self.apply_scenario_shock(returns_matrix, weights, scenario, asset_classes)

    def multi_scenario_stress_test(
        self,
        returns_matrix: np.ndarray,
        weights: np.ndarray,
        scenarios: list[str] | None = None,
        asset_classes: list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Run multiple stress scenarios

        Args:
            returns_matrix: Asset returns matrix
            weights: Portfolio weights
            scenarios: List of scenario names (if None, runs all)
            asset_classes: Asset class labels

        Returns:
            DataFrame with results for all scenarios
        """
        if scenarios is None:
            scenarios = list(self.historical_scenarios.keys())

        results = []

        for scenario_name in scenarios:
            result = self.historical_stress_test(
                returns_matrix, weights, scenario_name, asset_classes
            )

            results.append(
                {
                    "scenario": scenario_name,
                    "description": result["scenario_name"],
                    "loss_pct": result["loss_pct"] * 100,
                    "portfolio_shock": result["portfolio_shock"],
                    "volatility_multiplier": result["volatility_multiplier"],
                }
            )

        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values("loss_pct", ascending=False)

        logger.success(f"Ran {len(scenarios)} stress scenarios")

        return results_df

    def sensitivity_analysis(
        self,
        returns_matrix: np.ndarray,
        weights: np.ndarray,
        factor: str = "equity",
        shock_range: tuple[float, float] = (-0.30, 0.30),
        n_steps: int = 13,
        asset_classes: list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Perform sensitivity analysis for a single factor

        Args:
            returns_matrix: Asset returns matrix
            weights: Portfolio weights
            factor: Factor to shock (e.g., 'equity', 'bonds', 'rates')
            shock_range: Range of shocks (min, max)
            n_steps: Number of shock levels to test
            asset_classes: Asset class labels

        Returns:
            DataFrame with sensitivity results
        """
        shocks = np.linspace(shock_range[0], shock_range[1], n_steps)

        results = []

        for shock in shocks:
            scenario = {factor: shock, "description": f"{factor} shock {shock:.1%}"}

            result = self.apply_scenario_shock(returns_matrix, weights, scenario, asset_classes)

            results.append(
                {
                    "shock_level": shock * 100,
                    "portfolio_return": result["portfolio_shock"] * 100,
                    "portfolio_value": result["stressed_value"],
                }
            )

        results_df = pd.DataFrame(results)

        logger.info(f"Sensitivity analysis complete for {factor}")

        return results_df

    def factor_stress_test(
        self,
        returns_matrix: np.ndarray,
        weights: np.ndarray,
        factor_shocks: dict[str, float],
        factor_exposures: np.ndarray,
    ) -> dict[str, Any]:
        """
        Stress test based on factor model

        R_portfolio = Σ (β_i * F_i) + ε

        Args:
            returns_matrix: Asset returns matrix
            weights: Portfolio weights
            factor_shocks: Dictionary of factor shocks
            factor_exposures: Matrix of factor exposures (n_assets x n_factors)

        Returns:
            Stress test results
        """
        # Calculate factor contribution to portfolio return
        factor_shock_vector = np.array(list(factor_shocks.values()))

        # Portfolio factor exposure
        portfolio_exposures = weights @ factor_exposures

        # Portfolio shock from factors
        portfolio_shock = portfolio_exposures @ factor_shock_vector

        # Individual asset shocks
        asset_shocks = factor_exposures @ factor_shock_vector

        stressed_value = 1.0 * (1 + portfolio_shock)

        logger.info(f"Factor stress test: {portfolio_shock:.2%} shock")

        return {
            "portfolio_shock": portfolio_shock,
            "loss_pct": -portfolio_shock,
            "stressed_value": stressed_value,
            "portfolio_exposures": portfolio_exposures,
            "asset_shocks": asset_shocks,
            "factor_contributions": portfolio_exposures * factor_shock_vector,
        }

    def monte_carlo_stress_test(
        self,
        returns_matrix: np.ndarray,
        weights: np.ndarray,
        n_simulations: int = 10000,
        shock_magnitude: float = 3.0,
        random_state: int | None = None,
    ) -> dict[str, Any]:
        """
        Monte Carlo stress testing with random shocks

        Args:
            returns_matrix: Asset returns matrix
            weights: Portfolio weights
            n_simulations: Number of Monte Carlo runs
            shock_magnitude: Standard deviations for shocks
            random_state: Random seed

        Returns:
            Monte Carlo stress test results
        """
        if random_state is not None:
            np.random.seed(random_state)

        # Calculate historical mean and covariance
        mean_returns = np.mean(returns_matrix, axis=1)
        cov_matrix = np.cov(returns_matrix)

        # Generate shocked returns
        # Use larger standard deviation for stress scenarios
        shocked_returns = np.random.multivariate_normal(
            mean_returns * shock_magnitude, cov_matrix * (shock_magnitude**2), n_simulations
        )

        # Calculate portfolio returns for each simulation
        portfolio_returns = shocked_returns @ weights

        # Statistics
        var_95 = np.percentile(portfolio_returns, 5)
        var_99 = np.percentile(portfolio_returns, 1)
        cvar_95 = np.mean(portfolio_returns[portfolio_returns <= var_95])
        worst_case = np.min(portfolio_returns)
        best_case = np.max(portfolio_returns)

        logger.info(f"Monte Carlo stress: VaR(95%)={-var_95:.2%}, Worst={worst_case:.2%}")

        return {
            "n_simulations": n_simulations,
            "mean_shock": np.mean(portfolio_returns),
            "std_shock": np.std(portfolio_returns),
            "var_95": -var_95,
            "var_99": -var_99,
            "cvar_95": -cvar_95,
            "worst_case": worst_case,
            "best_case": best_case,
            "simulated_returns": portfolio_returns,
        }

    def reverse_stress_test(
        self,
        returns_matrix: np.ndarray,
        weights: np.ndarray,
        max_loss_threshold: float = -0.20,
        asset_classes: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Reverse stress test: Find scenarios that cause specific loss

        Args:
            returns_matrix: Asset returns matrix
            weights: Portfolio weights
            max_loss_threshold: Target loss level (e.g., -0.20 for -20%)
            asset_classes: Asset class labels

        Returns:
            Scenarios that breach the threshold
        """
        n_assets = len(weights)

        if asset_classes is None:
            asset_classes = ["equity"] * n_assets

        # Simple approach: find uniform shock across asset classes
        unique_classes = list(set(asset_classes))

        scenarios = []

        for asset_class in unique_classes:
            # Find shock magnitude needed for this asset class
            # to cause the threshold loss

            # Calculate exposure to this asset class
            class_mask = np.array([ac == asset_class for ac in asset_classes])
            class_exposure = np.sum(weights[class_mask])

            if class_exposure > 0:
                # Required shock
                required_shock = max_loss_threshold / class_exposure

                scenario = {
                    asset_class: required_shock,
                    "description": f"{asset_class} shock to reach {max_loss_threshold:.1%} loss",
                }

                result = self.apply_scenario_shock(returns_matrix, weights, scenario, asset_classes)

                scenarios.append(
                    {
                        "asset_class": asset_class,
                        "required_shock_pct": required_shock * 100,
                        "actual_loss_pct": result["loss_pct"] * 100,
                        "exposure": class_exposure * 100,
                    }
                )

        scenarios_df = pd.DataFrame(scenarios)

        logger.info(f"Reverse stress test for {max_loss_threshold:.1%} loss threshold")

        return {
            "threshold": max_loss_threshold,
            "scenarios": scenarios_df,
        }

    def correlation_breakdown_test(
        self,
        returns_matrix: np.ndarray,
        weights: np.ndarray,
        correlation_increase: float = 0.5,
    ) -> dict[str, Any]:
        """
        Test portfolio under correlation breakdown (all correlations increase)

        In crisis, correlations tend to 1, reducing diversification benefits

        Args:
            returns_matrix: Asset returns matrix
            weights: Portfolio weights
            correlation_increase: How much to increase correlations

        Returns:
            Results under increased correlation
        """
        # Calculate current correlation and volatility
        corr_matrix = np.corrcoef(returns_matrix)
        std_devs = np.std(returns_matrix, axis=1)

        # Increase off-diagonal correlations
        stressed_corr = corr_matrix.copy()
        n = len(corr_matrix)

        for i in range(n):
            for j in range(i + 1, n):
                current_corr = stressed_corr[i, j]
                # Move correlation toward 1
                stressed_corr[i, j] = min(1.0, current_corr + correlation_increase)
                stressed_corr[j, i] = stressed_corr[i, j]

        # Reconstruct covariance matrix
        stressed_cov = np.outer(std_devs, std_devs) * stressed_corr

        # Portfolio variance under normal and stressed correlations
        normal_var = weights @ np.cov(returns_matrix) @ weights
        stressed_var = weights @ stressed_cov @ weights

        normal_vol = np.sqrt(normal_var)
        stressed_vol = np.sqrt(stressed_var)

        vol_increase = stressed_vol - normal_vol
        vol_increase_pct = vol_increase / normal_vol

        logger.info(
            f"Correlation breakdown: Vol increases from {normal_vol:.2%} "
            f"to {stressed_vol:.2%} (+{vol_increase_pct:.1%})"
        )

        return {
            "normal_volatility": normal_vol,
            "stressed_volatility": stressed_vol,
            "volatility_increase": vol_increase,
            "volatility_increase_pct": vol_increase_pct,
            "correlation_increase": correlation_increase,
        }


def run_stress_tests(
    returns_matrix: np.ndarray,
    weights: np.ndarray,
    asset_classes: list[str] | None = None,
    scenarios: list[str] | None = None,
) -> dict[str, Any]:
    """
    Convenience function to run comprehensive stress tests

    Args:
        returns_matrix: Asset returns matrix (n_assets x n_periods)
        weights: Portfolio weights
        asset_classes: Asset class labels
        scenarios: Specific scenarios to run (None = all)

    Returns:
        Comprehensive stress test results

    Example:
        >>> returns = np.random.normal(0.001, 0.02, (5, 1000))
        >>> weights = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
        >>> asset_classes = ['equity', 'equity', 'bonds', 'bonds', 'commodity']
        >>> results = run_stress_tests(returns, weights, asset_classes)
        >>> print(results['historical_scenarios'])
    """
    tester = StressTester()

    # Historical scenarios
    historical_results = tester.multi_scenario_stress_test(
        returns_matrix, weights, scenarios, asset_classes
    )

    # Monte Carlo stress
    mc_results = tester.monte_carlo_stress_test(returns_matrix, weights)

    # Correlation breakdown
    corr_results = tester.correlation_breakdown_test(returns_matrix, weights)

    # Reverse stress test
    reverse_results = tester.reverse_stress_test(
        returns_matrix, weights, max_loss_threshold=-0.20, asset_classes=asset_classes
    )

    logger.success("Comprehensive stress testing complete")

    return {
        "historical_scenarios": historical_results,
        "monte_carlo": mc_results,
        "correlation_breakdown": corr_results,
        "reverse_stress_test": reverse_results,
    }
