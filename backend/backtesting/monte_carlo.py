# backend/backtesting/monte_carlo.py
"""
Monte Carlo simulation for portfolio analysis and risk assessment
Generates thousands of possible future scenarios based on historical statistics
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger


@dataclass
class MonteCarloConfig:
    """Configuration for Monte Carlo simulation"""

    num_simulations: int = 10000
    time_horizon_days: int = 252  # 1 year
    initial_value: float = 100000.0
    confidence_levels: list[float] = []
    seed: int | None = 42

    def __post_init__(self):
        if self.confidence_levels is None:
            self.confidence_levels = [0.95, 0.99]


class MonteCarloSimulator:
    """
    Monte Carlo simulator for portfolio analysis

    Features:
    - Multiple simulation methods (geometric Browniano motion, historical bootstrap)
    - Confidence intervals and percentiles
    - Risk metrics (VaR, CVaR)
    - Path-dependent analysis
    """

    def __init__(self, config: MonteCarloConfig | None = None):
        self.config = config or MonteCarloConfig()

        if self.config.seed is not None:
            np.random.seed(self.config.seed)

        # Storage for results
        self.simulated_paths: np.ndarray | None = None
        self.final_values: np.ndarray | None = None
        self.returns_history: np.ndarray | None = None

    def simulate_gbm(
        self,
        mean_return: float,
        volatility: float,
        correlation_matrix: np.ndarray | None = None,
        num_assets: int = 1,
    ) -> np.ndarray:
        """
        Simulate using Geometric Brownian Motion

        dS = μS dt + σS dW

        Args:
            mean_return: Expected annual return
            volatility: Annual volatility
            correlation_matrix: Correlation matrix for multiple assets
            num_assets: Number of assets to simulate

        Returns:
            Array of shape (num_simulations, time_horizon, num_assets)
        """
        logger.info(
            f"Running GBM simulation: {self.config.num_simulations} paths, {self.config.time_horizon_days} days"
        )

        dt = 1 / 252  # Daily time step

        # Generate random returns
        if num_assets == 1 or correlation_matrix is None:
            # Single asset or uncorrelated assets
            random_returns = np.random.normal(
                mean_return * dt,
                volatility * np.sqrt(dt),
                (self.config.num_simulations, self.config.time_horizon_days, num_assets),
            )
        else:
            # Multiple correlated assets
            mean_vector = np.full(num_assets, mean_return * dt)
            cov_matrix = correlation_matrix * (volatility**2) * dt

            random_returns = np.random.multivariate_normal(
                mean_vector,
                cov_matrix,
                (self.config.num_simulations, self.config.time_horizon_days),
            )
            random_returns = random_returns.reshape(
                self.config.num_simulations, self.config.time_horizon_days, num_assets
            )

        # Calculate price paths
        price_paths = self.config.initial_value * np.exp(np.cumsum(random_returns, axis=1))

        self.simulated_paths = price_paths
        self.final_values = (
            price_paths[:, -1, 0] if num_assets == 1 else price_paths[:, -1, :].sum(axis=1)
        )

        logger.success(f"Simulation complete: {self.config.num_simulations} paths generated")

        return price_paths

    def simulate_historical_bootstrap(
        self,
        historical_returns: pd.Series,
        block_size: int = 1,
    ) -> np.ndarray:
        """
        Simulate using historical bootstrap method
        Samples from actual historical returns to preserve distribution characteristics

        Args:
            historical_returns: Historical returns data
            block_size: Size of blocks to sample (1 = random sampling, >1 = block bootstrap)

        Returns:
            Array of simulated paths
        """
        logger.info(f"Running historical bootstrap: block_size={block_size}")

        returns_array = historical_returns.values
        paths = np.zeros((self.config.num_simulations, self.config.time_horizon_days))

        for sim in range(self.config.num_simulations):
            if block_size == 1:
                # Simple random sampling
                sampled_returns = np.random.choice(
                    returns_array,
                    size=self.config.time_horizon_days,
                    replace=True,
                )
            else:
                # Block bootstrap
                sampled_returns = self._block_bootstrap(returns_array, block_size)

            # Calculate cumulative path
            paths[sim] = self.config.initial_value * np.cumprod(1 + sampled_returns)

        self.simulated_paths = paths
        self.final_values = paths[:, -1]

        return paths

    def _block_bootstrap(self, data: np.ndarray, block_size: int) -> np.ndarray:
        """
        Perform block bootstrap to preserve serial correlation

        Args:
            data: Input data array
            block_size: Size of blocks to sample

        Returns:
            Bootstrapped array of length time_horizon_days
        """
        num_blocks = int(np.ceil(self.config.time_horizon_days / block_size))
        result = []

        for _ in range(num_blocks):
            # Random starting point
            start_idx = np.random.randint(0, len(data) - block_size)
            block = data[start_idx : start_idx + block_size]
            result.extend(block)

        return np.array(result[: self.config.time_horizon_days])

    def simulate_jump_diffusion(
        self,
        mean_return: float,
        volatility: float,
        jump_intensity: float = 0.1,
        jump_mean: float = -0.02,
        jump_std: float = 0.03,
    ) -> np.ndarray:
        """
        Simulate using Jump Diffusion model (Merton model)
        Includes both continuous diffusion and discrete jumps

        Args:
            mean_return: Drift term
            volatility: Diffusion volatility
            jump_intensity: Average number of jumps per year
            jump_std: Standard deviation of jump size

        Returns:
            Array of simulated paths
        """
        logger.info("Running jump diffusion simulation")

        dt = 1 / 252
        paths = np.zeros((self.config.num_simulations, self.config.time_horizon_days))

        for sim in range(self.config.num_simulations):
            price = self.config.initial_value

            for t in range(self.config.time_horizon_days):
                # Diffusion component
                dW = np.random.normal(0, np.sqrt(dt))
                diffusion = mean_return * dt + volatility * dW

                # Jump component
                num_jumps = np.random.poisson(jump_intensity * dt)
                jump = 0
                if num_jumps > 0:
                    jump = np.sum(np.random.normal(jump_mean, jump_std, num_jumps))

                # Update price
                price *= np.exp(diffusion + jump)
                paths[sim, t] = price

        self.simulated_paths = paths
        self.final_values = paths[:, -1]

        return paths

    def calculate_metrics(self) -> dict[str, Any]:
        """
        Calculate comprehensive metrics from simulation results

        Returns:
            Dictionary of metrics including percentiles, VaR, CVaR, etc.
        """
        if self.final_values is None:
            raise ValueError("No simulation results available. Run a simulation first.")

        # Percentiles
        percentiles = {
            f"{int(p * 100)}th": float(np.percentile(self.final_values, p * 100))
            for p in [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
        }

        # Basic statistics
        mean_value = float(np.mean(self.final_values))
        std_value = float(np.std(self.final_values))

        # Returns
        returns = (self.final_values - self.config.initial_value) / self.config.initial_value
        mean_return = float(np.mean(returns))
        std_return = float(np.std(returns))

        # Probability metrics
        prob_profit = float(np.mean(self.final_values > self.config.initial_value))
        prob_loss_10 = float(np.mean(returns < -0.10))
        prob_loss_20 = float(np.mean(returns < -0.20))

        # Value at Risk (VaR)
        var_metrics = {}
        for conf in self.config.confidence_levels:
            percentile = (1 - conf) * 100
            var_value = float(np.percentile(self.final_values, percentile))
            var_loss = self.config.initial_value - var_value
            var_pct = var_loss / self.config.initial_value

            var_metrics[f"VaR_{int(conf * 100)}"] = {
                "value": var_value,
                "loss": var_loss,
                "loss_pct": var_pct,
            }

        # Conditional Value at Risk (CVaR / Expected Shortfall)
        cvar_metrics = {}
        for conf in self.config.confidence_levels:
            percentile = (1 - conf) * 100
            var_threshold = np.percentile(self.final_values, percentile)
            tail_losses = self.final_values[self.final_values <= var_threshold]

            if len(tail_losses) > 0:
                cvar_value = float(np.mean(tail_losses))
                cvar_loss = self.config.initial_value - cvar_value
                cvar_pct = cvar_loss / self.config.initial_value

                cvar_metrics[f"CVaR_{int(conf * 100)}"] = {
                    "value": cvar_value,
                    "loss": cvar_loss,
                    "loss_pct": cvar_pct,
                }

        # Sharpe ratio (annualized)
        risk_free_rate = 0.05
        years = self.config.time_horizon_days / 252
        annualized_return = (mean_value / self.config.initial_value) ** (1 / years) - 1
        annualized_vol = std_return / np.sqrt(years)
        sharpe_ratio = (
            (annualized_return - risk_free_rate) / annualized_vol if annualized_vol > 0 else 0
        )

        return {
            "initial_value": self.config.initial_value,
            "num_simulations": self.config.num_simulations,
            "time_horizon_days": self.config.time_horizon_days,
            "percentiles": percentiles,
            "mean_final_value": mean_value,
            "std_final_value": std_value,
            "mean_return": mean_return,
            "std_return": std_return,
            "probability_of_profic": prob_profit,
            "probability_loss_10pct": prob_loss_10,
            "probability_loss_20pct": prob_loss_20,
            "var": var_metrics,
            "cvar": cvar_metrics,
            "annualized_return": float(annualized_return),
            "annualized_volatility": float(annualized_vol),
            "sharpe_ratio": float(sharpe_ratio),
        }

    def get_confidence_intervals(
        self,
        confidence_level: float = 0.95,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get confidence intervals over time

        Args:
            confidence_level: Confidence level (e.g., 0.95 for 95%)

        Returns:
            Tuple of (median, lower_bound, upper_bound) paths
        """
        if self.simulated_paths is None:
            raise ValueError("No simulation results available")

        alpha = 1 - confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100

        median = np.percentile(self.simulated_paths, 50, axis=0)
        lower = np.percentile(self.simulated_paths, lower_percentile, axis=0)
        upper = np.percentile(self.simulated_paths, upper_percentile, axis=0)

        return median, lower, upper

    def analyze_drawdown(self) -> dict[str, Any]:
        """
        Analyze drawdowns across all simulated paths

        Returns:
            Dictionary with drawdown statistics
        """
        if self.simulated_paths is None:
            raise ValueError("No simulation results available")

        all_drawdowns = []
        max_drawdowns = []

        for path in self.simulated_paths:
            # Calculate running maximum
            running_max = np.maximum.accumulate(path)

            # Calculate drawdown
            drawdown = (path - running_max) / running_max

            all_drawdowns.append(drawdown)
            max_drawdowns.append(np.min(drawdown))

        all_drawdowns = np.array(all_drawdowns)
        max_drawdowns = np.array(max_drawdowns)

        return {
            "mean_max_drawdown": float(np.mean(max_drawdowns)),
            "median_max_drawdown": float(np.median(max_drawdowns)),
            "worst_drawdown": float(np.min(max_drawdowns)),
            "drawdown_percentiles": {
                "5th": float(np.percentile(max_drawdowns, 5)),
                "25th": float(np.percentile(max_drawdowns, 25)),
                "50th": float(np.percentile(max_drawdowns, 50)),
                "75th": float(np.percentile(max_drawdowns, 75)),
                "95th": float(np.percentile(max_drawdowns, 95)),
            },
        }

    def get_sample_paths(self, num_samples: int = 100) -> np.ndarray:
        """
        Get a random sample of paths for visualization

        Args:
            num_samples: Number of paths to sample

        Returns:
            Array of sampled paths
        """
        if self.simulated_paths is None:
            raise ValueError("No simulation results available")

        num_samples = min(num_samples, self.config.num_simulations)
        indices = np.random.choice(self.config.num_simulations, num_samples, replace=False)

        return self.simulated_paths[indices]

    def scenario_analysis(self, scenarios: dict[str, dict[str, float]]) -> dict[str, float]:
        """
        Analyze specific scenarios

        Args:
            scenarios: Dictionary of scenarios with parameters
                      e.g., {"crash": {"return": -0.30, "volatility": 0.50}}

        Returns:
            Dictionary of scenario outcomes
        """
        results = {}

        for scenario_name, params in scenarios.items():
            logger.info(f"Analyzing scenario: {scenario_name}")

            # Run simulation with scenario parameters
            original_config = self.config
            temp_paths = self.simulate_gbm(
                mean_return=params.get("return", 0.0), volatility=params.get("volatility", 0.15)
            )

            # Calculate final value
            scenario_final = temp_paths[:, -1, 0]
            mean_final = float(np.mean(scenario_final))

            results[scenario_name] = {
                "mean_final_value": mean_final,
                "mean_return": (mean_final - self.config.initial_value) / self.config.initial_value,
                "probability_loss": float(np.mean(scenario_final < self.config.initial_value)),
            }

            # Restore original config
            self.config = original_config

        return results


def run_portfolio_monte_carlo(
    returns_data: pd.DataFrame,
    weights: np.ndarray,
    initial_value: float = 100000.0,
    num_simulations: int = 10000,
    time_horizon_days: int = 252,
) -> dict[str, Any]:
    """
    Convenience function to run Monte Carlo for a portfolio

    Args:
        returns_data: DataFrame of asset returns
        weights: Portfolio weights
        initial_value: Starting portfolio value
        num_simulations: Number of simulations
        time_horizon_days: Time horizon in days

    Returns:
        Dictionary with simulation results and metrics
    """
    # Calculate portfolio statistics
    mean_returns = returns_data.mean()
    cov_matrix = returns_data.cov()

    # Portfolio expected return and volatility
    portfolio_return = np.dot(weights, mean_returns) * 252
    portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix * 252, weights)))

    # Run simulation
    config = MonteCarloConfig(
        num_simulations=num_simulations,
        time_horizon_days=time_horizon_days,
        initial_value=initial_value,
    )

    simulator = MonteCarloSimulator(config)

    # Use GBM with portfolio parameters
    paths = simulator.simulate_gbm(
        mean_return=float(portfolio_return), volatility=float(portfolio_vol)
    )

    # Calculate metrics
    metrics = simulator.calculate_metrics()
    drawdown_analysis = simulator.analyze_drawdown()

    # Get confidence intervals
    median, lower_95, upper_95 = simulator.get_confidence_intervals(0.95)

    return {
        "metrics": metrics,
        "drawdown_analysis": drawdown_analysis,
        "confidence_intervals": {
            "median": median.tolist(),
            "lower_95": lower_95.tolist(),
            "upper_95": upper_95.tolist(),
        },
        "sample_paths": simulator.get_sample_paths(100).tolist(),
    }
