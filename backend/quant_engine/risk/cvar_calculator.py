# backend/quant_engine/risk/cvar_calculator.py
"""
Conditional Value at Risk (CVaR) Calculator

Also known as Expected Shortfall (ES), CVaR measures the expected loss
in the worst α% of cases. It's a coherent risk measure that addresses
some limitations of VaR.

Methods:
- Historical CVaR
- Parametric CVaR (Normal distribution)
- Modified CVaR (Cornish-Fisher expansion)
- Monte Carlo CVaR
- Kernel Density Estimation CVaR

CVaR is superior to VaR because:
- It considers the tail beyond VaR
- It's a coherent risk measure (sub-additive)
- Useful for portfolio optimization under CVaR constraint

References:
- Rockafellar, R. T., & Uryasev, S. (2000). Optimization of conditional value-at-risk.
- Acerbi, C., & Tasche, D. (2002). On the coherence of expected shortfall.
"""

from typing import Any

import numpy as np
import pandas as pd
import polars as pl
from loguru import logger
from scipy import stats
from scipy.optimize import minimize


class CVaRCalculator:
    """
    Calculate Conditional Value at Risk (Expected Shortfall)

    CVaR represents the expected loss given that the loss exceeds VaR.
    It provides a more complete picture of tail risk than VaR alone.
    """

    def __init__(
        self,
        confidence_level: float = 0.95,
        method: str = "historical",
    ):
        """
        Initialize CVaR calculator

        Args:
            confidence_level: Confidence level (e.g., 0.95 for 95%)
            method: Calculation method
                - 'historical': Historical simulation
                - 'parametric': Assumes normal distribution
                - 'modified': Cornish-Fisher expansion for fat tails
                - 'monte_carlo': Monte Carlo simulation
                - 'kde': Kernel Density Estimation
        """
        self.confidence_level = confidence_level
        self.method = method
        self.alpha = 1 - confidence_level

        logger.info(f"CVaR calculator initialized: {method}, α={self.alpha:.4f}")

    def calculate_historical_cvar(
        self,
        returns: np.ndarray,
        weights: np.ndarray | None = None,
    ) -> dict[str, float]:
        """
        Calculate CVaR using historical simulation

        Args:
            returns: Return series or matrix (assets x time)
            weights: Portfolio weights (if returns is a matrix)

        Returns:
            Dictionary with CVaR and related metrics
        """
        # Calculate portfolio returns if weights provided
        if weights is not None and returns.ndim == 2:
            portfolio_returns = returns.T @ weights
        else:
            portfolio_returns = returns

        # Sort returns
        sorted_returns = np.sort(portfolio_returns)

        # Find VaR cutoff
        var_idx = int(np.floor(len(sorted_returns) * self.alpha))
        var = -sorted_returns[var_idx]

        # CVaR is the average of returns beyond VaR
        tail_returns = sorted_returns[:var_idx]
        cvar = -np.mean(tail_returns) if len(tail_returns) > 0 else var

        # Additional tail statistics
        tail_std = np.std(tail_returns) if len(tail_returns) > 0 else 0
        worst_case = -sorted_returns[0]

        logger.debug(f"Historical CVaR: {cvar:.4f} (VaR: {var:.4f})")

        return {
            "cvar": cvar,
            "var": var,
            "tail_std": tail_std,
            "worst_case": worst_case,
            "n_tail_observations": len(tail_returns),
        }

    def calculate_parametric_cvar(
        self,
        returns: np.ndarray,
        weights: np.ndarray | None = None,
    ) -> dict[str, float]:
        """
        Calculate CVaR assuming normal distribution

        For normal distribution:
        CVaR = μ - σ * φ(Φ^(-1)(α)) / α

        where φ is the PDF and Φ is the CDF of standard normal

        Args:
            returns: Return series or matrix
            weights: Portfolio weights

        Returns:
            Dictionary with CVaR metrics
        """
        # Calculate portfolio returns
        if weights is not None and returns.ndim == 2:
            portfolio_returns = returns.T @ weights
        else:
            portfolio_returns = returns

        # Calculate moments
        mu = np.mean(portfolio_returns)
        sigma = np.std(portfolio_returns)

        # VaR calculation
        z_alpha = stats.norm.ppf(self.alpha)
        var = -(mu + sigma * z_alpha)

        # CVaR calculation
        # Expected value in tail = μ - σ * φ(z_α) / α
        phi_z_alpha = stats.norm.pdf(z_alpha)
        cvar = -(mu - sigma * phi_z_alpha / self.alpha)

        logger.debug(f"Parametric CVaR: {cvar:.4f} (assuming normality)")

        return {
            "cvar": cvar,
            "var": var,
            "mean": mu,
            "std": sigma,
        }

    def calculate_modified_cvar(
        self,
        returns: np.ndarray,
        weights: np.ndarray | None = None,
    ) -> dict[str, float]:
        """
        Calculate modified CVaR using Cornish-Fisher expansion

        Adjusts for skewness and kurtosis in the return distribution

        Args:
            returns: Return series or matrix
            weights: Portfolio weights

        Returns:
            Dictionary with CVaR metrics
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

        # Cornish-Fisher VaR
        z_alpha = stats.norm.ppf(self.alpha)
        z_cf = (
            z_alpha
            + (z_alpha**2 - 1) * skew / 6
            + (z_alpha**3 - 3 * z_alpha) * kurt / 24
            - (2 * z_alpha**3 - 5 * z_alpha) * skew**2 / 36
        )

        var = -(mu + sigma * z_cf)

        # Modified CVaR approximation
        # Use modified VaR and adjust based on tail shape
        phi_z_alpha = stats.norm.pdf(z_alpha)
        cvar_adjustment = 1 + skew * z_alpha + kurt * (z_alpha**2 - 1) / 4
        cvar = -(mu - sigma * phi_z_alpha / self.alpha * cvar_adjustment)

        logger.debug(f"Modified CVaR: {cvar:.4f} (skew={skew:.3f}, kurt={kurt:.3f})")

        return {
            "cvar": cvar,
            "var": var,
            "mean": mu,
            "std": sigma,
            "skewness": skew,
            "kurtosis": kurt,
        }

    def calculate_monte_carlo_cvar(
        self,
        returns: np.ndarray,
        weights: np.ndarray | None = None,
        n_simulations: int = 10000,
        random_state: int | None = None,
    ) -> dict[str, float]:
        """
        Calculate CVaR using Monte Carlo simulation

        Args:
            returns: Return series or matrix
            weights: Portfolio weights
            n_simulations: Number of Monte Carlo simulations
            random_state: Random seed

        Returns:
            Dictionary with CVaR metrics
        """
        if random_state is not None:
            np.random.seed(random_state)

        # Calculate portfolio returns
        if weights is not None and returns.ndim == 2:
            portfolio_returns = returns.T @ weights
        else:
            portfolio_returns = returns

        # Estimate distribution parameters
        mu = np.mean(portfolio_returns)
        sigma = np.std(portfolio_returns)

        # Generate simulated returns
        simulated_returns = np.random.normal(mu, sigma, n_simulations)

        # Calculate CVaR from simulations
        sorted_sim = np.sort(simulated_returns)
        var_idx = int(np.floor(n_simulations * self.alpha))
        var = -sorted_sim[var_idx]

        tail_returns = sorted_sim[:var_idx]
        cvar = -np.mean(tail_returns)

        logger.debug(f"Monte Carlo CVaR: {cvar:.4f} ({n_simulations} simulations)")

        return {
            "cvar": cvar,
            "var": var,
            "n_simulations": n_simulations,
            "tail_mean": -np.mean(tail_returns),
            "tail_std": np.std(tail_returns),
        }

    def calculate_kde_cvar(
        self,
        returns: np.ndarray,
        weights: np.ndarray | None = None,
        bandwidth: float | None = None,
    ) -> dict[str, float]:
        """
        Calculate CVaR using Kernel Density Estimation

        Args:
            returns: Return series or matrix
            weights: Portfolio weights
            bandwidth: KDE bandwidth (if None, uses Scott's rule)

        Returns:
            Dictionary with CVaR metrics
        """
        from scipy.stats import gaussian_kde

        # Calculate portfolio returns
        if weights is not None and returns.ndim == 2:
            portfolio_returns = returns.T @ weights
        else:
            portfolio_returns = returns

        # Fit KDE
        if bandwidth is None:
            kde = gaussian_kde(portfolio_returns)
        else:
            kde = gaussian_kde(portfolio_returns, bw_method=bandwidth)

        # Generate fine grid for evaluation
        x_grid = np.linspace(portfolio_returns.min(), portfolio_returns.max(), 1000)
        pdf_values = kde(x_grid)

        # Calculate CDF
        dx = x_grid[1] - x_grid[0]
        cdf_values = np.cumsum(pdf_values) * dx

        # Find VaR
        var_idx = np.searchsorted(cdf_values, self.alpha)
        var = -x_grid[var_idx]

        # Calculate CVaR (expected value in tail)
        tail_mask = x_grid <= x_grid[var_idx]
        tail_x = x_grid[tail_mask]
        tail_pdf = pdf_values[tail_mask]

        # Normalize tail PDF
        tail_pdf_normalized = tail_pdf / (np.sum(tail_pdf) * dx)
        cvar = -np.sum(tail_x * tail_pdf_normalized * dx)

        logger.debug(f"KDE CVaR: {cvar:.4f}")

        return {
            "cvar": cvar,
            "var": var,
            "kde_bandwidth": kde.factor,
        }

    def calculate(
        self,
        returns: np.ndarray | pd.Series | pl.Series,
        weights: np.ndarray | None = None,
        **kwargs,
    ) -> dict[str, float]:
        """
        Calculate CVaR using the configured method

        Args:
            returns: Return data (numpy array, pandas Series, or polars Series)
            weights: Portfolio weights (optional)
            **kwargs: Method-specific parameters

        Returns:
            Dictionary with CVaR and related metrics
        """
        # Convert to numpy if needed
        if isinstance(returns, (pd.Series, pl.Series)):
            returns = returns.to_numpy()

        # Route to appropriate method
        if self.method == "historical":
            return self.calculate_historical_cvar(returns, weights)

        elif self.method == "parametric":
            return self.calculate_parametric_cvar(returns, weights)

        elif self.method == "modified":
            return self.calculate_modified_cvar(returns, weights)

        elif self.method == "monte_carlo":
            n_sims = kwargs.get("n_simulations", 10000)
            random_state = kwargs.get("random_state", None)
            return self.calculate_monte_carlo_cvar(returns, weights, n_sims, random_state)

        elif self.method == "kde":
            bandwidth = kwargs.get("bandwidth", None)
            return self.calculate_kde_cvar(returns, weights, bandwidth)

        else:
            raise ValueError(f"Unknown method: {self.method}")

    def calculate_portfolio_cvar(
        self,
        returns_matrix: np.ndarray,
        weights: np.ndarray,
    ) -> dict[str, float]:
        """
        Calculate CVaR for a portfolio

        Args:
            returns_matrix: Asset returns matrix (n_assets x n_periods)
            weights: Portfolio weights (n_assets,)

        Returns:
            Dictionary with portfolio CVaR metrics
        """
        return self.calculate(returns_matrix, weights)

    def calculate_marginal_cvar(
        self,
        returns_matrix: np.ndarray,
        weights: np.ndarray,
        delta: float = 0.01,
    ) -> np.ndarray:
        """
        Calculate marginal CVaR contribution of each asset

        Marginal CVaR = ∂CVaR/∂w_i

        Args:
            returns_matrix: Asset returns matrix (n_assets x n_periods)
            weights: Current portfolio weights
            delta: Finite difference step size

        Returns:
            Array of marginal CVaR values
        """
        n_assets = len(weights)
        base_cvar = self.calculate(returns_matrix, weights)["cvar"]

        marginal_cvars = np.zeros(n_assets)

        for i in range(n_assets):
            # Perturb weight
            perturbed_weights = weights.copy()
            perturbed_weights[i] += delta
            perturbed_weights = perturbed_weights / perturbed_weights.sum()  # Renormalize

            # Calculate CVaR with perturbed weights
            perturbed_cvar = self.calculate(returns_matrix, perturbed_weights)["cvar"]

            # Marginal CVaR
            marginal_cvars[i] = (perturbed_cvar - base_cvar) / delta

        logger.debug(f"Calculated marginal CVaR for {n_assets} assets")

        return marginal_cvars

    def calculate_component_cvar(
        self,
        returns_matrix: np.ndarray,
        weights: np.ndarray,
    ) -> dict[str, np.ndarray]:
        """
        Calculate component CVaR (CVaR contribution of each asset)

        Component CVaR_i = w_i * Marginal CVaR_i

        Args:
            returns_matrix: Asset returns matrix
            weights: Portfolio weights

        Returns:
            Dictionary with component CVaR and percentage contributions
        """
        marginal_cvars = self.calculate_marginal_cvar(returns_matrix, weights)
        component_cvars = weights * marginal_cvars

        portfolio_cvar = self.calculate(returns_matrix, weights)["cvar"]

        # Percentage contribution
        pct_contributions = (
            component_cvars / portfolio_cvar
            if portfolio_cvar != 0
            else np.zeros_like(component_cvars)
        )

        return {
            "component_cvar": component_cvars,
            "pct_contribution": pct_contributions,
            "marginal_cvar": marginal_cvars,
        }


def calculate_cvar(
    returns: np.ndarray | pd.Series | pl.Series,
    confidence_level: float = 0.95,
    method: str = "historical",
    weights: np.ndarray | None = None,
    **kwargs,
) -> dict[str, float]:
    """
    Convenience function to calculate CVaR

    Args:
        returns: Return data
        confidence_level: Confidence level (0 to 1)
        method: Calculation method
        weights: Portfolio weights (if returns is a matrix)
        **kwargs: Method-specific parameters

    Returns:
        Dictionary with CVaR metrics

    Example:
        >>> returns = np.random.normal(0.001, 0.02, 1000)
        >>> cvar_metrics = calculate_cvar(returns, confidence_level=0.95, method='historical')
        >>> print(f"CVaR (95%): {cvar_metrics['cvar']:.4f}")

        >>> # Portfolio CVaR
        >>> returns_matrix = np.random.normal(0.001, 0.02, (3, 1000))  # 3 assets
        >>> weights = np.array([0.4, 0.4, 0.2])
        >>> portfolio_cvar = calculate_cvar(returns_matrix, weights=weights)
    """
    calculator = CVaRCalculator(confidence_level, method)
    return calculator.calculate(returns, weights, **kwargs)


def optimize_cvar_portfolio(
    returns_matrix: np.ndarray,
    target_return: float | None = None,
    min_weight: float = 0.0,
    max_weight: float = 1.0,
    confidence_level: float = 0.95,
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Optimize portfolio to minimize CVaR

    Args:
        returns_matrix: Asset returns matrix (n_assets x n_periods)
        target_return: Target portfolio return (optional)
        min_weight: Minimum weight per asset
        max_weight: Maximum weight per asset
        confidence_level: CVaR confidence level

    Returns:
        Tuple of (optimal_weights, metrics)

    Example:
        >>> returns_matrix = np.random.normal(0.001, 0.02, (5, 1000))
        >>> optimal_weights, metrics = optimize_cvar_portfolio(returns_matrix)
    """
    n_assets = returns_matrix.shape[0]

    # CVaR calculator
    calculator = CVaRCalculator(confidence_level, method="historical")

    # Objective: minimize CVaR
    def objective(weights):
        return calculator.calculate(returns_matrix, weights)["cvar"]

    # Constraints
    constraints = [
        {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}  # Weights sum to 1
    ]

    if target_return is not None:
        mean_returns = np.mean(returns_matrix, axis=1)
        constraints.append({"type": "eq", "fun": lambda w: w @ mean_returns - target_return})

    # Bounds
    bounds = tuple((min_weight, max_weight) for _ in range(n_assets))

    # Initial guess
    w0 = np.ones(n_assets) / n_assets

    # Optimize
    result = minimize(
        objective,
        w0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 1000},
    )

    if not result.success:
        logger.warning(f"CVaR optimization warning: {result.message}")

    optimal_weights = result.x

    # Calculate metrics
    portfolio_cvar = calculator.calculate(returns_matrix, optimal_weights)["cvar"]
    mean_returns = np.mean(returns_matrix, axis=1)
    portfolio_return = optimal_weights @ mean_returns

    metrics = {
        "cvar": portfolio_cvar,
        "expected_return": portfolio_return,
        "optimization_success": result.success,
    }

    logger.success(f"CVaR optimization complete: CVaR={portfolio_cvar:.4f}")

    return optimal_weights, metrics
