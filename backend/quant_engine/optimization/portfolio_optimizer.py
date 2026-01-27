# backend/quant_engine/optimization/portfolio_optimizer.py
"""
Portfolio Optimizer - Main orchestration module

Provides a unified interface for various portfolio optimization methods including:
- Mean-Variance Optimization (Markowitz)
- Black-Litterman Model
- Risk Parity
- Minimum Variance
- Maximum Sharpe Ratio
- Hierarchical Risk Parity (HRP)
- Genetic Algorithm
- Constrained optimization with various objectives

This module serves as the main entry point for portfolio optimization in the system.
"""

from typing import Any

import numpy as np
import pandas as pd
from loguru import logger
from scipy.optimize import minimize

from backend.quant_engine.optimization.black_litterman import (
    black_litterman_optimization,
)
from backend.quant_engine.optimization.genetic_algorithm import (
    genetic_portfolio_optimization,
)


class PortfolioOptimizer:
    """
    Main portfolio optimization class

    Supports multiple optimization methods and constraints.
    Provides a clean API for portfolio construction.
    """

    def __init__(
        self,
        returns_data: pd.DataFrame,
        risk_free_rate: float = 0.02,
    ):
        """
        Initialize portfolio optimizer

        Args:
            returns_data: DataFrame with asset returns (columns = assets, rows = time periods)
            risk_free_rate: Risk-free rate (annualized)
        """
        self.returns_data = returns_data
        self.asset_names = returns_data.columns.tolist()
        self.n_assets = len(self.asset_names)
        self.risk_free_rate = risk_free_rate

        # Calculate expected returns and covariance
        self.mean_returns = returns_data.mean() * 252  # Annualized
        self.cov_matrix = returns_data.cov() * 252  # Annualized

        # Store optimization results
        self.last_weights: dict[str, float] | None = None
        self.last_metrics: dict[str, Any] | None = None

        logger.info(
            f"Portfolio Optimizer initialized: {self.n_assets} assets, RF={risk_free_rate:.4f}"
        )

    def optimize(
        self, method: str = "max_sharpe", constraints: dict[str, Any] | None = None, **kwargs
    ) -> tuple[dict[str, float], dict[str, Any]]:
        """
        Optimize portfolio using specified method

        Args:
            method: Optimization method
                - 'max_sharpe': Maximum Sharpe ratio
                - 'min_volatility': Minimum volatility
                - 'risk_parity': Equal risk contribution
                - 'black_litterman': Black-Litterman with views
                - 'hrp': Hierarchical Risk Parity
                - 'genetic': Genetic algorithm
                - 'efficient_frontier': Generate efficient frontier
            constraints: Dictionary of constraints
                - 'min_weight': Minimum weight per asset (default: 0.0)
                - 'max_weight': Maximum weight per asset (default: 1.0)
                - 'target_return': Target portfolio return
                - 'target_volatility': Target portfolio volatility
                - 'sector_limits': Dict of sector limits
            **kwargs: Method-specific parameters

        Returns:
            Tuple of (weights_dict, metrics_dict)

        Example:
            >>> optimizer = PortfolioOptimizer(returns_df)
            >>> weights, metrics = optimizer.optimize(
            ...     method='max_sharpe',
            ...     constraints={'min_weight': 0.05, 'max_weight': 0.30}
            ... )
        """
        # Parse constraints
        constraints = constraints or {}
        min_weight = constraints.get("min_weight", 0.0)
        max_weight = constraints.get("max_weight", 1.0)
        target_return = constraints.get("target_return")
        target_volatility = constraints.get("target_volatility")

        logger.info(f"Optimizing portfolio using method: {method}")

        # Route to appropriate optimization method
        if method == "max_sharpe":
            weights, metrics = self._optimize_max_sharpe(min_weight, max_weight)

        elif method == "min_volatility":
            weights, metrics = self._optimize_min_volatility(min_weight, max_weight)

        elif method == "risk_parity":
            weights, metrics = self._optimize_risk_parity(min_weight, max_weight)

        elif method == "black_litterman":
            views = kwargs.get("views", {})
            market_caps = kwargs.get("market_caps")
            tau = kwargs.get("tau", 0.025)
            weights, metrics = self._optimize_black_litterman(
                views, market_caps, tau, min_weight, max_weight
            )

        elif method == "hrp":
            weights, metrics = self._optimize_hrp(min_weight, max_weight)

        elif method == "genetic":
            objective = kwargs.get("objective", "sharpe")
            population_size = kwargs.get("population_size", 100)
            n_generations = kwargs.get("n_generations", 100)
            weights, metrics = self._optimize_genetic(
                objective, min_weight, max_weight, population_size, n_generations
            )

        elif method == "efficient_frontier":
            n_points = kwargs.get("n_points", 50)
            return self._generate_efficient_frontier(n_points, min_weight, max_weight)

        elif method == "target_return":
            if target_return is None:
                raise ValueError("target_return must be specified for this method")
            weights, metrics = self._optimize_target_return(target_return, min_weight, max_weight)

        elif method == "target_volatility":
            if target_volatility is None:
                raise ValueError("target_volatility must be specified for this method")
            weights, metrics = self._optimize_target_volatility(
                target_volatility, min_weight, max_weight
            )

        else:
            raise ValueError(f"Unknown optimization method: {method}")

        # Store results
        self.last_weights = weights
        self.last_metrics = metrics

        logger.success(
            f"Optimization complete: Return={metrics.get('expected_return', 0):.4f}, "
            f"Volatility={metrics.get('volatility', 0):.4f}, "
            f"Sharpe={metrics.get('sharpe_ratio', 0):.4f}"
        )

        return weights, metrics

    def _optimize_max_sharpe(
        self,
        min_weight: float,
        max_weight: float,
    ) -> tuple[dict[str, float], dict[str, Any]]:
        """Maximize Sharpe ratio"""

        def neg_sharpe(weights):
            port_return = weights @ self.mean_returns
            port_vol = np.sqrt(weights @ self.cov_matrix @ weights)
            return -(port_return - self.risk_free_rate) / port_vol

        weights_array = self._optimize_scipy(
            objective_func=neg_sharpe,
            min_weight=min_weight,
            max_weight=max_weight,
        )

        weights_dict, metrics = self._calculate_portfolio_metrics(weights_array)
        metrics["method"] = "max_sharpe"

        return weights_dict, metrics

    def _optimize_min_volatility(
        self,
        min_weight: float,
        max_weight: float,
    ) -> tuple[dict[str, float], dict[str, Any]]:
        """Minimize portfolio volatility"""

        def portfolio_volatility(weights):
            return np.sqrt(weights @ self.cov_matrix @ weights)

        weights_array = self._optimize_scipy(
            objective_func=portfolio_volatility,
            min_weight=min_weight,
            max_weight=max_weight,
        )

        weights_dict, metrics = self._calculate_portfolio_metrics(weights_array)
        metrics["method"] = "min_volatility"

        return weights_dict, metrics

    def _optimize_risk_parity(
        self,
        min_weight: float,
        max_weight: float,
    ) -> tuple[dict[str, float], dict[str, Any]]:
        """Risk parity - equal risk contribution from each asset"""

        def risk_parity_objective(weights):
            # Portfolio volatility
            port_vol = np.sqrt(weights @ self.cov_matrix @ weights)

            # Marginal risk contribution
            marginal_contrib = (self.cov_matrix @ weights) / port_vol

            # Risk contribution of each asset
            risk_contrib = weights * marginal_contrib

            # Target: equal risk from each asset
            target_risk = port_vol / self.n_assets

            # Minimize sum of squared deviations
            return np.sum((risk_contrib - target_risk) ** 2)

        weights_array = self._optimize_scipy(
            objective_func=risk_parity_objective,
            min_weight=min_weight,
            max_weight=max_weight,
        )

        weights_dict, metrics = self._calculate_portfolio_metrics(weights_array)
        metrics["method"] = "risk_parity"

        return weights_dict, metrics

    def _optimize_black_litterman(
        self,
        views: dict[str, dict[str, Any]],
        market_caps: dict[str, float] | None,
        tau: float,
        min_weight: float,
        max_weight: float,
    ) -> tuple[dict[str, float], dict[str, Any]]:
        """Black-Litterman optimization with investor views"""

        weights_dict, metrics = black_litterman_optimization(
            returns_df=self.returns_data,
            market_caps=market_caps,
            views=views,
            risk_free_rate=self.risk_free_rate,
            tau=tau,
            min_weight=min_weight,
            max_weight=max_weight,
        )

        metrics["method"] = "black_litterman"
        return weights_dict, metrics

    def _optimize_hrp(
        self,
        min_weight: float,
        max_weight: float,
    ) -> tuple[dict[str, float], dict[str, Any]]:
        """Hierarchical Risk Parity"""
        from scipy.cluster.hierarchy import dendrogram, linkage
        from scipy.spatial.distance import squareform

        # Calculate correlation matrix
        corr_matrix = self.returns_data.corr()

        # Convert to distance matrix
        dist_matrix = np.sqrt(0.5 * (1 - corr_matrix))

        # Hierarchical clustering
        dist_condensed = squareform(dist_matrix, checks=False)
        linkage_matrix = linkage(dist_condensed, method="single")

        # Quasi-diagonalization
        sort_idx = self._quasi_diag(linkage_matrix)
        sorted_returns = self.returns_data.iloc[:, sort_idx]

        # Recursive bisection
        weights_array = self._recursive_bisection(sorted_returns.values)

        # Unsort weights
        weights_unsorted = np.zeros(self.n_assets)
        weights_unsorted[sort_idx] = weights_array

        # Apply constraints
        weights_unsorted = np.clip(weights_unsorted, min_weight, max_weight)
        weights_unsorted = weights_unsorted / weights_unsorted.sum()

        weights_dict, metrics = self._calculate_portfolio_metrics(weights_unsorted)
        metrics["method"] = "hrp"

        return weights_dict, metrics

    def _optimize_genetic(
        self,
        objective: str,
        min_weight: float,
        max_weight: float,
        population_size: int,
        n_generations: int,
    ) -> tuple[dict[str, float], dict[str, Any]]:
        """Genetic algorithm optimization"""

        weights_array, ga_metrics = genetic_portfolio_optimization(
            expected_returns=self.mean_returns.values,
            cov_matrix=self.cov_matrix.values,
            objective=objective,
            risk_free_rate=self.risk_free_rate,
            min_weight=min_weight,
            max_weight=max_weight,
            population_size=population_size,
            n_generations=n_generations,
        )

        weights_dict, metrics = self._calculate_portfolio_metrics(weights_array)
        metrics.update(ga_metrics)
        metrics["method"] = "genetic"

        return weights_dict, metrics

    def _optimize_target_return(
        self,
        target_return: float,
        min_weight: float,
        max_weight: float,
    ) -> tuple[dict[str, float], dict[str, Any]]:
        """Minimize volatility given target return"""

        def portfolio_volatility(weights):
            return np.sqrt(weights @ self.cov_matrix @ weights)

        # Add return constraint
        return_constraint = {"type": "eq", "fun": lambda w: w @ self.mean_returns - target_return}

        weights_array = self._optimize_scipy(
            objective_func=portfolio_volatility,
            min_weight=min_weight,
            max_weight=max_weight,
            additional_constraints=[return_constraint],
        )

        weights_dict, metrics = self._calculate_portfolio_metrics(weights_array)
        metrics["method"] = "target_return"
        metrics["target_return"] = target_return

        return weights_dict, metrics

    def _optimize_target_volatility(
        self,
        target_volatility: float,
        min_weight: float,
        max_weight: float,
    ) -> tuple[dict[str, float], dict[str, Any]]:
        """Maximize return given target volatility"""

        def neg_return(weights):
            return -(weights @ self.mean_returns)

        # Add volatility constraint
        vol_constraint = {
            "type": "eq",
            "fun": lambda w: np.sqrt(w @ self.cov_matrix @ w) - target_volatility,
        }

        weights_array = self._optimize_scipy(
            objective_func=neg_return,
            min_weight=min_weight,
            max_weight=max_weight,
            additional_constraints=[vol_constraint],
        )

        weights_dict, metrics = self._calculate_portfolio_metrics(weights_array)
        metrics["method"] = "target_volatility"
        metrics["target_volatility"] = target_volatility

        return weights_dict, metrics

    def _generate_efficient_frontier(
        self,
        n_points: int,
        min_weight: float,
        max_weight: float,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Generate efficient frontier points"""

        # Range of target returns
        min_return = self.mean_returns.min()
        max_return = self.mean_returns.max()
        target_returns = np.linspace(min_return, max_return, n_points)

        frontier_portfolios = []

        for target_return in target_returns:
            try:
                weights_dict, metrics = self._optimize_target_return(
                    target_return, min_weight, max_weight
                )

                frontier_portfolios.append(
                    {
                        "weights": weights_dict,
                        "return": metrics["expected_return"],
                        "volatility": metrics["volatility"],
                        "sharpe": metrics["sharpe_ratio"],
                    }
                )
            except Exception as e:
                logger.warning(f"Could not optimize for return {target_return:.4f}: {e}")
                continue

        # Find max Sharpe point
        if frontier_portfolios:
            max_sharpe_portfolio = max(frontier_portfolios, key=lambda x: x["sharpe"])
        else:
            max_sharpe_portfolio = {}

        summary = {
            "n_points": len(frontier_portfolios),
            "max_sharpe_portfolio": max_sharpe_portfolio,
            "method": "efficient_frontier",
        }

        return frontier_portfolios, summary

    def _optimize_scipy(
        self,
        objective_func: callable,
        min_weight: float,
        max_weight: float,
        additional_constraints: list | None = None,
    ) -> np.ndarray:
        """
        Generic scipy optimization wrapper

        Args:
            objective_func: Function to minimize
            min_weight: Minimum weight per asset
            max_weight: Maximum weight per asset
            additional_constraints: Additional constraint dictionaries

        Returns:
            Optimal weights array
        """
        # Base constraint: weights sum to 1
        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

        # Add additional constraints
        if additional_constraints:
            constraints.extend(additional_constraints)

        # Bounds
        bounds = tuple((min_weight, max_weight) for _ in range(self.n_assets))

        # Initial guess - equal weight
        w0 = np.ones(self.n_assets) / self.n_assets

        # Optimize
        result = minimize(
            objective_func,
            w0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 1000, "ftol": 1e-9},
        )

        if not result.success:
            logger.warning(f"Optimization warning: {result.message}")

        return result.x

    def _calculate_portfolio_metrics(
        self,
        weights: np.ndarray,
    ) -> tuple[dict[str, float], dict[str, Any]]:
        """
        Calculate portfolio performance metrics

        Args:
            weights: Portfolio weights

        Returns:
            Tuple of (weights_dict, metrics_dict)
        """
        # Portfolio return and risk
        portfolio_return = weights @ self.mean_returns
        portfolio_vol = np.sqrt(weights @ self.cov_matrix @ weights)
        sharpe = (portfolio_return - self.risk_free_rate) / portfolio_vol

        # Diversification metrics
        individual_vols = np.sqrt(np.diag(self.cov_matrix))
        weighted_vol = weights @ individual_vols
        diversification_ratio = weighted_vol / portfolio_vol

        # Effective number of assets (ENB)
        enb = 1 / np.sum(weights**2)

        # Concentration (HHI)
        hhi = np.sum(weights**2)

        # Create weights dictionary
        weights_dict = dict(zip(self.asset_names, weights))

        # Create metrics dictionary
        metrics = {
            "expected_return": float(portfolio_return),
            "volatility": float(portfolio_vol),
            "sharpe_ratio": float(sharpe),
            "diversification_ratio": float(diversification_ratio),
            "effective_n_assets": float(enb),
            "concentration_hhi": float(hhi),
            "max_weight": float(np.max(weights)),
            "min_weight": float(np.min(weights)),
        }

        return weights_dict, metrics

    def _quasi_diag(self, linkage_matrix: np.ndarray) -> np.ndarray:
        """Quasi-diagonalization for HRP"""
        sort_idx = []
        n = linkage_matrix.shape[0] + 1

        def _quasi_diag_recursive(cluster_idx):
            if cluster_idx < n:
                sort_idx.append(cluster_idx)
            else:
                left_idx = int(linkage_matrix[cluster_idx - n, 0])
                right_idx = int(linkage_matrix[cluster_idx - n, 1])
                _quasi_diag_recursive(left_idx)
                _quasi_diag_recursive(right_idx)

        _quasi_diag_recursive(2 * n - 2)
        return np.array(sort_idx)

    def _recursive_bisection(self, returns: np.ndarray) -> np.ndarray:
        """Recursive bisection for HRP weights"""
        cov = np.cov(returns, rowvar=False)
        weights = np.ones(returns.shape[1]) / returns.shape[1]

        def _get_cluster_var(cov_matrix, cluster_items):
            sub_cov = cov_matrix[np.ix_(cluster_items, cluster_items)]
            inv_var = 1.0 / np.diag(sub_cov)
            w = inv_var / inv_var.sum()
            return w @ sub_cov @ w

        def _recursive_split(items):
            if len(items) == 1:
                return

            # Split cluster
            mid = len(items) // 2
            left = items[:mid]
            right = items[mid:]

            # Calculate cluster variances
            var_left = _get_cluster_var(cov, left)
            var_right = _get_cluster_var(cov, right)

            # Allocate weights inversely proportional to variance
            alpha = 1 - var_left / (var_left + var_right)

            weights[left] *= alpha
            weights[right] *= 1 - alpha

            # Recurse
            _recursive_split(left)
            _recursive_split(right)

        items = list(range(returns.shape[1]))
        _recursive_split(items)

        return weights


def optimize_portfolio(
    returns_df: pd.DataFrame, method: str = "max_sharpe", risk_free_rate: float = 0.02, **kwargs
) -> tuple[dict[str, float], dict[str, Any]]:
    """
    Convenience function for portfolio optimization

    Args:
        returns_df: DataFrame of asset returns
        method: Optimization method
        risk_free_rate: Risk-free rate
        **kwargs: Additional optimization parameters

    Returns:
        Tuple of (weights_dict, metrics_dict)

    Example:
        >>> weights, metrics = optimize_portfolio(
        ...     returns_df,
        ...     method='max_sharpe',
        ...     constraints={'min_weight': 0.05, 'max_weight': 0.30}
        ... )
    """
    optimizer = PortfolioOptimizer(returns_df, risk_free_rate)
    return optimizer.optimize(method=method, **kwargs)
