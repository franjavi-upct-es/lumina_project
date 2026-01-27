# backend/quant_engine/optimization/black_litterman.py
"""
Black-Litterman Portfolio Optimization Model

Combines market equilibrium returns with investor views to generate
expected returns for portfolio optimization. This implementation supports
both absolute and relative views with confidence levels.

References:
- Black, F., & Litterman, R. (1992). Global portfolio optimization.
- He, G., & Litterman, R. (1999). The intuition behind Black-Litterman model portfolios.
"""

from typing import Any

import numpy as np
import pandas as pd
from loguru import logger
from scipy.optimize import minimize


class BlackLittermanModel:
    """
    Black-Litterman portfolio optimization model

    The model combines market equilibrium (CAPM) with investor views to produce
    posterior expected returns that blend both sources of information.

    Key Features:
    - Absolute and relative views
    - Confidence levels for views
    - Market equilibrium from reverse optimization
    - Bayesian framework for combining information
    """

    def __init__(
        self,
        risk_free_rate: float = 0.02,
        market_price_of_risk: float = 0.3,
        tau: float = 0.025,
    ):
        """
        Initialize Black-Litterman model

        Args:
            risk_free_rate: Risk-free rate (annualized)
            market_price_of_risk: Market price of risk (Sharpe ratio)
            tau: Uncertainty in the prior estimate (typically 0.01-0.05)
        """
        self.risk_free_rate = risk_free_rate
        self.market_price_of_risk = market_price_of_risk
        self.tau = tau

        # Will be populated during optimization
        self.market_weights: np.ndarray | None = None
        self.cov_matrix: np.ndarray | None = None
        self.prior_returns: np.ndarray | None = None
        self.posterior_returns: np.ndarray | None = None
        self.posterior_cov: np.ndarray | None = None

        logger.debug(
            f"Black-Litterman initialized: rf={risk_free_rate}, "
            f"market_sharpe={market_price_of_risk}, tau={tau}"
        )

    def calculate_implied_returns(
        self,
        market_weights: np.ndarray,
        cov_matrix: np.ndarray,
        risk_aversion: float | None = None,
    ) -> np.ndarray:
        """
        Calculate implied equilibrium returns from market weights (reverse optimization)

        Pi = delta * Sigma * w_mkt

        Args:
            market_weights: Market capitalization weights
            cov_matrix: Covariance matrix of returns
            risk_aversion: Risk aversion coefficient (if None, calculated from Sharpe)

        Returns:
            Implied equilibrium returns
        """
        if risk_aversion is None:
            # Calculate from market Sharpe ratio
            market_var = market_weights.T @ cov_matrix @ market_weights
            market_vol = np.sqrt(market_var)
            risk_aversion = self.market_price_of_risk / market_vol

        # Reverse optimization: Pi = delta * Sigma * w
        implied_returns = risk_aversion * (cov_matrix @ market_weights)

        logger.debug(f"Calculated implied returns with risk aversion = {risk_aversion:.4f}")
        return implied_returns

    def build_view_matrix(
        self,
        views: dict[str, dict[str, Any]],
        asset_names: list[str],
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Build view matrix P and view vector Q from investor views

        Supports:
        - Absolute views: "Asset A will return 10%"
        - Relative views: "Asset A will outperform Asset B by 5%"

        Args:
            views: Dictionary of views with structure:
                {
                    'view_1': {
                        'type': 'absolute' or 'relative',
                        'assets': ['AAPL'] or ['AAPL', 'MSFT'],
                        'return': 0.10,  # Expected return or outperformance
                        'confidence': 0.5  # 0 to 1, higher = more confident
                    }
                }
            asset_names: List of asset names in order

        Returns:
            Tuple of (P matrix, Q vector)
        """
        n_assets = len(asset_names)
        n_views = len(views)

        P = np.zeros((n_views, n_assets))
        Q = np.zeros(n_views)

        for i, (view_name, view_data) in enumerate(views.items()):
            view_type = view_data.get("type", "absolute")
            assets = view_data["assets"]
            expected_return = view_data["return"]

            if view_type == "absolute":
                # Absolute view: P has 1 for the asset, 0 elsewhere
                asset_idx = asset_names.index(assets[0])
                P[i, asset_idx] = 1.0
                Q[i] = expected_return

            elif view_type == "relative":
                # Relative view: Asset A - Asset B = expected outperformance
                asset_a_idx = asset_names.index(assets[0])
                asset_b_idx = asset_names.index(assets[1])
                P[i, asset_a_idx] = 1.0
                P[i, asset_b_idx] = -1.0
                Q[i] = expected_return

            else:
                raise ValueError(f"Unknown view type: {view_type}")

        logger.info(f"Built view matrix with {n_views} views")
        return P, Q

    def calculate_view_uncertainty(
        self,
        P: np.ndarray,
        cov_matrix: np.ndarray,
        confidences: list[float],
    ) -> np.ndarray:
        """
        Calculate view uncertainty matrix Omega

        Higher confidence = lower uncertainty

        Args:
            P: View matrix
            cov_matrix: Asset covariance matrix
            confidences: List of confidence levels (0 to 1) for each view

        Returns:
            Omega matrix (diagonal)
        """
        n_views = P.shape[0]
        omega = np.zeros((n_views, n_views))

        for i, confidence in enumerate(confidences):
            # Uncertainty is inversely related to confidence
            # Scale by the variance of the view portfolio
            view_variance = P[i] @ cov_matrix @ P[i].T
            omega[i, i] = view_variance / confidence if confidence > 0 else view_variance * 1000

        return omega

    def calculate_posterior(
        self,
        prior_returns: np.ndarray,
        prior_cov: np.ndarray,
        P: np.ndarray,
        Q: np.ndarray,
        omega: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Calculate posterior returns and covariance using Bayesian update

        Posterior mean: E[R] = Pi + tau*Sigma*P'*[P*tau*Sigma*P' + Omega]^-1 * [Q - P*Pi]
        Posterior cov: Var[R] = (1+tau)*Sigma - tau*Sigma*P'*[P*tau*Sigma*P' + Omega]^-1*P*tau*Sigma

        Args:
            prior_returns: Prior (implied) returns
            prior_cov: Prior covariance matrix
            P: View matrix
            Q: View returns
            omega: View uncertainty matrix

        Returns:
            Tuple of (posterior_returns, posterior_covariance)
        """
        # Intermediate calculations
        tau_sigma = self.tau * prior_cov
        M = P @ tau_sigma @ P.T + omega
        M_inv = np.linalg.inv(M)

        # Posterior returns
        adjustment = tau_sigma @ P.T @ M_inv @ (Q - P @ prior_returns)
        posterior_returns = prior_returns + adjustment

        # Posterior covariance
        posterior_cov = prior_cov + tau_sigma - tau_sigma @ P.T @ M_inv @ P @ tau_sigma

        logger.info("Calculated posterior distribution")
        return posterior_returns, posterior_cov

    def optimize(
        self,
        asset_names: list[str],
        market_weights: np.ndarray,
        cov_matrix: np.ndarray,
        views: dict[str, dict[str, Any]] | None = None,
        min_weight: float = 0.0,
        max_weight: float = 1.0,
        target_return: float | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """
        Run Black-Litterman optimization

        Args:
            asset_names: List of asset names
            market_weights: Market capitalization weights
            cov_matrix: Covariance matrix
            views: Dictionary of investor views (optional)
            min_weight: Minimum weight per asset
            max_weight: Maximum weight per asset
            target_return: Target portfolio return (optional)

        Returns:
            Tuple of (optimal_weights, metrics_dict)
        """
        n_assets = len(asset_names)
        self.market_weights = market_weights
        self.cov_matrix = cov_matrix

        # Step 1: Calculate implied equilibrium returns
        self.prior_returns = self.calculate_implied_returns(market_weights, cov_matrix)
        logger.info(
            f"Prior returns range: [{self.prior_returns.min():.4f}, {self.prior_returns.max():.4f}]"
        )

        # Step 2: Incorporate views (if provided)
        if views and len(views) > 0:
            # Build view matrices
            P, Q = self.build_view_matrix(views, asset_names)
            confidences = [v.get("confidence", 0.5) for v in views.values()]
            omega = self.calculate_view_uncertainty(P, cov_matrix, confidences)

            # Calculate posterior
            self.posterior_returns, self.posterior_cov = self.calculate_posterior(
                self.prior_returns,
                cov_matrix,
                P,
                Q,
                omega,
            )

            logger.info(
                f"Posterior returns range: [{self.posterior_returns.min():.4f}, {self.posterior_returns.max():.4f}]"
            )
        else:
            # No views - use prior as posterior
            self.posterior_returns = self.prior_returns
            self.posterior_cov = cov_matrix
            logger.info("No views provided, using equilibrium returns")

        # Step 3: Portfolio optimization with posterior returns
        optimal_weights = self._optimize_weights(
            self.posterior_returns,
            self.posterior_cov,
            min_weight,
            max_weight,
            target_return,
        )

        # Calculate portfolio metrics
        portfolio_return = optimal_weights @ self.posterior_returns
        portfolio_variance = optimal_weights @ self.posterior_cov @ optimal_weights
        portfolio_vol = np.sqrt(portfolio_variance)
        sharpe = (portfolio_return - self.risk_free_rate) / portfolio_vol

        metrics = {
            "expected_return": float(portfolio_return),
            "volatility": float(portfolio_vol),
            "sharpe_ratio": float(sharpe),
            "prior_returns": self.prior_returns.tolist(),
            "posterior_returns": self.posterior_returns.tolist(),
            "n_views": len(views) if views else 0,
        }

        logger.success(
            f"Black-Litterman optimization complete: "
            f"Return={portfolio_return:.4f}, Vol={portfolio_vol:.4f}, Sharpe={sharpe:.4f}"
        )

        return optimal_weights, metrics

    def _optimize_weights(
        self,
        expected_returns: np.ndarray,
        cov_matrix: np.ndarray,
        min_weight: float,
        max_weight: float,
        target_return: float | None,
    ) -> np.ndarray:
        """
        Optimize portfolio weights given expected returns and covariance

        Maximizes Sharpe ratio or minimizes variance (if target return specified)

        Args:
            expected_returns: Expected returns vector
            cov_matrix: Covariance matrix
            min_weight: Minimum weight constraint
            max_weight: Maximum weight constraint
            target_return: Target return constraint (optional)

        Returns:
            Optimal weights
        """
        n_assets = len(expected_returns)

        # Objective function
        if target_return is None:
            # Maximize Sharpe ratio = minimize negative Sharpe
            def objective(w):
                port_return = w @ expected_returns
                port_vol = np.sqrt(w @ cov_matrix @ w)
                return -(port_return - self.risk_free_rate) / port_vol
        else:
            # Minimize variance given target return
            def objective(w):
                return w @ cov_matrix @ w

        # Constraints
        constraints = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}  # Weights sum to 1
        ]

        if target_return is not None:
            constraints.append(
                {"type": "eq", "fun": lambda w: w @ expected_returns - target_return}
            )

        # Bounds
        bounds = tuple((min_weight, max_weight) for _ in range(n_assets))

        # Initial guess - equal weight
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
            logger.warning(f"Optimization did not fully converge: {result.message}")

        return result.x


def black_litterman_optimization(
    returns_df: pd.DataFrame,
    market_caps: dict[str, float] | None = None,
    views: dict[str, dict[str, Any]] | None = None,
    risk_free_rate: float = 0.02,
    tau: float = 0.025,
    min_weight: float = 0.0,
    max_weight: float = 1.0,
) -> tuple[dict[str, float], dict[str, Any]]:
    """
    Convenience function for Black-Litterman optimization

    Args:
        returns_df: DataFrame of historical returns (columns = assets)
        market_caps: Market capitalization by asset (for equilibrium weights)
        views: Dictionary of investor views
        risk_free_rate: Risk-free rate
        tau: Prior uncertainty parameter
        min_weight: Minimum weight per asset
        max_weight: Maximum weight per asset

    Returns:
        Tuple of (weights_dict, metrics_dict)

    Example:
        >>> views = {
        ...     'tech_bullish': {
        ...         'type': 'absolute',
        ...         'assets': ['AAPL'],
        ...         'return': 0.15,
        ...         'confidence': 0.7
        ...     },
        ...     'outperformance': {
        ...         'type': 'relative',
        ...         'assets': ['AAPL', 'MSFT'],
        ...         'return': 0.05,
        ...         'confidence': 0.6
        ...     }
        ... }
        >>> weights, metrics = black_litterman_optimization(returns_df, views=views)
    """
    asset_names = returns_df.columns.tolist()
    n_assets = len(asset_names)

    # Calculate covariance matrix
    cov_matrix = returns_df.cov().values * 252  # Annualized

    # Market weights - use market cap if provided, else equal weight
    if market_caps:
        total_cap = sum(market_caps.values())
        market_weights = np.array([market_caps.get(asset, 0) / total_cap for asset in asset_names])
    else:
        market_weights = np.ones(n_assets) / n_assets
        logger.warning("No market caps provided, using equal weights as market proxy")

    # Run optimization
    bl_model = BlackLittermanModel(
        risk_free_rate=risk_free_rate,
        tau=tau,
    )

    optimal_weights, metrics = bl_model.optimize(
        asset_names=asset_names,
        market_weights=market_weights,
        cov_matrix=cov_matrix,
        views=views,
        min_weight=min_weight,
        max_weight=max_weight,
    )

    # Convert to dictionary
    weights_dict = dict(zip(asset_names, optimal_weights))

    return weights_dict, metrics
