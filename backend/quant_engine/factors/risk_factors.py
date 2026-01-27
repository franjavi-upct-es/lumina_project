# backend/quant_engine/factors/risk_factors.py
"""
Custom Risk Factor Analysis Module for Lumina 2.0

This module provides comprehensive risk factor construction and analysis,
including custom factor definitions, risk decomposition, and factor exposure
management for quantitative portfolio analysis.

Key Features:
- Custom risk factor construction from market data
- Standard risk factors (volatility, momentum, quality, etc.)
- Multi-factor risk decomposition
- Factor covariance estimation
- Risk attribution and contribution analysis
- Factor mimicking portfolio construction
- Stress testing with factor shocks

Dependencies:
- numpy: Numerical computations
- polars: DataFrame operations
- scipy: Statistical functions
- statsmodels: Regression analysis
- scikit-learn: Machine learning utilities

Author: Lumina Quant Lab
Version: 2.0.0
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

import numpy as np
import polars as pl
from loguru import logger

try:
    from scipy.optimize import minimize

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logger.warning("scipy not available - some risk factor features will be limited")

try:
    import statsmodels.api as sm

    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    logger.warning("statsmodels not available - regression features will be limited")

try:
    from sklearn.covariance import LedoitWolf, ShrunkCovariance

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("sklearn not available - some covariance features will be limited")


# =============================================================================
# ENUMERATIONS
# =============================================================================


class RiskFactorType(str, Enum):
    """Types of risk factors available in the system"""

    # Market factors
    MARKET_BETA = "market_beta"
    MARKET_VOLATILITY = "market_volatility"

    # Style factors
    MOMENTUM = "momentum"
    REVERSAL = "reversal"
    VOLATILITY = "volatility"
    SIZE = "size"
    VALUE = "value"
    QUALITY = "quality"
    GROWTH = "growth"
    DIVIDEND_YIELD = "dividend_yield"

    # Technical factors
    LIQUIDITY = "liquidity"
    TURNOVER = "turnover"
    PRICE_LEVEL = "price_level"

    # Risk factors
    BETA = "beta"
    IDIOSYNCRATIC_VOL = "idiosyncratic_vol"
    DOWNSIDE_BETA = "downside_beta"
    TAIL_RISK = "tail_risk"

    # Macro factors
    INTEREST_RATE = "interest_rate"
    CREDIT_SPREAD = "credit_spread"
    INFLATION = "inflation"

    # Custom factor
    CUSTOM = "custom"


class CovarianceMethod(str, Enum):
    """Methods for covariance estimation"""

    SAMPLE = "sample"
    LEDOIT_WOLF = "ledoit_wolf"
    SHRUNK = "shrunk"
    EXPONENTIAL = "exponential"
    FACTOR_MODEL = "factor_model"


class RiskMeasure(str, Enum):
    """Risk measurement methods"""

    VOLATILITY = "volatility"
    VAR = "var"
    CVAR = "cvar"
    DRAWDOWN = "drawdown"
    BETA = "beta"
    TRACKING_ERROR = "tracking_error"


# =============================================================================
# DATA CLASSES
# =============================================================================


@dataclass
class RiskFactor:
    """
    Represents a single risk factor with its definition and data.

    Attributes:
        name: Factor identifier
        factor_type: Type of risk factor
        values: Factor values as time series (Polars Series)
        description: Human-readable description
        calculation_params: Parameters used to calculate the factor
        metadata: Additional metadata
    """

    name: str
    factor_type: RiskFactorType
    values: pl.Series
    description: str = ""
    calculation_params: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def mean(self) -> float:
        """Average factor value"""
        return float(self.values.mean())

    @property
    def std(self) -> float:
        """Standard deviation of factor values"""
        return float(self.values.std())

    @property
    def sharpe(self) -> float:
        """Sharpe ratio of factor (assuming zero risk-free rate)"""
        if self.std == 0:
            return 0.0
        return self.mean / self.std * np.sqrt(252)


@dataclass
class FactorExposure:
    """
    Factor exposure results for an asset or portfolio.

    Attributes:
        asset_id: Asset identifier
        exposures: Dictionary of factor name to exposure value
        r_squared: Model R-squared
        adj_r_squared: Adjusted R-squared
        residual_std: Standard deviation of residuals
        t_stats: T-statistics for each exposure
        p_values: P-values for each exposure
    """

    asset_id: str
    exposures: dict[str, float]
    r_squared: float
    adj_r_squared: float
    residual_std: float
    t_stats: dict[str, float] = field(default_factory=dict)
    p_values: dict[str, float] = field(default_factory=dict)


@dataclass
class RiskDecomposition:
    """
    Risk decomposition results.

    Attributes:
        total_risk: Total portfolio risk
        factor_risk: Risk attributable to factor exposures
        idiosyncratic_risk: Residual/specific risk
        factor_contributions: Risk contribution by factor
        marginal_contributions: Marginal risk contribution by asset
        percentage_contributions: Percentage contribution by factor
    """

    total_risk: float
    factor_risk: float
    idiosyncratic_risk: float
    factor_contributions: dict[str, float]
    marginal_contributions: dict[str, float] = field(default_factory=dict)
    percentage_contributions: dict[str, float] = field(default_factory=dict)


@dataclass
class FactorMimickingPortfolio:
    """
    Factor mimicking portfolio construction results.

    Attributes:
        factor_name: Name of the target factor
        weights: Asset weights in the mimicking portfolio
        expected_return: Expected factor premium
        tracking_error: Tracking error vs. pure factor
        information_ratio: Information ratio of mimicking portfolio
        correlation: Correlation with target factor
    """

    factor_name: str
    weights: dict[str, float]
    expected_return: float
    tracking_error: float
    information_ratio: float
    correlation: float


# =============================================================================
# RISK FACTOR CALCULATOR CLASS
# =============================================================================


class RiskFactorCalculator:
    """
    Calculator for constructing risk factors from market data.

    This class provides methods to calculate various standard risk factors
    from price and fundamental data.

    Example:
        >>> calculator = RiskFactorCalculator()
        >>> prices = pl.DataFrame({...})
        >>> momentum = calculator.calculate_momentum(prices, window=252)
        >>> volatility = calculator.calculate_volatility(prices, window=20)
    """

    def __init__(self, risk_free_rate: float = 0.0, trading_days: int = 252):
        """
        Initialize the risk factor calculator.

        Args:
            risk_free_rate: Annual risk-free rate for calculations
            trading_days: Number of trading days per year
        """
        self.risk_free_rate = risk_free_rate
        self.trading_days = trading_days
        self.daily_rf = risk_free_rate / trading_days

        logger.info(f"RiskFactorCalculator initialized with rf={risk_free_rate:.2%}")

    def calculate_returns(
        self, prices: pl.DataFrame, price_col: str = "close", method: str = "log"
    ) -> pl.DataFrame:
        """
        Calculate returns from price data.

        Args:
            prices: DataFrame with price data
            price_col: Column name for prices
            method: 'log' for log returns, 'simple' for arithmetic returns

        Returns:
            DataFrame with returns column added
        """
        if method == "log":
            returns = prices.with_columns(pl.col(price_col).log().diff().alias("returns"))
        else:
            returns = prices.with_columns(pl.col(price_col).pct_change().alias("returns"))

        return returns

    def calculate_momentum(
        self,
        returns: pl.DataFrame | pl.Series,
        window: int = 252,
        skip_recent: int = 21,
        returns_col: str = "returns",
    ) -> RiskFactor:
        """
        Calculate momentum factor (past returns excluding recent period).

        Standard momentum is 12-1 month momentum: returns over past 12 months
        excluding the most recent month.

        Args:
            returns: DataFrame or Series with returns
            window: Lookback window in days (default 252 = 12 months)
            skip_recent: Recent days to skip (default 21 = 1 month)
            returns_col: Column name if DataFrame

        Returns:
            RiskFactor with momentum values
        """
        if isinstance(returns, pl.DataFrame):
            ret_series = returns[returns_col]
        else:
            ret_series = returns

        # Calculate cumulative returns over window, excluding recent period
        momentum_values = []
        ret_np = ret_series.to_numpy()

        for i in range(len(ret_np)):
            if i < window:
                momentum_values.append(np.nan)
            else:
                # Sum returns from (i-window) to (i-skip_recent)
                period_returns = ret_np[i - window : i - skip_recent]
                if len(period_returns) > 0:
                    cum_return = np.exp(np.nansum(period_returns)) - 1
                    momentum_values.append(cum_return)
                else:
                    momentum_values.append(np.nan)

        return RiskFactor(
            name="momentum",
            factor_type=RiskFactorType.MOMENTUM,
            values=pl.Series("momentum", momentum_values),
            description=f"{window}-day momentum excluding last {skip_recent} days",
            calculation_params={"window": window, "skip_recent": skip_recent},
        )

    def calculate_reversal(
        self, returns: pl.DataFrame | pl.Series, window: int = 21, returns_col: str = "returns"
    ) -> RiskFactor:
        """
        Calculate short-term reversal factor (recent returns).

        Args:
            returns: DataFrame or Series with returns
            window: Lookback window in days (default 21 = 1 month)
            returns_col: Column name if DataFrame

        Returns:
            RiskFactor with reversal values
        """
        if isinstance(returns, pl.DataFrame):
            ret_series = returns[returns_col]
        else:
            ret_series = returns

        # Negative of recent returns (high recent returns -> low reversal factor)
        reversal_values = -ret_series.rolling_sum(window_size=window)

        return RiskFactor(
            name="reversal",
            factor_type=RiskFactorType.REVERSAL,
            values=reversal_values.alias("reversal"),
            description=f"{window}-day short-term reversal",
            calculation_params={"window": window},
        )

    def calculate_volatility(
        self,
        returns: pl.DataFrame | pl.Series,
        window: int = 20,
        annualize: bool = True,
        returns_col: str = "returns",
    ) -> RiskFactor:
        """
        Calculate volatility factor (realized volatility).

        Args:
            returns: DataFrame or Series with returns
            window: Lookback window for volatility calculation
            annualize: Whether to annualize the volatility
            returns_col: Column name if DataFrame

        Returns:
            RiskFactor with volatility values
        """
        if isinstance(returns, pl.DataFrame):
            ret_series = returns[returns_col]
        else:
            ret_series = returns

        vol_values = ret_series.rolling_std(window_size=window)

        if annualize:
            vol_values = vol_values * np.sqrt(self.trading_days)

        return RiskFactor(
            name="volatility",
            factor_type=RiskFactorType.VOLATILITY,
            values=vol_values.alias("volatility"),
            description=f"{window}-day realized volatility{'(annualized)' if annualize else ''}",
            calculation_params={"window": window, "annualize": annualize},
        )

    def calculate_beta(
        self, asset_returns: pl.Series, market_returns: pl.Series, window: int = 252
    ) -> RiskFactor:
        """
        Calculate rolling beta relative to market.

        Args:
            asset_returns: Asset return series
            market_returns: Market return series
            window: Rolling window for beta calculation

        Returns:
            RiskFactor with beta values
        """
        asset_np = asset_returns.to_numpy()
        market_np = market_returns.to_numpy()

        beta_values = []

        for i in range(len(asset_np)):
            if i < window:
                beta_values.append(np.nan)
            else:
                asset_window = asset_np[i - window : i]
                market_window = market_np[i - window : i]

                # Filter NaN values
                valid_mask = ~(np.isnan(asset_window) | np.isnan(market_window))
                if valid_mask.sum() < 30:  # Minimum observations
                    beta_values.append(np.nan)
                    continue

                asset_valid = asset_window[valid_mask]
                market_valid = market_window[valid_mask]

                # Calculate beta via covariance / variance
                cov = np.cov(asset_valid, market_valid)[0, 1]
                var = np.var(market_valid)

                beta = cov / var if var > 0 else np.nan
                beta_values.append(beta)

        return RiskFactor(
            name="beta",
            factor_type=RiskFactorType.BETA,
            values=pl.Series("beta", beta_values),
            description=f"{window}-day rolling beta",
            calculation_params={"window": window},
        )

    def calculate_downside_beta(
        self,
        asset_returns: pl.Series,
        market_returns: pl.Series,
        window: int = 252,
        threshold: float = 0.0,
    ) -> RiskFactor:
        """
        Calculate downside beta (beta during market downturns).

        Args:
            asset_returns: Asset return series
            market_returns: Market return series
            window: Rolling window for calculation
            threshold: Return threshold for downside (default 0.0)

        Returns:
            RiskFactor with downside beta values
        """
        asset_np = asset_returns.to_numpy()
        market_np = market_returns.to_numpy()

        downside_beta_values = []

        for i in range(len(asset_np)):
            if i < window:
                downside_beta_values.append(np.nan)
            else:
                asset_window = asset_np[i - window : i]
                market_window = market_np[i - window : i]

                # Filter to downside periods
                downside_mask = market_window < threshold
                if downside_mask.sum() < 20:  # Minimum observations
                    downside_beta_values.append(np.nan)
                    continue

                asset_down = asset_window[downside_mask]
                market_down = market_window[downside_mask]

                # Remove NaN
                valid_mask = ~(np.isnan(asset_down) | np.isnan(market_down))
                if valid_mask.sum() < 10:
                    downside_beta_values.append(np.nan)
                    continue

                asset_valid = asset_down[valid_mask]
                market_valid = market_down[valid_mask]

                cov = np.cov(asset_valid, market_valid)[0, 1]
                var = np.var(market_valid)

                beta = cov / var if var > 0 else np.nan
                downside_beta_values.append(beta)

        return RiskFactor(
            name="downside_beta",
            factor_type=RiskFactorType.DOWNSIDE_BETA,
            values=pl.Series("downside_beta", downside_beta_values),
            description=f"{window}-day downside beta (threshold={threshold})",
            calculation_params={"window": window, "threshold": threshold},
        )

    def calculate_idiosyncratic_volatility(
        self,
        asset_returns: pl.Series,
        market_returns: pl.Series,
        window: int = 252,
        annualize: bool = True,
    ) -> RiskFactor:
        """
        Calculate idiosyncratic volatility (volatility of residuals after market adjustment).

        Args:
            asset_returns: Asset return series
            market_returns: Market return series
            window: Rolling window for calculation
            annualize: Whether to annualize

        Returns:
            RiskFactor with idiosyncratic volatility values
        """
        asset_np = asset_returns.to_numpy()
        market_np = market_returns.to_numpy()

        idio_vol_values = []

        for i in range(len(asset_np)):
            if i < window:
                idio_vol_values.append(np.nan)
            else:
                asset_window = asset_np[i - window : i]
                market_window = market_np[i - window : i]

                # Filter NaN
                valid_mask = ~(np.isnan(asset_window) | np.isnan(market_window))
                if valid_mask.sum() < 30:
                    idio_vol_values.append(np.nan)
                    continue

                asset_valid = asset_window[valid_mask]
                market_valid = market_window[valid_mask]

                # Calculate beta and residuals
                cov = np.cov(asset_valid, market_valid)[0, 1]
                var = np.var(market_valid)
                beta = cov / var if var > 0 else 0

                alpha = np.mean(asset_valid) - beta * np.mean(market_valid)
                residuals = asset_valid - (alpha + beta * market_valid)

                idio_vol = np.std(residuals)
                if annualize:
                    idio_vol *= np.sqrt(self.trading_days)

                idio_vol_values.append(idio_vol)

        return RiskFactor(
            name="idiosyncratic_vol",
            factor_type=RiskFactorType.IDIOSYNCRATIC_VOL,
            values=pl.Series("idiosyncratic_vol", idio_vol_values),
            description=f"{window}-day idiosyncratic volatility",
            calculation_params={"window": window, "annualize": annualize},
        )

    def calculate_tail_risk(
        self,
        returns: pl.DataFrame | pl.Series,
        window: int = 252,
        percentile: float = 5.0,
        returns_col: str = "returns",
    ) -> RiskFactor:
        """
        Calculate tail risk factor (left-tail VaR).

        Args:
            returns: DataFrame or Series with returns
            window: Rolling window for calculation
            percentile: Percentile for VaR (default 5%)
            returns_col: Column name if DataFrame

        Returns:
            RiskFactor with tail risk values
        """
        if isinstance(returns, pl.DataFrame):
            ret_series = returns[returns_col]
        else:
            ret_series = returns

        ret_np = ret_series.to_numpy()
        tail_risk_values = []

        for i in range(len(ret_np)):
            if i < window:
                tail_risk_values.append(np.nan)
            else:
                window_returns = ret_np[i - window : i]
                valid_returns = window_returns[~np.isnan(window_returns)]

                if len(valid_returns) < 20:
                    tail_risk_values.append(np.nan)
                else:
                    # VaR (negative of percentile to make it positive for risk)
                    var = -np.percentile(valid_returns, percentile)
                    tail_risk_values.append(var)

        return RiskFactor(
            name="tail_risk",
            factor_type=RiskFactorType.TAIL_RISK,
            values=pl.Series("tail_risk", tail_risk_values),
            description=f"{window}-day {percentile}% VaR",
            calculation_params={"window": window, "percentile": percentile},
        )

    def calculate_liquidity(
        self, volume: pl.Series, price: pl.Series, window: int = 20
    ) -> RiskFactor:
        """
        Calculate liquidity factor (average dollar volume).

        Args:
            volume: Trading volume series
            price: Price series
            window: Rolling window for averaging

        Returns:
            RiskFactor with liquidity values
        """
        dollar_volume = volume * price
        avg_dollar_volume = dollar_volume.rolling_mean(window_size=window)

        # Log-transform for better distribution
        liquidity_values = avg_dollar_volume.log()

        return RiskFactor(
            name="liquidity",
            factor_type=RiskFactorType.LIQUIDITY,
            values=liquidity_values.alias("liquidity"),
            description=f"{window}-day average log dollar volume",
            calculation_params={"window": window},
        )

    def calculate_turnover(
        self, volume: pl.Series, shares_outstanding: float | pl.Series, window: int = 20
    ) -> RiskFactor:
        """
        Calculate turnover factor (volume relative to shares outstanding).

        Args:
            volume: Trading volume series
            shares_outstanding: Total shares outstanding
            window: Rolling window for averaging

        Returns:
            RiskFactor with turnover values
        """
        if isinstance(shares_outstanding, (int, float)):
            turnover = volume / shares_outstanding
        else:
            turnover = volume / shares_outstanding

        avg_turnover = turnover.rolling_mean(window_size=window)

        return RiskFactor(
            name="turnover",
            factor_type=RiskFactorType.TURNOVER,
            values=avg_turnover.alias("turnover"),
            description=f"{window}-day average turnover ratio",
            calculation_params={"window": window},
        )

    def create_custom_factor(
        self,
        name: str,
        values: pl.Series | np.ndarray | list,
        description: str = "",
        params: dict[str, Any] | None = None,
    ) -> RiskFactor:
        """
        Create a custom risk factor from provided values.

        Args:
            name: Factor name
            values: Factor values
            description: Factor description
            params: Calculation parameters

        Returns:
            Custom RiskFactor
        """
        if isinstance(values, np.ndarray):
            values = pl.Series(name, values)
        elif isinstance(values, list):
            values = pl.Series(name, values)

        return RiskFactor(
            name=name,
            factor_type=RiskFactorType.CUSTOM,
            values=values,
            description=description,
            calculation_params=params or {},
        )


# =============================================================================
# RISK FACTOR ANALYZER CLASS
# =============================================================================


class RiskFactorAnalyzer:
    """
    Analyzer for multi-factor risk decomposition and attribution.

    This class provides methods for analyzing risk exposures, decomposing
    portfolio risk into factor and idiosyncratic components, and calculating
    risk attribution.

    Example:
        >>> analyzer = RiskFactorAnalyzer()
        >>> factors = [momentum_factor, volatility_factor]
        >>> exposure = analyzer.estimate_factor_exposure(returns, factors)
        >>> decomposition = analyzer.decompose_risk(weights, factor_exposures, factor_cov)
    """

    def __init__(
        self,
        covariance_method: CovarianceMethod = CovarianceMethod.LEDOIT_WOLF,
        half_life: int = 63,
    ):
        """
        Initialize the risk factor analyzer.

        Args:
            covariance_method: Method for covariance estimation
            half_life: Half-life for exponential weighting (in days)
        """
        self.covariance_method = covariance_method
        self.half_life = half_life
        self.decay = np.log(2) / half_life

        logger.info(f"RiskFactorAnalyzer initialized with cov_method={covariance_method.value}")

    def estimate_factor_exposure(
        self, asset_returns: pl.Series, factors: list[RiskFactor], include_intercept: bool = True
    ) -> FactorExposure:
        """
        Estimate factor exposures for an asset using OLS regression.

        Args:
            asset_returns: Asset return series
            factors: List of RiskFactor objects
            include_intercept: Whether to include intercept (alpha)

        Returns:
            FactorExposure with estimated loadings
        """
        if not STATSMODELS_AVAILABLE:
            raise ImportError("statsmodels is required for factor exposure estimation")

        # Prepare data
        y = asset_returns.to_numpy()

        # Build factor matrix
        X_dict = {}
        for factor in factors:
            X_dict[factor.name] = factor.values.to_numpy()

        X = np.column_stack(list(X_dict.values()))
        factor_names = list(X_dict.keys())

        # Align data and remove NaN
        valid_mask = ~(np.isnan(y) | np.any(np.isnan(X), axis=1))
        y_valid = y[valid_mask]
        X_valid = X[valid_mask]

        if len(y_valid) < len(factors) + 10:
            logger.warning("Insufficient data for factor exposure estimation")
            return FactorExposure(
                asset_id="unknown",
                exposures={f.name: np.nan for f in factors},
                r_squared=np.nan,
                adj_r_squared=np.nan,
                residual_std=np.nan,
            )

        # Add constant for intercept
        if include_intercept:
            X_valid = sm.add_constant(X_valid)

        # Run OLS regression
        model = sm.OLS(y_valid, X_valid)
        results = model.fit()

        # Extract results
        exposures = {}
        t_stats = {}
        p_values = {}

        param_idx = 1 if include_intercept else 0
        for i, name in enumerate(factor_names):
            exposures[name] = float(results.params[param_idx + i])
            t_stats[name] = float(results.tvalues[param_idx + i])
            p_values[name] = float(results.pvalues[param_idx + i])

        # Include alpha if present
        if include_intercept:
            exposures["alpha"] = float(results.params[0])
            t_stats["alpha"] = float(results.tvalues[0])
            p_values["alpha"] = float(results.pvalues[0])

        return FactorExposure(
            asset_id="asset",
            exposures=exposures,
            r_squared=float(results.rsquared),
            adj_r_squared=float(results.rsquared_adj),
            residual_std=float(np.std(results.resid)),
            t_stats=t_stats,
            p_values=p_values,
        )

    def estimate_factor_covariance(
        self,
        factors: list[RiskFactor],
        method: CovarianceMethod | None = None,
        annualize: bool = True,
    ) -> np.ndarray:
        """
        Estimate the covariance matrix of factors.

        Args:
            factors: List of RiskFactor objects
            method: Covariance estimation method (defaults to class setting)
            annualize: Whether to annualize (multiply by 252)

        Returns:
            Factor covariance matrix
        """
        method = method or self.covariance_method

        # Build factor matrix
        factor_data = []
        for factor in factors:
            factor_data.append(factor.values.to_numpy())

        X = np.column_stack(factor_data)

        # Remove rows with NaN
        valid_mask = ~np.any(np.isnan(X), axis=1)
        X_valid = X[valid_mask]

        if len(X_valid) < len(factors) + 10:
            logger.warning("Insufficient data for covariance estimation")
            return np.eye(len(factors))

        # Estimate covariance based on method
        if method == CovarianceMethod.SAMPLE:
            cov = np.cov(X_valid.T)

        elif method == CovarianceMethod.LEDOIT_WOLF and SKLEARN_AVAILABLE:
            lw = LedoitWolf()
            lw.fit(X_valid)
            cov = lw.covariance_

        elif method == CovarianceMethod.SHRUNK and SKLEARN_AVAILABLE:
            shrunk = ShrunkCovariance()
            shrunk.fit(X_valid)
            cov = shrunk.covariance_

        elif method == CovarianceMethod.EXPONENTIAL:
            # Exponentially weighted covariance
            n = len(X_valid)
            weights = np.exp(-self.decay * np.arange(n)[::-1])
            weights = weights / weights.sum()

            # Weighted mean
            mean = np.average(X_valid, axis=0, weights=weights)

            # Weighted covariance
            centered = X_valid - mean
            cov = np.cov(centered.T, aweights=weights)

        else:
            # Fallback to sample covariance
            cov = np.cov(X_valid.T)

        # Ensure positive semi-definite
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        eigenvalues = np.maximum(eigenvalues, 1e-10)
        cov = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T

        if annualize:
            cov = cov * 252

        return cov

    def decompose_risk(
        self,
        weights: np.ndarray,
        factor_exposures: np.ndarray,
        factor_covariance: np.ndarray,
        idiosyncratic_variances: np.ndarray | None = None,
    ) -> RiskDecomposition:
        """
        Decompose portfolio risk into factor and idiosyncratic components.

        Uses the formula: σ² = w'(B*Σ_f*B' + D)*w
        where B is the factor loading matrix, Σ_f is factor covariance,
        and D is the diagonal idiosyncratic variance matrix.

        Args:
            weights: Portfolio weights (n_assets,)
            factor_exposures: Factor loading matrix (n_assets, n_factors)
            factor_covariance: Factor covariance matrix (n_factors, n_factors)
            idiosyncratic_variances: Asset-specific variances (n_assets,)

        Returns:
            RiskDecomposition with component breakdown
        """
        n_assets = len(weights)
        n_factors = factor_covariance.shape[0]

        # Factor risk contribution
        # Portfolio factor exposure: w'B
        portfolio_factor_exposure = weights @ factor_exposures  # (n_factors,)

        # Factor variance contribution: (w'B) * Σ_f * (w'B)'
        factor_variance = portfolio_factor_exposure @ factor_covariance @ portfolio_factor_exposure

        # Idiosyncratic risk contribution
        if idiosyncratic_variances is None:
            idiosyncratic_variances = np.zeros(n_assets)

        idio_variance = np.sum((weights**2) * idiosyncratic_variances)

        # Total risk
        total_variance = factor_variance + idio_variance
        total_risk = np.sqrt(total_variance)
        factor_risk = np.sqrt(factor_variance)
        idiosyncratic_risk = np.sqrt(idio_variance)

        # Individual factor contributions (marginal)
        factor_contributions = {}
        for i in range(n_factors):
            # Contribution of factor i to portfolio variance
            contrib = portfolio_factor_exposure[i] ** 2 * factor_covariance[
                i, i
            ] + 2 * portfolio_factor_exposure[i] * np.sum(
                portfolio_factor_exposure[:i] * factor_covariance[i, :i]
            )
            factor_contributions[f"factor_{i}"] = float(np.sqrt(max(contrib, 0)))

        # Percentage contributions
        percentage_contributions = {}
        if total_variance > 0:
            percentage_contributions["factor_risk"] = factor_variance / total_variance
            percentage_contributions["idiosyncratic_risk"] = idio_variance / total_variance

        return RiskDecomposition(
            total_risk=float(total_risk),
            factor_risk=float(factor_risk),
            idiosyncratic_risk=float(idiosyncratic_risk),
            factor_contributions=factor_contributions,
            percentage_contributions=percentage_contributions,
        )

    def calculate_risk_contribution(
        self, weights: np.ndarray, covariance: np.ndarray
    ) -> dict[str, np.ndarray]:
        """
        Calculate marginal and total risk contributions by asset.

        Args:
            weights: Portfolio weights
            covariance: Asset covariance matrix

        Returns:
            Dictionary with 'marginal' and 'total' risk contributions
        """
        # Portfolio variance and volatility
        portfolio_variance = weights @ covariance @ weights
        portfolio_vol = np.sqrt(portfolio_variance)

        # Marginal risk contribution: d(σ)/d(w) = Σw / σ
        marginal = (covariance @ weights) / portfolio_vol

        # Total risk contribution: w * marginal
        total = weights * marginal

        # Percentage contribution
        percentage = total / portfolio_vol

        return {
            "marginal": marginal,
            "total": total,
            "percentage": percentage,
            "portfolio_volatility": portfolio_vol,
        }

    def construct_factor_mimicking_portfolio(
        self, target_factor: RiskFactor, asset_returns: pl.DataFrame, method: str = "regression"
    ) -> FactorMimickingPortfolio:
        """
        Construct a portfolio that mimics a factor's returns.

        Args:
            target_factor: The factor to mimic
            asset_returns: DataFrame with asset returns (columns = assets)
            method: 'regression' or 'optimization'

        Returns:
            FactorMimickingPortfolio with weights
        """
        factor_values = target_factor.values.to_numpy()

        # Get asset returns as matrix
        asset_names = [col for col in asset_returns.columns if col != "date"]
        returns_matrix = asset_returns.select(asset_names).to_numpy()

        # Align data
        min_len = min(len(factor_values), len(returns_matrix))
        factor_values = factor_values[-min_len:]
        returns_matrix = returns_matrix[-min_len:]

        # Remove NaN
        valid_mask = ~(np.isnan(factor_values) | np.any(np.isnan(returns_matrix), axis=1))
        factor_valid = factor_values[valid_mask]
        returns_valid = returns_matrix[valid_mask]

        if method == "regression":
            # Use regression to find weights
            X = sm.add_constant(returns_valid)
            model = sm.OLS(factor_valid, X)
            results = model.fit()

            weights = results.params[1:]  # Exclude intercept

            # Normalize weights to sum to 1
            weights = weights / np.sum(np.abs(weights))

        else:  # optimization
            # Minimize tracking error
            def tracking_error(w):
                portfolio_returns = returns_valid @ w
                te = np.std(portfolio_returns - factor_valid)
                return te

            n_assets = returns_valid.shape[1]
            initial_weights = np.ones(n_assets) / n_assets

            constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
            bounds = [(-1, 1)] * n_assets

            result = minimize(
                tracking_error,
                initial_weights,
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
            )

            weights = result.x

        # Calculate metrics
        portfolio_returns = returns_valid @ weights
        tracking_error = np.std(portfolio_returns - factor_valid) * np.sqrt(252)
        expected_return = np.mean(portfolio_returns) * 252
        correlation = np.corrcoef(portfolio_returns, factor_valid)[0, 1]

        info_ratio = expected_return / tracking_error if tracking_error > 0 else 0

        weights_dict = {name: float(w) for name, w in zip(asset_names, weights)}

        return FactorMimickingPortfolio(
            factor_name=target_factor.name,
            weights=weights_dict,
            expected_return=float(expected_return),
            tracking_error=float(tracking_error),
            information_ratio=float(info_ratio),
            correlation=float(correlation),
        )

    def stress_test_factors(
        self,
        weights: np.ndarray,
        factor_exposures: np.ndarray,
        factor_shocks: dict[str, float],
        factor_names: list[str],
    ) -> dict[str, Any]:
        """
        Perform stress test with specified factor shocks.

        Args:
            weights: Portfolio weights
            factor_exposures: Factor loading matrix
            factor_shocks: Dictionary of factor name to shock magnitude
            factor_names: Names corresponding to factor columns

        Returns:
            Dictionary with stress test results
        """
        # Portfolio factor exposures
        portfolio_exposures = weights @ factor_exposures

        # Apply shocks
        total_impact = 0.0
        factor_impacts = {}

        for i, name in enumerate(factor_names):
            if name in factor_shocks:
                shock = factor_shocks[name]
                impact = portfolio_exposures[i] * shock
                factor_impacts[name] = float(impact)
                total_impact += impact
            else:
                factor_impacts[name] = 0.0

        return {
            "total_impact": float(total_impact),
            "factor_impacts": factor_impacts,
            "portfolio_exposures": {
                name: float(portfolio_exposures[i]) for i, name in enumerate(factor_names)
            },
        }


# =============================================================================
# ASYNC WRAPPER FUNCTIONS
# =============================================================================


async def calculate_risk_factors_async(
    prices: pl.DataFrame,
    market_prices: pl.DataFrame | None = None,
    factors_to_calculate: list[RiskFactorType] | None = None,
) -> dict[str, RiskFactor]:
    """
    Asynchronously calculate multiple risk factors from price data.

    Args:
        prices: DataFrame with price data (columns: date, close, volume, etc.)
        market_prices: Optional market benchmark prices for beta calculations
        factors_to_calculate: List of factor types to calculate

    Returns:
        Dictionary of factor name to RiskFactor
    """
    if factors_to_calculate is None:
        factors_to_calculate = [
            RiskFactorType.MOMENTUM,
            RiskFactorType.VOLATILITY,
            RiskFactorType.REVERSAL,
        ]

    loop = asyncio.get_event_loop()
    calculator = RiskFactorCalculator()

    # Calculate returns
    returns_df = calculator.calculate_returns(prices)
    returns = returns_df["returns"]

    # Calculate market returns if provided
    market_returns = None
    if market_prices is not None:
        market_returns_df = calculator.calculate_returns(market_prices)
        market_returns = market_returns_df["returns"]

    factors = {}

    for factor_type in factors_to_calculate:
        try:
            if factor_type == RiskFactorType.MOMENTUM:
                factor = await loop.run_in_executor(
                    None, lambda: calculator.calculate_momentum(returns)
                )
                factors[factor_type.value] = factor

            elif factor_type == RiskFactorType.VOLATILITY:
                factor = await loop.run_in_executor(
                    None, lambda: calculator.calculate_volatility(returns)
                )
                factors[factor_type.value] = factor

            elif factor_type == RiskFactorType.REVERSAL:
                factor = await loop.run_in_executor(
                    None, lambda: calculator.calculate_reversal(returns)
                )
                factors[factor_type.value] = factor

            elif factor_type == RiskFactorType.TAIL_RISK:
                factor = await loop.run_in_executor(
                    None, lambda: calculator.calculate_tail_risk(returns)
                )
                factors[factor_type.value] = factor

            elif factor_type == RiskFactorType.BETA and market_returns is not None:
                factor = await loop.run_in_executor(
                    None, lambda: calculator.calculate_beta(returns, market_returns)
                )
                factors[factor_type.value] = factor

            elif factor_type == RiskFactorType.DOWNSIDE_BETA and market_returns is not None:
                factor = await loop.run_in_executor(
                    None, lambda: calculator.calculate_downside_beta(returns, market_returns)
                )
                factors[factor_type.value] = factor

            elif factor_type == RiskFactorType.IDIOSYNCRATIC_VOL and market_returns is not None:
                factor = await loop.run_in_executor(
                    None,
                    lambda: calculator.calculate_idiosyncratic_volatility(returns, market_returns),
                )
                factors[factor_type.value] = factor

            elif factor_type == RiskFactorType.LIQUIDITY:
                if "volume" in prices.columns and "close" in prices.columns:
                    factor = await loop.run_in_executor(
                        None,
                        lambda: calculator.calculate_liquidity(prices["volume"], prices["close"]),
                    )
                    factors[factor_type.value] = factor

        except Exception as e:
            logger.warning(f"Failed to calculate {factor_type.value}: {e}")

    return factors


async def analyze_factor_exposures_async(
    asset_returns: dict[str, pl.Series], factors: list[RiskFactor]
) -> dict[str, FactorExposure]:
    """
    Asynchronously analyze factor exposures for multiple assets.

    Args:
        asset_returns: Dictionary of asset name to return series
        factors: List of RiskFactor objects

    Returns:
        Dictionary of asset name to FactorExposure
    """
    loop = asyncio.get_event_loop()
    analyzer = RiskFactorAnalyzer()

    exposures = {}

    for asset_name, returns in asset_returns.items():
        try:
            exposure = await loop.run_in_executor(
                None, lambda r=returns: analyzer.estimate_factor_exposure(r, factors)
            )
            exposure.asset_id = asset_name
            exposures[asset_name] = exposure

        except Exception as e:
            logger.warning(f"Failed to analyze {asset_name}: {e}")

    return exposures


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


def get_standard_risk_factors(
    returns: pl.DataFrame, market_returns: pl.Series | None = None, returns_col: str = "returns"
) -> list[RiskFactor]:
    """
    Calculate a standard set of risk factors.

    Args:
        returns: DataFrame with returns
        market_returns: Optional market benchmark returns
        returns_col: Column name for returns

    Returns:
        List of common risk factors
    """
    calculator = RiskFactorCalculator()
    factors = []

    # Momentum
    factors.append(calculator.calculate_momentum(returns, returns_col=returns_col))

    # Volatility
    factors.append(calculator.calculate_volatility(returns, returns_col=returns_col))

    # Reversal
    factors.append(calculator.calculate_reversal(returns, returns_col=returns_col))

    # Tail risk
    factors.append(calculator.calculate_tail_risk(returns, returns_col=returns_col))

    # Beta-related factors if market returns provided
    if market_returns is not None:
        ret_series = returns[returns_col] if isinstance(returns, pl.DataFrame) else returns
        factors.append(calculator.calculate_beta(ret_series, market_returns))
        factors.append(calculator.calculate_downside_beta(ret_series, market_returns))
        factors.append(calculator.calculate_idiosyncratic_volatility(ret_series, market_returns))

    return factors


def decompose_portfolio_risk(
    weights: dict[str, float], returns: pl.DataFrame, factors: list[RiskFactor]
) -> RiskDecomposition:
    """
    Decompose portfolio risk into factor components.

    Args:
        weights: Asset weights dictionary
        returns: DataFrame with asset returns (columns = assets)
        factors: List of risk factors

    Returns:
        RiskDecomposition results
    """
    analyzer = RiskFactorAnalyzer()

    # Convert weights to array
    asset_names = list(weights.keys())
    weights_array = np.array([weights[name] for name in asset_names])

    # Estimate exposures for each asset
    exposures_list = []
    for asset_name in asset_names:
        if asset_name in returns.columns:
            exposure = analyzer.estimate_factor_exposure(returns[asset_name], factors)
            exposures_list.append([exposure.exposures.get(f.name, 0) for f in factors])
        else:
            exposures_list.append([0] * len(factors))

    factor_exposures = np.array(exposures_list)

    # Estimate factor covariance
    factor_covariance = analyzer.estimate_factor_covariance(factors)

    # Decompose
    return analyzer.decompose_risk(weights_array, factor_exposures, factor_covariance)


# =============================================================================
# MAIN EXECUTION (EXAMPLE USAGE)
# =============================================================================

if __name__ == "__main__":
    import asyncio

    async def main():
        """Example usage of the risk factors module"""

        # Generate sample data
        np.random.seed(42)
        n_days = 500
        dates = [datetime.now() - timedelta(days=n_days - i) for i in range(n_days)]

        # Simulate price and market data
        market_returns = np.random.normal(0.0005, 0.01, n_days)
        asset_returns = 1.2 * market_returns + np.random.normal(0, 0.005, n_days)

        # Create DataFrames
        prices_df = pl.DataFrame(
            {
                "date": dates,
                "close": 100 * np.exp(np.cumsum(asset_returns)),
                "volume": np.random.uniform(1e6, 5e6, n_days),
                "returns": asset_returns,
            }
        )

        market_df = pl.DataFrame(
            {
                "date": dates,
                "close": 100 * np.exp(np.cumsum(market_returns)),
                "returns": market_returns,
            }
        )

        print("=" * 60)
        print("Risk Factors Module - Example Usage")
        print("=" * 60)

        # Initialize calculator
        calculator = RiskFactorCalculator(risk_free_rate=0.03)

        # Calculate individual factors
        print("\n1. Calculating risk factors...")
        momentum = calculator.calculate_momentum(prices_df, returns_col="returns")
        volatility = calculator.calculate_volatility(prices_df, returns_col="returns")
        beta = calculator.calculate_beta(pl.Series(asset_returns), pl.Series(market_returns))

        print(f"   Momentum: mean={momentum.mean:.4f}, std={momentum.std:.4f}")
        print(f"   Volatility: mean={volatility.mean:.4f}")
        print(f"   Beta: mean={beta.mean:.4f}")

        # Initialize analyzer
        analyzer = RiskFactorAnalyzer(covariance_method=CovarianceMethod.LEDOIT_WOLF)

        # Estimate factor exposures
        print("\n2. Estimating factor exposures...")
        factors = [momentum, volatility]
        exposure = analyzer.estimate_factor_exposure(pl.Series(asset_returns), factors)

        print(f"   R-squared: {exposure.r_squared:.4f}")
        print(f"   Exposures: {exposure.exposures}")

        # Estimate factor covariance
        print("\n3. Estimating factor covariance...")
        factor_cov = analyzer.estimate_factor_covariance(factors)
        print(f"   Factor covariance matrix shape: {factor_cov.shape}")
        print(
            f"   Factor correlation: {factor_cov[0, 1] / np.sqrt(factor_cov[0, 0] * factor_cov[1, 1]):.4f}"
        )

        # Async factor calculation
        print("\n4. Async factor calculation...")
        factors_dict = await calculate_risk_factors_async(
            prices_df,
            market_df,
            [RiskFactorType.MOMENTUM, RiskFactorType.VOLATILITY, RiskFactorType.BETA],
        )
        print(f"   Calculated factors: {list(factors_dict.keys())}")

        # Stress test
        print("\n5. Stress testing...")
        weights = np.array([0.5, 0.3, 0.2])
        factor_exposures = np.array([[1.2, 0.5], [0.8, 0.3], [1.5, 0.8]])

        stress_results = analyzer.stress_test_factors(
            weights,
            factor_exposures,
            {"factor_0": -0.10, "factor_1": 0.05},  # 10% drop in factor 0, 5% rise in factor 1
            ["factor_0", "factor_1"],
        )

        print(f"   Total impact: {stress_results['total_impact']:.4%}")
        print(f"   Factor impacts: {stress_results['factor_impacts']}")

        print("\n" + "=" * 60)
        print("Risk factors analysis complete!")
        print("=" * 60)

    asyncio.run(main())
