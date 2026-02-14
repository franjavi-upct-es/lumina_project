# backend/simulation/generators/synthetic_data.py
"""
Synthetic Data Generator

Generates synthetic OHLCV price data using stochastic models:
- Geometric Brownian Motion (GBM)
- Jump Diffusion Process
- Mean-reverting processes

Used for:
- Training data augmentation
- Stress testing with custom market conditions
- Generating scenarios that haven't occurred historically

Mathematical Models:
1. GBM: dS = μS dt + σS dW
2. Jump Diffusion: dS = μS dt + σS dW + J dN
   where N is a Poisson process

References:
- Hull (2018): "Options, Futures, and Other Derivatives"
- Merton (1976): "Option Pricing with Jump Diffusions"
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from loguru import logger


@dataclass
class SyntheticDataConfig:
    """
    Configuration for synthetic data generation.

    Attributes:
        initial_price: Starting price
        num_steps: Number of time steps
        dt: Time step size (e.g., 1/252 for daily)
        mu: Drift (expected return)
        sigma: Volatility
        seed: Random seed for reproducibility
    """

    initial_price: float = 100.0
    num_steps: int = 252
    dt: float = 1 / 252
    mu: float = 0.10  # 10% annual drift
    sigma: float = 0.20  # 20% annual volatility
    seed: int | None = None


class GBMGenerator:
    """
    Geometric Brownian Motion generator.

    Models stock prices as:
        S(t+dt) = S(t) * exp((μ - σ²/2)dt + σ√dt * Z)

    where Z ~ N(0,1)

    This is the classic Black-Scholes model for asset prices.
    """

    def __init__(self, config: SyntheticDataConfig | None = None):
        """
        Initialize GBM generator.

        Args:
            config: Configuration parameters
        """
        self.config = config or SyntheticDataConfig()

        if self.config.seed is not None:
            np.random.seed(self.config.seed)

        logger.debug(f"GBM Generator: μ={self.config.mu:.2%}, σ={self.config.sigma:.2%}")

    def generate(self) -> np.ndarray:
        """
        Generate price path using GBM.

        Returns:
            Array of prices [num_steps]
        """
        # Generate random shocks
        Z = np.random.standard_normal(self.config.num_steps)

        # Calculate log returns
        log_returns = (
            self.config.mu - 0.5 * self.config.sigma**2
        ) * self.config.dt + self.config.sigma * np.sqrt(self.config.dt) * Z

        # Convert to price path
        log_prices = np.log(self.config.initial_price) + np.cumsum(log_returns)
        prices = np.exp(log_prices)

        return prices

    def generate_ohlcv(self, intraday_steps: int = 60) -> pd.DataFrame:
        """
        Generate OHLCV data with intraday variation.

        Args:
            intraday_steps: Number of intraday ticks per bar

        Returns:
            DataFrame with OHLCV columns
        """
        prices = self.generate()

        ohlcv = []
        for i, close_price in enumerate(prices):
            # Generate intraday prices around close
            intraday_mu = self.config.mu / self.config.num_steps
            intraday_sigma = self.config.sigma / np.sqrt(self.config.num_steps)

            # Start from previous close
            prev_close = self.config.initial_price if i == 0 else prices[i - 1]

            # Generate intraday path
            intraday_returns = np.random.normal(intraday_mu, intraday_sigma, intraday_steps)
            intraday_prices = prev_close * np.exp(np.cumsum(intraday_returns))

            # Ensure close matches generated price
            intraday_prices = intraday_prices * (close_price / intraday_prices[-1])

            # OHLC
            open_price = prev_close
            high_price = np.max(intraday_prices)
            low_price = np.min(intraday_prices)

            # Synthetic volume (correlated with volatility)
            volume = np.random.lognormal(mean=15, sigma=0.5) * 1_000_000

            ohlcv.append(
                {
                    "open": open_price,
                    "high": high_price,
                    "low": low_price,
                    "close": close_price,
                    "volume": volume,
                }
            )

        df = pd.DataFrame(ohlcv)
        df.index = pd.date_range(start="2020-01-01", periods=len(df), freq="D")

        return df


class JumpDiffusionGenerator:
    """
    Merton Jump Diffusion model.

    Adds sudden jumps to GBM to model crashes and rallies:
        dS = μS dt + σS dW + J dN

    where:
    - dN: Poisson process (jump occurrences)
    - J: Jump size (log-normally distributed)
    """

    def __init__(
        self,
        config: SyntheticDataConfig | None = None,
        jump_intensity: float = 10.0,  # λ (jumps per year)
        jump_mean: float = -0.05,  # Mean jump size (negative = crash)
        jump_std: float = 0.10,  # Jump size volatility
    ):
        """
        Initialize jump diffusion generator.

        Args:
            config: Base configuration
            jump_intensity: Expected jumps per year (λ)
            jump_mean: Mean log jump size
            jump_std: Jump size standard deviation
        """
        self.config = config or SyntheticDataConfig()
        self.jump_intensity = jump_intensity
        self.jump_mean = jump_mean
        self.jump_std = jump_std

        if self.config.seed is not None:
            np.random.seed(self.config.seed)

        logger.debug(
            f"Jump Diffusion: λ={jump_intensity}, "
            f"jump_mean={jump_mean:.2%}, jump_std={jump_std:.2%}"
        )

    def generate(self) -> np.ndarray:
        """
        Generate price path with jumps.

        Returns:
            Array of prices [num_steps]
        """
        # GBM component
        Z = np.random.standard_normal(self.config.num_steps)

        # Diffusion component
        diffusion = (
            self.config.mu - 0.5 * self.config.sigma**2
        ) * self.config.dt + self.config.sigma * np.sqrt(self.config.dt) * Z

        # Jump component
        # Number of jumps per step (Poisson process)
        jump_prob = self.jump_intensity * self.config.dt
        jumps = np.random.poisson(jump_prob, self.config.num_steps)

        # Jump sizes (log-normal distribution)
        jump_sizes = np.zeros(self.config.num_steps)
        for i in range(self.config.num_steps):
            if jumps[i] > 0:
                # Sum of multiple jumps if more than one occurs
                jump_sizes[i] = np.sum(np.random.normal(self.jump_mean, self.jump_std, jumps[i]))

        # Combine diffusion and jumps
        log_returns = diffusion + jump_sizes

        # Convert to prices
        log_prices = np.log(self.config.initial_price) + np.cumsum(log_returns)
        prices = np.exp(log_prices)

        return prices

    def generate_ohlcv(self, intraday_steps: int = 60) -> pd.DataFrame:
        """
        Generate OHLCV data with jumps.

        Args:
            intraday_steps: Intraday ticks per bar

        Returns:
            DataFrame with OHLCV columns
        """
        prices = self.generate()

        ohlcv = []
        for i, close_price in enumerate(prices):
            prev_close = self.config.initial_price if i == 0 else prices[i - 1]

            # Simple OHLC approximation
            # For jumps, open != prev_close (gap)
            change = (close_price - prev_close) / prev_close

            if abs(change) > 0.05:  # Large move indicates jump
                # Gap opening
                open_price = prev_close * (1 + 0.7 * change)
            else:
                open_price = prev_close

            # High/Low with some noise
            high_price = max(open_price, close_price) * (1 + abs(np.random.normal(0, 0.01)))
            low_price = min(open_price, close_price) * (1 - abs(np.random.normal(0, 0.01)))

            # Volume spikes on jumps
            base_volume = 1_000_000
            if abs(change) > 0.05:
                volume = base_volume * np.random.uniform(3, 10)
            else:
                volume = base_volume * np.random.lognormal(0, 0.5)

            ohlcv.append(
                {
                    "open": open_price,
                    "high": high_price,
                    "low": low_price,
                    "close": close_price,
                    "volume": volume,
                }
            )

        df = pd.DataFrame(ohlcv)
        df.index = pd.date_range(start="2020-01-01", periods=len(df), freq="D")

        return df


class SyntheticDataGenerator:
    """
    Main synthetic data generator with multiple models.

    Provides a unified interface for generating synthetic market data
    with various stochastic processes.
    """

    def __init__(self, model: str = "gbm", config: SyntheticDataConfig | None = None, **kwargs):
        """
        Initialize synthetic data generator.

        Args:
            model: Model type ('gbm' or 'jump_diffusion')
            config: Configuration parameters
            **kwargs: Model-specific parameters
        """
        self.model = model
        self.config = config or SyntheticDataConfig()

        if model == "gbm":
            self.generator = GBMGenerator(config)
        elif model == "jump_diffusion":
            self.generator = JumpDiffusionGenerator(config, **kwargs)
        else:
            raise ValueError(f"Unknown model: {model}")

        logger.info(f"SyntheticDataGenerator initialized with model: {model}")

    def generate_prices(self) -> np.ndarray:
        """Generate price array."""
        return self.generator.generate()

    def generate_ohlcv(self, **kwargs) -> pd.DataFrame:
        """Generate OHLCV DataFrame."""
        return self.generator.generate_ohlcv(**kwargs)

    def generate_multiple_assets(
        self, n_assets: int, correlation: np.ndarray | None = None
    ) -> pd.DataFrame:
        """
        Generate correlated multi-asset data.

        Args:
            n_assets: Number of assets
            correlation: Correlation matrix [n_assets, n_assets]

        Returns:
            DataFrame with prices for each asset
        """
        if correlation is None:
            # Random correlation matrix
            correlation = np.random.uniform(0.3, 0.8, (n_assets, n_assets))
            np.fill_diagonal(correlation, 1.0)
            # Make symmetric
            correlation = (correlation + correlation.T) / 2

        # Generate correlated random shocks
        L = np.linalg.cholesky(correlation)

        prices_dict = {}
        for i in range(n_assets):
            # Generate independent shocks
            Z = np.random.standard_normal(self.config.num_steps)

            # Apply correlation via Cholesky decomposition
            correlated_Z = L[i] @ np.random.standard_normal((n_assets, self.config.num_steps))

            # Generate prices
            log_returns = (
                self.config.mu - 0.5 * self.config.sigma**2
            ) * self.config.dt + self.config.sigma * np.sqrt(self.config.dt) * correlated_Z[i]

            log_prices = np.log(self.config.initial_price) + np.cumsum(log_returns)
            prices = np.exp(log_prices)

            prices_dict[f"asset_{i + 1}"] = prices

        df = pd.DataFrame(prices_dict)
        df.index = pd.date_range(start="2020-01-01", periods=len(df), freq="D")

        return df


def generate_test_data(
    days: int = 252, volatility: float = 0.20, trend: float = 0.10
) -> pd.DataFrame:
    """
    Quick function to generate test OHLCV data.

    Args:
        days: Number of trading days
        volatility: Annual volatility
        trend: Annual drift

    Returns:
        OHLCV DataFrame
    """
    config = SyntheticDataConfig(
        initial_price=100.0, num_steps=days, dt=1 / 252, mu=trend, sigma=volatility
    )

    generator = SyntheticDataGenerator(model="gbm", config=config)
    return generator.generate_ohlcv()
