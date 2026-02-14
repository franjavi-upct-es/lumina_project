# backend/simulation/environments/reward_functions.py
"""
Reward Functions for RL Training

Implements various reward functions for trading agents:
- Sharpe Ratio: Maximize risk-adjusted returns
- Sortino Ratio: Focus on downside risk
- Calmar Ratio: Return relative to max drawdown
- Composite Rewards: Weighted combinations

The choice of reward function fundamentally shapes agent behavior:
- Sharpe → Balanced risk-return
- Sortino → Asymmetric risk (downside matters more)
- Calmar → Drawdown-averse strategies

Mathematical Formulations included in docstrings.
"""

from abc import ABC, abstractmethod

import numpy as np
from loguru import logger


class RewardFunction(ABC):
    """
    Abstract base class for reward functions.

    All reward functions must implement calculate() method that
    takes episode history and returns a scalar reward.
    """

    @abstractmethod
    def calculate(self, returns: np.ndarray, portfolio_values: np.ndarray, **kwargs) -> float:
        """
        Calculate reward from episode data.

        Args:
            returns: Array of period returns
            portfolio_values: Array of portfolio values over time
            **kwargs: Additional data (drawdowns, volatility, etc.)

        Returns:
            Scalar reward value
        """
        pass

    @abstractmethod
    def get_name(self) -> str:
        """Return reward function name."""
        pass


class SharpeReward(RewardFunction):
    """
    Sharpe Ratio reward function.

    Maximizes risk-adjusted returns:
        Sharpe = (E[R] - R_f) / σ_R

    Where:
    - E[R] = Expected return
    - R_f = Risk-free rate
    - σ_R = Standard deviation of returns

    This encourages consistent returns with low volatility.
    """

    def __init__(
        self,
        risk_free_rate: float = 0.0,
        annualization_factor: float = np.sqrt(252),
        clip_range: tuple = (-10, 10),
    ):
        """
        Initialize Sharpe reward.

        Args:
            risk_free_rate: Annual risk-free rate
            annualization_factor: Factor to annualize (sqrt(252) for daily)
            clip_range: Clip reward to prevent extremes
        """
        self.risk_free_rate = risk_free_rate
        self.annualization_factor = annualization_factor
        self.clip_range = clip_range

        logger.debug(f"SharpeReward: rf={risk_free_rate:.2%}")

    def calculate(self, returns: np.ndarray, portfolio_values: np.ndarray, **kwargs) -> float:
        """Calculate annualized Sharpe ratio."""
        if len(returns) < 2:
            return 0.0

        # Remove any NaN or inf
        returns = returns[np.isfinite(returns)]

        if len(returns) == 0:
            return 0.0

        # Calculate excess returns
        excess_returns = returns - self.risk_free_rate / 252  # Daily risk-free

        # Sharpe ratio
        mean_excess = np.mean(excess_returns)
        std_excess = np.std(excess_returns)

        if std_excess == 0 or np.isnan(std_excess):
            return 0.0

        sharpe = (mean_excess / std_excess) * self.annualization_factor

        # Clip to reasonable range
        sharpe = np.clip(sharpe, *self.clip_range)

        return float(sharpe)

    def get_name(self) -> str:
        return "sharpe_ratio"


class SortinoReward(RewardFunction):
    """
    Sortino Ratio reward function.

    Penalizes downside volatility only:
        Sortino = (E[R] - R_f) / σ_downside

    Where σ_downside uses only returns below target.

    This is more appropriate for traders who care primarily
    about avoiding losses rather than reducing all volatility.
    """

    def __init__(
        self,
        risk_free_rate: float = 0.0,
        target_return: float = 0.0,
        annualization_factor: float = np.sqrt(252),
        clip_range: tuple = (-10, 10),
    ):
        """
        Initialize Sortino reward.

        Args:
            risk_free_rate: Annual risk-free rate
            target_return: Minimum acceptable return
            annualization_factor: Annualization factor
            clip_range: Clipping range
        """
        self.risk_free_rate = risk_free_rate
        self.target_return = target_return
        self.annualization_factor = annualization_factor
        self.clip_range = clip_range

        logger.debug(f"SortinoReward: target={target_return:.2%}")

    def calculate(self, returns: np.ndarray, portfolio_values: np.ndarray, **kwargs) -> float:
        """Calculate annualized Sortino ratio."""
        if len(returns) < 2:
            return 0.0

        returns = returns[np.isfinite(returns)]

        if len(returns) == 0:
            return 0.0

        # Excess returns
        excess_returns = returns - self.risk_free_rate / 252

        # Downside deviation (only negative excess returns)
        downside_returns = excess_returns[excess_returns < self.target_return]

        if len(downside_returns) == 0:
            # No downside - excellent!
            return self.clip_range[1]

        downside_std = np.std(downside_returns)

        if downside_std == 0 or np.isnan(downside_std):
            return 0.0

        mean_excess = np.mean(excess_returns)
        sortino = (mean_excess / downside_std) * self.annualization_factor

        sortino = np.clip(sortino, *self.clip_range)

        return float(sortino)

    def get_name(self) -> str:
        return "sortino_ratio"


class CalmarReward(RewardFunction):
    """
    Calmar Ratio reward function.

    Reward relative to maximum drawdown:
        Calmar = Annualized Return / Max Drawdown

    This penalizes strategies with large drawdowns even if
    they have high returns.

    Particularly useful for risk-averse investors and for
    Phase B training where surviving drawdowns is critical.
    """

    def __init__(
        self,
        annualization_factor: float = 252,
        min_drawdown: float = 0.01,
        clip_range: tuple = (-10, 10),
    ):
        """
        Initialize Calmar reward.

        Args:
            annualization_factor: Days per year
            min_drawdown: Minimum drawdown to avoid division by zero
            clip_range: Clipping range
        """
        self.annualization_factor = annualization_factor
        self.min_drawdown = min_drawdown
        self.clip_range = clip_range

        logger.debug("CalmarReward initialized")

    def calculate(self, returns: np.ndarray, portfolio_values: np.ndarray, **kwargs) -> float:
        """Calculate Calmar ratio."""
        if len(returns) < 2 or len(portfolio_values) < 2:
            return 0.0

        # Annualized return
        total_return = (portfolio_values[-1] / portfolio_values[0]) - 1
        n_periods = len(returns)
        annualized_return = (1 + total_return) ** (self.annualization_factor / n_periods) - 1

        # Maximum drawdown
        running_max = np.maximum.accumulate(portfolio_values)
        drawdowns = (running_max - portfolio_values) / running_max
        max_drawdown = np.max(drawdowns)

        # Avoid division by zero
        max_drawdown = max(max_drawdown, self.min_drawdown)

        calmar = annualized_return / max_drawdown

        calmar = np.clip(calmar, *self.clip_range)

        return float(calmar)

    def get_name(self) -> str:
        return "calmar_ratio"


class CompositeReward(RewardFunction):
    """
    Weighted combination of multiple reward functions.

    Allows training agents with multiple objectives:
        R_total = w1 * R_sharpe + w2 * R_sortino + w3 * R_calmar

    Useful for Phase C training where we want to optimize
    multiple metrics simultaneously.
    """

    def __init__(self, reward_functions: list[RewardFunction], weights: list[float]):
        """
        Initialize composite reward.

        Args:
            reward_functions: List of reward functions
            weights: Corresponding weights (must sum to 1.0)
        """
        if len(reward_functions) != len(weights):
            raise ValueError("Number of functions and weights must match")

        if not np.isclose(sum(weights), 1.0):
            logger.warning(f"Weights sum to {sum(weights)}, not 1.0. Normalizing.")
            weights = [w / sum(weights) for w in weights]

        self.reward_functions = reward_functions
        self.weights = weights

        logger.info(f"CompositeReward: {len(reward_functions)} functions, weights={weights}")

    def calculate(self, returns: np.ndarray, portfolio_values: np.ndarray, **kwargs) -> float:
        """Calculate weighted combination of rewards."""
        total_reward = 0.0

        for func, weight in zip(self.reward_functions, self.weights):
            reward = func.calculate(returns, portfolio_values, **kwargs)
            total_reward += weight * reward

        return float(total_reward)

    def get_name(self) -> str:
        names = [f.get_name() for f in self.reward_functions]
        return f"composite_({'+'.join(names)})"

    def get_component_rewards(
        self, returns: np.ndarray, portfolio_values: np.ndarray, **kwargs
    ) -> dict[str, float]:
        """
        Get individual component rewards.

        Useful for logging and analysis.

        Returns:
            Dictionary mapping function names to rewards
        """
        components = {}

        for func, weight in zip(self.reward_functions, self.weights):
            reward = func.calculate(returns, portfolio_values, **kwargs)
            components[func.get_name()] = reward
            components[f"{func.get_name()}_weighted"] = reward * weight

        components["total"] = self.calculate(returns, portfolio_values, **kwargs)

        return components


class SimpleReturnReward(RewardFunction):
    """
    Simple return-based reward.

    Just uses raw returns without risk adjustment.
    Useful for Phase A (behavioral cloning) where we want
    the agent to simply mimic profitable trades.
    """

    def __init__(self, scale: float = 1.0):
        """
        Initialize simple return reward.

        Args:
            scale: Scaling factor for rewards
        """
        self.scale = scale

    def calculate(self, returns: np.ndarray, portfolio_values: np.ndarray, **kwargs) -> float:
        """Calculate total return."""
        if len(portfolio_values) < 2:
            return 0.0

        total_return = (portfolio_values[-1] / portfolio_values[0]) - 1

        return float(total_return * self.scale)

    def get_name(self) -> str:
        return "simple_return"


def create_phase_reward(phase: str) -> RewardFunction:
    """
    Create appropriate reward function for training phase.

    Args:
        phase: Training phase ('A', 'B', or 'C')

    Returns:
        Reward function appropriate for phase
    """
    if phase == "A":
        # Phase A: Simple return (imitation learning)
        return SimpleReturnReward(scale=100.0)

    elif phase == "B":
        # Phase B: Survival focus (Calmar with Sortino)
        return CompositeReward(
            reward_functions=[CalmarReward(), SortinoReward()], weights=[0.6, 0.4]
        )

    elif phase == "C":
        # Phase C: Pure optimization (Sharpe + Sortino + Calmar)
        return CompositeReward(
            reward_functions=[SharpeReward(), SortinoReward(), CalmarReward()],
            weights=[0.4, 0.3, 0.3],
        )

    else:
        raise ValueError(f"Unknown phase: {phase}")
