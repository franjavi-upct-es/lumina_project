# backend/simulation/environments/base_env.py
"""
Base Trading Environment - Gymnasium Compatible

Main RL training environment implementing the Gymnasium API.
Supports continuous action space for V3 agents.

State Space (224d):
- Fused multi-modal embedding from perception + fusion layers
- Temporal (128d) + Semantic (64d) + Structural (32d)

Action Space (4d continuous):
- action[0]: Direction [-1, 1] (Short to Long)
- action[1]: Urgency [0, 1] (Limit to Market)
- action[2]: Sizing [0, 1] (Position fraction)
- action[3]: Stop-Distance [0, 1] (Stop loss ATR multiple)

Follows Gymnasium API:
- reset() → (observation, info)
- step(action) → (observation, reward, terminated, truncated, info)
"""

from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
from loguru import logger

from backend.simulation.environments.reward_functions import RewardFunction, SharpeReward
from backend.utils.calculations import calculate_max_drawdown


@dataclass
class EnvConfig:
    """Trading environment configuration."""

    initial_capital: float = 10000.0
    max_steps: int = 1000
    transaction_cost: float = 0.001  # 0.1%
    slippage: float = 0.0005  # 0.05%
    max_drawdown_limit: float = 0.10  # 10% max drawdown
    daily_loss_limit: float = 0.03  # 3% daily loss
    state_dim: int = 224  # Fused embedding dimension
    action_dim: int = 4  # Continuous action space


class TradingEnv(gym.Env):
    """
    Main trading environment for RL training.

    Example:
        >>> env = TradingEnv(df=price_data, config=EnvConfig())
        >>> obs, info = env.reset()
        >>> action = agent.select_action(obs)
        >>> obs, reward, terminated, truncated, info = env.step(action)
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        df: pd.DataFrame,
        config: EnvConfig | None = None,
        reward_fn: RewardFunction | None = None,
    ):
        """Initialize trading environment."""
        super().__init__()

        self.df = df
        self.config = config or EnvConfig()
        self.reward_fn = reward_fn or SharpeReward()

        # Action and observation spaces
        self.action_space = spaces.Box(
            low=np.array([-1.0, 0.0, 0.0, 0.0]),
            high=np.array([1.0, 1.0, 1.0, 1.0]),
            dtype=np.float32,
        )

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.config.state_dim,), dtype=np.float32
        )

        # State variables
        self.current_step = 0
        self.capital = self.config.initial_capital
        self.position = 0.0
        self.portfolio_value = self.config.initial_capital

        # History tracking
        self.portfolio_history = []
        self.returns_history = []
        self.actions_history = []

        logger.info(f"TradingEnv initialized: {len(df)} steps, reward={self.reward_fn.get_name()}")

    def reset(
        self, seed: int | None = None, options: dict | None = None
    ) -> tuple[np.ndarray, dict]:
        """Reset environment to initial state."""
        super().reset(seed=seed)

        self.current_step = 0
        self.capital = self.config.initial_capital
        self.position = 0.0
        self.portfolio_value = self.config.initial_capital

        self.portfolio_history = [self.portfolio_value]
        self.returns_history = []
        self.actions_history = []

        obs = self._get_observation()
        info = self._get_info()

        return obs, info

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Execute one time step."""
        # Interpret action
        direction = action[0]  # [-1, 1]
        urgency = action[1]  # [0, 1]
        sizing = action[2]  # [0, 1]
        stop_dist = action[3]  # [0, 1]

        # Execute trade
        self._execute_trade(direction, urgency, sizing, stop_dist)

        # Update state
        self.current_step += 1

        # Calculate reward
        reward = self._calculate_reward()

        # Check termination
        terminated = self._check_terminated()
        truncated = self.current_step >= self.config.max_steps

        # Get observation and info
        obs = self._get_observation()
        info = self._get_info()

        return obs, reward, terminated, truncated, info

    def _execute_trade(self, direction: float, urgency: float, sizing: float, stop_dist: float):
        """Execute trade based on action."""
        if self.current_step >= len(self.df):
            return

        current_price = self.df.iloc[self.current_step]["close"]

        # Calculate target position
        target_position = direction * sizing  # [-1, 1]

        # Calculate trade size
        position_change = target_position - self.position

        if abs(position_change) < 0.01:  # Minimum trade size
            return

        # Calculate costs
        transaction_cost = abs(position_change) * self.capital * self.config.transaction_cost
        slippage_cost = abs(position_change) * self.capital * self.config.slippage * urgency

        total_cost = transaction_cost + slippage_cost

        # Update position and capital
        self.position = target_position
        self.capital -= total_cost

        # Update portfolio value
        position_value = self.position * self.capital * current_price
        self.portfolio_value = self.capital + position_value

        # Track
        self.portfolio_history.append(self.portfolio_value)
        if len(self.portfolio_history) > 1:
            ret = (self.portfolio_value - self.portfolio_history[-2]) / self.portfolio_history[-2]
            self.returns_history.append(ret)

        self.actions_history.append([direction, urgency, sizing, stop_dist])

    def _calculate_reward(self) -> float:
        """Calculate reward using reward function."""
        if len(self.returns_history) < 2:
            return 0.0

        returns = np.array(self.returns_history)
        portfolio_values = np.array(self.portfolio_history)

        reward = self.reward_fn.calculate(returns, portfolio_values)

        return reward

    def _check_terminated(self) -> bool:
        """Check if episode should terminate."""
        if len(self.portfolio_history) < 2:
            return False

        # Check max drawdown
        portfolio_array = np.array(self.portfolio_history)
        max_dd = calculate_max_drawdown(portfolio_array)

        if max_dd > self.config.max_drawdown_limit:
            logger.warning(f"Max drawdown limit hit: {max_dd:.2%}")
            return True

        # Check daily loss
        if len(self.portfolio_history) > 1:
            daily_return = (
                self.portfolio_value - self.portfolio_history[-2]
            ) / self.portfolio_history[-2]
            if daily_return < -self.config.daily_loss_limit:
                logger.warning(f"Daily loss limit hit: {daily_return:.2%}")
                return True

        return False

    def _get_observation(self) -> np.ndarray:
        """Get current observation (stub - to be filled by perception layer)."""
        # In production, this would return the fused 224d embedding
        # For now, return random for testing
        return np.random.randn(self.config.state_dim).astype(np.float32)

    def _get_info(self) -> dict[str, Any]:
        """Get info dictionary."""
        info = {
            "step": self.current_step,
            "capital": self.capital,
            "position": self.position,
            "portfolio_value": self.portfolio_value,
        }

        if len(self.portfolio_history) > 1:
            portfolio_array = np.array(self.portfolio_history)
            info["max_drawdown"] = calculate_max_drawdown(portfolio_array)
            info["total_return"] = (
                self.portfolio_value - self.config.initial_capital
            ) / self.config.initial_capital

        return info

    def render(self):
        """Render environment (optional)."""
        if len(self.portfolio_history) > 0:
            print(f"Step: {self.current_step}, Portfolio: ${self.portfolio_value:.2f}")
