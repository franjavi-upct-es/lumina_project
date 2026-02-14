# backend/simulation/environments/__init__.py
"""
RL Trading Environments Module

Gymnasium-compatible trading environments for RL agent training:
- TradingEnv: Main training environment with configurable rewards
- LiveShadowEnv: Paper trading environment for live testing

Reward Functions:
- Sharpe Ratio: Risk-adjusted returns
- Sortino Ratio: Downside risk focus
- Calmar Ratio: Return / max drawdown

All environments follow the Gymnasium API:
- reset() → (observation, info)
- step(action) → (observation, reward, terminated, truncated, info)
- render() → visual representation (optional)
"""

from backend.simulation.environments.base_env import (
    EnvConfig,
    TradingEnv,
)
from backend.simulation.environments.live_shadow import (
    LiveShadowEnv,
)
from backend.simulation.environments.reward_functions import (
    CalmarReward,
    CompositeReward,
    RewardFunction,
    SharpeReward,
    SortinoReward,
)

__all__ = [
    # Reward Functions
    "RewardFunction",
    "SharpeReward",
    "SortinoReward",
    "CalmarReward",
    "CompositeReward",
    # Environments
    "TradingEnv",
    "EnvConfig",
    "LiveShadowEnv",
]
