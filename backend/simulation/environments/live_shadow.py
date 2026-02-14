# backend/simulation/environments/live_shadow.py
"""
Live Shadow Environment - Paper Trading

Shadow trading environment that mirrors live market conditions
without executing real trades.

Used for:
- Paper trading validation
- Live agent testing before deployment
- Performance monitoring in production conditions

Connects to live data feeds but uses simulated execution.
"""

from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from backend.simulation.environments.base_env import EnvConfig, TradingEnv
from backend.simulation.environments.reward_functions import RewardFunction


class LiveShadowEnv(TradingEnv):
    """
    Paper trading environment mirroring live conditions.

    Extends TradingEnv with:
    - Live data feed integration
    - Realistic slippage modeling
    - Market hours awareness
    - Order book simulation

    Example:
        >>> env = LiveShadowEnv(data_feed=live_feed)
        >>> obs, info = env.reset()
        >>> # Runs in real-time, synced with market
        >>> while market_open():
        >>>     action = agent.select_action(obs)
        >>>     obs, reward, done, truncated, info = env.step(action)
    """

    def __init__(
        self,
        df: pd.DataFrame | None = None,
        config: EnvConfig | None = None,
        reward_fn: RewardFunction | None = None,
        live_mode: bool = False,
    ):
        """
        Initialize live shadow environment.

        Args:
            df: Historical data (optional if live_mode=True)
            config: Environment configuration
            reward_fn: Reward function
            live_mode: If True, connects to live data feed
        """
        super().__init__(df=df, config=config, reward_fn=reward_fn)

        self.live_mode = live_mode
        self.last_update_time = None

        # Paper trading specific tracking
        self.paper_orders = []
        self.paper_fills = []
        self.slippage_realized = []

        logger.info(f"LiveShadowEnv initialized: live_mode={live_mode}")

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        """
        Execute step with realistic market conditions.

        Adds:
        - Realistic slippage based on urgency
        - Order fill simulation
        - Market impact modeling
        """
        # In live mode, wait for next market tick
        if self.live_mode:
            self._wait_for_next_tick()

        # Get current market conditions
        market_conditions = self._get_market_conditions()

        # Adjust action for market conditions
        adjusted_action = self._adjust_for_market_conditions(action, market_conditions)

        # Execute parent step with adjusted action
        obs, reward, terminated, truncated, info = super().step(adjusted_action)

        # Add shadow trading specific info
        info["paper_trading"] = True
        info["market_conditions"] = market_conditions
        info["slippage_realized"] = self.slippage_realized[-1] if self.slippage_realized else 0.0

        return obs, reward, terminated, truncated, info

    def _wait_for_next_tick(self):
        """Wait for next market data tick (live mode only)."""
        # In production, this would wait for actual market data
        # For now, just track timing
        self.last_update_time = datetime.now()

    def _get_market_conditions(self) -> dict[str, float]:
        """
        Get current market conditions.

        Returns:
            dictionary with market metrics
        """
        if self.current_step >= len(self.df):
            return {"volatility": 0.02, "spread": 0.001, "volume": 1000000, "liquidity": 1.0}

        current_data = self.df.iloc[self.current_step]

        # Calculate volatility (simple range-based)
        if "high" in current_data and "low" in current_data:
            price_range = current_data["high"] - current_data["low"]
            volatility = price_range / current_data["close"]
        else:
            volatility = 0.02

        # Estimate spread from price
        spread = current_data["close"] * 0.001  # 10 bps default

        # Volume
        volume = current_data.get("volume", 1000000)

        # Liquidity score (0 to 1)
        liquidity = min(volume / 10000000, 1.0)

        return {
            "volatility": volatility,
            "spread": spread,
            "volume": volume,
            "liquidity": liquidity,
        }

    def _adjust_for_market_conditions(
        self, action: np.ndarray, conditions: dict[str, float]
    ) -> np.ndarray:
        """
        Adjust action based on market conditions.

        Models realistic constraints:
        - High urgency → more slippage
        - Low liquidity → size constraints
        - High volatility → wider spreads
        """
        adjusted = action.copy()

        direction, urgency, sizing, stop_dist = action

        # Adjust sizing based on liquidity
        liquidity = conditions["liquidity"]
        if liquidity < 0.5:
            # Reduce size in low liquidity
            adjusted[2] = sizing * liquidity

        # Calculate realized slippage
        base_slippage = self.config.slippage
        volatility_factor = 1 + (conditions["volatility"] / 0.02)  # Scale by volatility
        urgency_factor = 1 + urgency  # More urgency = more slippage

        realized_slippage = base_slippage * volatility_factor * urgency_factor
        self.slippage_realized.append(realized_slippage)

        return adjusted

    def get_paper_trading_summary(self) -> dict[str, Any]:
        """
        Get summary of paper trading performance.

        Returns:
            dictionary with paper trading metrics
        """
        if len(self.portfolio_history) < 2:
            return {}

        portfolio_array = np.array(self.portfolio_history)
        returns_array = np.array(self.returns_history) if self.returns_history else np.array([])

        summary = {
            "total_steps": self.current_step,
            "initial_capital": self.config.initial_capital,
            "final_capital": self.portfolio_value,
            "total_return": (self.portfolio_value - self.config.initial_capital)
            / self.config.initial_capital,
            "num_trades": len(self.actions_history),
        }

        if len(returns_array) > 0:
            from backend.utils.calculations import (
                calculate_max_drawdown,
                calculate_sharpe_ratio,
                calculate_volatility,
            )

            summary["sharpe_ratio"] = calculate_sharpe_ratio(returns_array)
            summary["max_drawdown"] = calculate_max_drawdown(portfolio_array)
            summary["volatility"] = calculate_volatility(returns_array)

        if self.slippage_realized:
            summary["avg_slippage"] = np.mean(self.slippage_realized)
            summary["total_slippage_cost"] = (
                sum(self.slippage_realized) * self.config.initial_capital
            )

        return summary

    def reset_paper_trading(self):
        """Reset paper trading state while preserving configuration."""
        self.paper_orders = []
        self.paper_fills = []
        self.slippage_realized = []

        return self.reset()
