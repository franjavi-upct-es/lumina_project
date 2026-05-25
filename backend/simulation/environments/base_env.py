# backend/simulation/environments/base_env.py
"""Gymnasium-compatible trading environment with a 4-D continuous action space.

This is the canonical training/evaluation environment for the Lumina V3
PPO/SAC agent. It complies with the Gymnasium API (``reset()`` and
``step()``) so any standard RL library can be used.

Action vocabulary (must match Lumina_V3_Deep_Fusion_Architecture.md §5)
----------------------------------------------------------------------
    action[0] = direction       in [-1, 1]
        Target portfolio fraction. -1 = full short, 0 = flat, +1 = full long.
    action[1] = urgency         in [-1, 1]
        Order aggressiveness. < 0 → limit/maker, > 0 → market/taker.
        We approximate the effect by scaling slippage:
            slippage_bps = base_slippage * (1 + max(urgency, 0))
    action[2] = sizing          in [-1, 1]
        Multiplier on the |direction|. We map (-1, 1) → (0, 1) via
        ``size_factor = (action[2] + 1) / 2`` so that sizing=0 keeps
        full direction and sizing=-1 zeroes it.
    action[3] = stop_distance   in [-1, 1]
        ATR-multiplier for the trailing stop, mapped to [0.5, 4.0]:
            stop_atr = 2.25 + 1.75 * action[3]
        Tight stops (low ATR multiplier) trigger easier; wide stops
        give the trade more room.

Observation vocabulary
----------------------
    obs[0:NEXUS_OUTPUT_DIM]                       = market_state from fusion
    obs[NEXUS_OUTPUT_DIM]   = current_position    in [-1, 1]
    obs[NEXUS_OUTPUT_DIM+1] = equity_normalised   = equity / initial_capital
    obs[NEXUS_OUTPUT_DIM+2] = current_drawdown    in [0, 1]
    obs[NEXUS_OUTPUT_DIM+3] = episode_progress    in [0, 1]

Reward function
---------------
We use a Sharpe-shaped reward (the "Sharpe-style" intermediate variant
from Moody & Saffell, 2001):

    r_t = SCALING * (pnl_t / capital) − λ_risk · |position_t| · vol_t

where ``vol_t`` is a recent realised volatility proxy. The full Sharpe
ratio is computed at episode end and used for evaluation/logging only.
"""

from __future__ import annotations

from dataclasses import dataclass

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from backend.config.constants import ACTION_DIM, NEXUS_OUTPUT_DIM
from backend.execution.safety.arbitrator import SafetyArbitrator
from backend.execution.safety.rules import SafetyContext


@dataclass
class EnvConfig:
    """Trading-environment hyper-parameters.

    All bookkeeping is in dollar-equivalent units; positions are expressed
    as fractions of equity in [-1, 1].
    """

    initial_capital: float = 100_000.0
    base_slippage_bps: float = 2.0
    """Slippage when ``urgency = 0``. Doubles when urgency = +1."""
    commission_bps: float = 1.0
    max_drawdown_pct: float = 0.20
    """Episode terminates if drawdown exceeds this — corresponds to the
    'hard kill' that the milestone test explicitly forbids the agent
    from triggering during the 2020-crisis drill."""
    reward_scaling: float = 100.0
    risk_penalty_coef: float = 0.1
    veto_penalty: float = 5.0
    """Penalty subtracted from reward whenever the Safety Arbitrator vetoes."""
    portfolio_state_dim: int = 4
    """Number of bookkeeping features appended to the market_state."""


class LuminaTradingEnv(gym.Env):
    """Single-asset trading environment for the Chimera agent.

    Parameters
    ----------
    episode_generator : iterator
        Object yielding dicts with keys:
            'prices'        : (T,) float
            'market_states' : (T, NEXUS_OUTPUT_DIM) float
            'volatility'    : (T,) float
            'uncertainties' : (T,) float
        Provided by ``backend.simulation.generators.scenario_loader`` or
        ``adversarial.AdversarialGenerator``.
    """

    metadata: dict[str, list[str]] = {"render_modes": []}

    def __init__(
        self,
        episode_generator,
        config: EnvConfig | None = None,
        arbitrator: SafetyArbitrator | None = None,
    ):
        super().__init__()
        self.gen = episode_generator
        self.config = config or EnvConfig()
        # Injected arbitrator (e.g. for Spartan curriculum Phase B)
        self.arbitrator = arbitrator or SafetyArbitrator()
        self.obs_dim = NEXUS_OUTPUT_DIM + self.config.portfolio_state_dim
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.obs_dim,),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(ACTION_DIM,),
            dtype=np.float32,
        )
        # Episode state
        self._episode: dict | None = None
        self._t: int = 0
        self._position: float = 0.0
        self._stop_price: float | None = None
        self._equity: float = self.config.initial_capital
        self._peak_equity: float = self.config.initial_capital
        self._n_trades: int = 0

    # ------------------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._episode = next(iter(self.gen))
        self._t = 0
        self._position = 0.0
        self._stop_price = None
        self._equity = self.config.initial_capital
        self._peak_equity = self.config.initial_capital
        self._n_trades = 0
        return self._build_obs(), {}

    # ------------------------------------------------------------------
    def _build_obs(self) -> np.ndarray:
        ep = self._episode
        assert ep is not None
        market_state = ep["market_states"][self._t].astype(np.float32)
        n_steps = len(ep["prices"])
        portfolio = np.array(
            [
                self._position,
                self._equity / self.config.initial_capital,
                1.0 - self._equity / max(self._peak_equity, 1e-6),
                self._t / max(n_steps - 1, 1),
            ],
            dtype=np.float32,
        )
        return np.concatenate([market_state, portfolio])

    # ------------------------------------------------------------------
    @staticmethod
    def _decode_action(action: np.ndarray) -> tuple[float, float, float, float]:
        """Translate the raw 4-D action vector into env-level controls."""
        direction = float(np.clip(action[0], -1.0, 1.0))
        urgency = float(np.clip(action[1], -1.0, 1.0))
        size_factor = (float(np.clip(action[2], -1.0, 1.0)) + 1.0) * 0.5  # [0, 1]
        stop_atr = 2.25 + 1.75 * float(np.clip(action[3], -1.0, 1.0))  # [0.5, 4]
        return direction, urgency, size_factor, stop_atr

    # ------------------------------------------------------------------
    def step(self, action: np.ndarray):
        ep = self._episode
        assert ep is not None

        # --- Safety Arbitration (Live-alignment) -----------------------
        uncertainty = float(ep["uncertainties"][min(self._t, len(ep["uncertainties"]) - 1)])
        # Current position is needed by rules; build a minimal context.
        ctx = SafetyContext(
            proposed_action=action,
            current_position=self._position,
            equity=self._equity,
            peak_equity=self._peak_equity,
            uncertainty=uncertainty,
            kill_switch_state="NORMAL",  # Always normal in training unless forced
        )
        decision = self.arbitrator.evaluate(ctx)

        # If vetoed, we replace the action with the arbitrator's safe target.
        # Note: the decoder is still called so we can simulate the trade.
        if not decision.approved:
            # Override target direction in the raw action for trade simulation.
            # We assume direction=0 sizing=0 (flat) from the arbitrator.
            action = action.copy()
            action[0] = 0.0
            action[2] = -1.0  # factor -> 0

        direction, urgency, size_factor, stop_atr = self._decode_action(action)
        target_position = direction * size_factor

        price_now = float(ep["prices"][self._t])
        next_t = min(self._t + 1, len(ep["prices"]) - 1)
        price_next = float(ep["prices"][next_t])
        recent_vol = float(
            ep.get("volatility", [0.01])[min(self._t, len(ep.get("volatility", [0.01])) - 1)]
        )

        # --- Trade execution -------------------------------------------
        delta = target_position - self._position
        slippage = self.config.base_slippage_bps * (1.0 + max(urgency, 0.0)) / 1e4
        commission = self.config.commission_bps / 1e4
        traded_notional = abs(delta) * self._equity
        cost = traded_notional * (slippage + commission)
        if abs(delta) > 1e-6:
            self._n_trades += 1

        # Apply trade
        self._position = target_position
        # Update stop price
        if self._position > 0:
            self._stop_price = price_now * (1.0 - stop_atr * recent_vol)
        elif self._position < 0:
            self._stop_price = price_now * (1.0 + stop_atr * recent_vol)
        else:
            self._stop_price = None

        # --- PnL --------------------------------------------------------
        ret = (price_next - price_now) / price_now
        pnl = self._equity * self._position * ret
        self._equity += pnl - cost
        self._peak_equity = max(self._peak_equity, self._equity)

        # --- Stop-loss check -------------------------------------------
        if self._stop_price is not None:
            if self._position > 0 and price_next <= self._stop_price:
                # Force close: realise loss at stop price
                self._position = 0.0
                self._stop_price = None
            elif self._position < 0 and price_next >= self._stop_price:
                self._position = 0.0
                self._stop_price = None

        # --- Reward -----------------------------------------------------
        risk_pen = self.config.risk_penalty_coef * abs(self._position) * recent_vol
        reward = self.config.reward_scaling * (pnl / self.config.initial_capital) - risk_pen

        # Apply Arbitrator penalty if vetoed
        if not decision.approved:
            reward -= self.config.veto_penalty

        # --- Termination conditions ------------------------------------
        drawdown = 1.0 - self._equity / self._peak_equity
        terminated = drawdown > self.config.max_drawdown_pct
        truncated = next_t >= len(ep["prices"]) - 1

        self._t = next_t
        info = {
            "equity": self._equity,
            "position": self._position,
            "pnl": pnl,
            "cost": cost,
            "drawdown": drawdown,
            "n_trades": self._n_trades,
            "stop_price": self._stop_price,
            "uncertainty": uncertainty,
            "vetoed": not decision.approved,
        }
        return self._build_obs(), reward, terminated, truncated, info
