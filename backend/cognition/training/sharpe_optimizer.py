# backend/cognition/training/sharpe_optimizer.py
"""Sharpe-ratio fine-tuner — Spartan Phase C.

In the previous phase the agent maximised raw episode reward (basically
total PnL). That is not what we ultimately care about: a strategy with
high mean return but huge variance is unusable in production. Phase C
shifts the optimisation target from PnL to risk-adjusted return by
swapping the environment's reward function for an *incremental Sharpe*
estimator (Moody & Saffell, 2001).

Specifically the reward signal becomes

    Δ Sharpe_t ≈ (r_t − μ̂_t) / σ̂_t

with exponentially-decaying μ̂_t and σ̂_t. This rewards the policy for
producing returns that are *consistent* relative to their own volatility.

Entropy annealing
=================
We cosine-anneal ``entropy_coef`` from 0.01 → 0.001 over the full
training run. In Phase B exploration is essential; in Phase C we want
the policy to commit to the strategies it has discovered. Annealing
preserves the early exploration without paying its cost forever.

Acceptance gate
===============
Annualised out-of-sample Sharpe ≥ 1.0, measured on a held-out *clean*
set every 50 episodes. The best checkpoint is persisted to
``models/agent/best_sharpe.pt``.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import mlflow
import numpy as np
import torch
from loguru import logger

from backend.cognition.agent.ppo_agent import PPOAgent
from backend.simulation.environments.base_env import LuminaTradingEnv
from backend.simulation.environments.reward_functions import IncrementalSharpe


@dataclass
class SharpeOptimizerConfig:
    eval_every_n_episodes: int = 50
    eval_episodes: int = 10
    entropy_start: float = 0.01
    entropy_end: float = 0.001
    checkpoint_path: Path = Path("models/agent/best_sharpe.pt")
    annualisation_periods: float = 252.0 * 390.0
    """Number of decision steps per year. NYSE: 252 trading days × 390 minutes."""


class SharpeOptimizer:
    """Phase-C trainer. See module docstring for the design rationale."""

    def __init__(
        self,
        env: LuminaTradingEnv,
        eval_generator,
        config: SharpeOptimizerConfig | None = None,
    ):
        self.env = env
        self.eval_gen = eval_generator
        self.config = config or SharpeOptimizerConfig()
        self.config.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    @staticmethod
    def _cosine_decay(start: float, end: float, step: int, total: int) -> float:
        """Cosine decay from ``start`` to ``end`` over ``total`` steps."""
        if total <= 0:
            return end
        progress = min(1.0, max(0.0, step / total))
        return end + 0.5 * (start - end) * (1.0 + math.cos(math.pi * progress))

    # ------------------------------------------------------------------
    def _annualise(self, daily_sharpe: float) -> float:
        return daily_sharpe * math.sqrt(self.config.annualisation_periods)

    # ------------------------------------------------------------------
    def _evaluate(self, agent: PPOAgent) -> float:
        """Run a few held-out episodes and return the annualised Sharpe."""
        episode_sharpes: list[float] = []
        original_gen = self.env.gen
        self.env.gen = self.eval_gen
        try:
            for _ in range(self.config.eval_episodes):
                obs, _info = self.env.reset()
                sharpe_tracker = IncrementalSharpe()
                done = False
                last_sharpe = 0.0
                while not done:
                    action, _lp, _v, _u, _vetoed = agent.act(obs, deterministic=True)
                    obs, reward, terminated, truncated, _info = self.env.step(action)
                    done = bool(terminated or truncated)
                    last_sharpe = sharpe_tracker.update(float(reward))
                episode_sharpes.append(last_sharpe)
        finally:
            self.env.gen = original_gen
        return self._annualise(float(np.mean(episode_sharpes))) if episode_sharpes else 0.0

    # ------------------------------------------------------------------
    def run(self, agent: PPOAgent, episodes: int) -> dict[str, float]:
        """Run the fine-tuning phase."""
        best_oos_sharpe = -math.inf
        rolling_rewards: deque[float] = deque(maxlen=100)
        n_updates = 0

        for ep_idx in range(episodes):
            # Anneal entropy coefficient.
            agent.config.entropy_coef = self._cosine_decay(
                self.config.entropy_start,
                self.config.entropy_end,
                ep_idx,
                episodes,
            )

            obs, _info = self.env.reset()
            sharpe_tracker = IncrementalSharpe()
            done = False
            steps = 0
            while not done:
                action, log_prob, value, uncertainty, _vetoed = agent.act(obs)
                next_obs, reward, terminated, truncated, _info = self.env.step(action)
                shaped_reward = sharpe_tracker.update(float(reward))
                done = bool(terminated or truncated)
                agent.record(
                    state=obs,
                    action=action,
                    log_prob=log_prob,
                    value=value,
                    reward=shaped_reward,
                    done=done,
                    uncertainty=uncertainty,
                )
                obs = next_obs
                steps += 1
            agent.update(last_value=0.0)
            n_updates += 1
            rolling_rewards.append(shaped_reward if rolling_rewards else 0.0)

            if (ep_idx + 1) % self.config.eval_every_n_episodes == 0:
                oos_sharpe = self._evaluate(agent)
                mlflow.log_metric("sharpe_oos", oos_sharpe, step=ep_idx + 1)
                logger.info(f"Sharpe ep {ep_idx + 1}: OOS annualised Sharpe={oos_sharpe:.3f}")
                if oos_sharpe > best_oos_sharpe:
                    best_oos_sharpe = oos_sharpe
                    torch.save(agent.policy.state_dict(), self.config.checkpoint_path)
                    logger.success(
                        f"New best checkpoint: Sharpe={oos_sharpe:.3f} "
                        f"saved to {self.config.checkpoint_path}",
                    )

        return {
            "sharpe": float(best_oos_sharpe) if math.isfinite(best_oos_sharpe) else 0.0,
            "n_updates": float(n_updates),
        }
