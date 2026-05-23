# backend/cognition/training/domain_randomization.py
"""Domain randomisation runner — Spartan Phase B.

This module orchestrates the second of the three curriculum stages
described in section 7B of ``Lumina_V3_Deep_Fusion_Architecture.md``
("The Spartan Forge"). After Behavioural Cloning has produced a
warm-started policy, we expose the agent to a stream of episodes that
are *randomly* a mix of:

* Clean historical episodes (70% by default)
* Adversarially warped episodes (30%) — one of six perturbation types
  drawn uniformly at random.

The agent uses PPO updates after each completed episode. We track a
rolling 100-episode mean reward; if that mean does not reach the
configured threshold by the end of training, the stage is declared
failed and the curriculum aborts (the ``SpartanCurriculum`` orchestrator
raises ``RuntimeError`` upstream).

Why the 70/30 mix?
==================
A 50/50 mix tends to over-emphasise the warps and the agent's policy
collapses to "always go flat". A 70/30 mix has been empirically robust
in our prototypes; the exact value should be re-tuned once we have real
encoder checkpoints.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import mlflow
import numpy as np
from loguru import logger

from backend.cognition.agent.ppo_agent import PPOAgent
from backend.simulation.environments.base_env import LuminaTradingEnv
from backend.simulation.generators.adversarial import AdversarialGenerator


@dataclass
class DRConfig:
    """Hyper-parameters for the domain-randomisation stage."""

    adversarial_fraction: float = 0.30
    """Probability of drawing an adversarial episode per reset."""
    update_every_n_steps: int = 2048
    """PPO update cadence. 2048 matches the PPO paper's default."""
    rolling_window: int = 100
    """How many recent episode rewards to average for the acceptance gate."""


class DomainRandomizationRunner:
    """Coordinates PPO training against a mix of clean and warped episodes.

    Parameters
    ----------
    env : LuminaTradingEnv
        The Gymnasium environment. Its ``gen`` attribute is mutated each
        episode to point at the appropriate generator.
    clean_generator : iterator
        Yields clean historical episode dicts.
    adversarial_generator : AdversarialGenerator
        Yields warped episode dicts.
    config : DRConfig | None
    rng : numpy.random.Generator | None
    """

    def __init__(
        self,
        env: LuminaTradingEnv,
        clean_generator,
        adversarial_generator: AdversarialGenerator,
        config: DRConfig | None = None,
        rng: np.random.Generator | None = None,
    ):
        self.env = env
        self.clean = clean_generator
        self.adv = adversarial_generator
        self.config = config or DRConfig()
        self.rng = rng or np.random.default_rng()

    # ------------------------------------------------------------------
    def _select_generator(self):
        """Choose between clean and adversarial generators for this episode."""
        if self.rng.random() < self.config.adversarial_fraction:
            return self.adv
        return self.clean

    # ------------------------------------------------------------------
    def run(self, agent: PPOAgent, episodes: int) -> dict[str, float]:
        """Train the agent for ``episodes`` full episodes.

        Returns a dictionary of summary metrics suitable for MLflow:

            mean_reward    — average of the last ``rolling_window`` rewards
            std_reward     — standard deviation of the same window
            n_episodes     — number of completed episodes
            n_updates      — number of PPO update calls performed

        The acceptance gate (mean_reward ≥ 10 by default) is checked by
        the caller via :class:`SpartanCurriculum`.
        """
        rolling_rewards: deque[float] = deque(maxlen=self.config.rolling_window)
        n_updates = 0
        steps_since_update = 0

        for ep_idx in range(episodes):
            self.env.gen = self._select_generator()
            obs, _info = self.env.reset()
            episode_reward = 0.0
            done = False
            while not done:
                action, log_prob, value, uncertainty, _vetoed = agent.act(obs)
                next_obs, reward, terminated, truncated, _info = self.env.step(action)
                done = bool(terminated or truncated)
                agent.record(
                    state=obs,
                    action=action,
                    log_prob=log_prob,
                    value=value,
                    reward=reward,
                    done=done,
                    uncertainty=uncertainty,
                )
                obs = next_obs
                episode_reward += reward
                steps_since_update += 1
                if steps_since_update >= self.config.update_every_n_steps:
                    metrics = agent.update(last_value=value)
                    n_updates += 1
                    steps_since_update = 0
                    if "skipped" not in metrics:
                        mlflow.log_metrics(
                            {
                                f"dr_{k}": v
                                for k, v in metrics.items()
                                if isinstance(v, (int, float))
                            },
                            step=n_updates,
                        )
            rolling_rewards.append(episode_reward)
            if (ep_idx + 1) % 10 == 0:
                mean_r = float(np.mean(rolling_rewards))
                logger.info(f"DR ep {ep_idx + 1}/{episodes}: rolling mean R={mean_r:.2f}")
                mlflow.log_metric("dr_rolling_reward", mean_r, step=ep_idx + 1)

        # Final flush of any remaining transitions.
        if steps_since_update > 0:
            agent.update(last_value=0.0)
            n_updates += 1

        rewards = list(rolling_rewards)
        return {
            "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
            "std_reward": float(np.std(rewards)) if rewards else 0.0,
            "n_episodes": float(len(rewards)),
            "n_updates": float(n_updates),
        }
