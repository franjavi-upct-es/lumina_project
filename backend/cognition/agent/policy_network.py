# backend/cognition/agent/policy_network.py
"""High-level policy module: ActorNetwork + action distribution + critic.

This is the *only* class the agent (PPO or SAC) interacts with directly.
It hides the choice of action distribution from the trainer and exposes the
three operations the rest of the system needs:

    sample(state, deterministic) → SampledAction         # for rollout
    evaluate_actions(state, action) → log_prob, entropy  # for PPO updates
    value(state) → V(s)                                  # for advantage est.

The MC-Dropout pathway used by the Uncertainty Gate is the *same* sampler;
we simply call ``sample()`` N times with the trunk in train() mode so dropout
remains active. The std-dev across the N action samples is the gate's input.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from backend.cognition.policy.distributions import (
    SampledAction,
    ScaledBeta,
    SquashedGaussian,
)
from backend.cognition.policy.networks import ActorNetwork, CriticNetwork
from backend.config.constants import ACTION_DIM, NEXUS_OUTPUT_DIM


class PolicyNetwork(nn.Module):
    """Combined actor + critic + distribution wrapper.

    The actor and critic share NO parameters; the architecture spec calls
    for two independent networks because PPO converges more reliably when
    the value function is not coupled to the policy gradient.

    Notes on training-time behaviour
    --------------------------------
    * ``train()`` enables Dropout in the trunks → stochastic forward passes.
    * ``eval()`` disables Dropout → deterministic value estimates.
    * The Uncertainty Gate (``backend.cognition.agent.uncertainty_gate``)
      *temporarily* puts the model in ``train()`` mode to harvest a batch
      of stochastic samples, then restores the previous mode.
    """

    def __init__(
        self,
        state_dim: int = NEXUS_OUTPUT_DIM,
        action_dim: int = ACTION_DIM,
        hidden_dim: int = 256,
        actor_dropout: float = 0.2,
        critic_dropout: float = 0.1,
        distribution: str = "gaussian",
    ):
        super().__init__()
        self.distribution = distribution
        self.actor = ActorNetwork(
            state_dim=state_dim,
            hidden_dim=hidden_dim,
            action_dim=action_dim,
            dropout=actor_dropout,
            distribution=distribution,
        )
        self.critic = CriticNetwork(
            state_dim=state_dim,
            hidden_dim=hidden_dim,
            dropout=critic_dropout,
        )

    # ------------------------------------------------------------------
    # Distribution helpers
    # ------------------------------------------------------------------
    def _make_dist(self, state: torch.Tensor):
        params = self.actor(state)
        if self.distribution == "gaussian":
            mean, log_std = params
            return SquashedGaussian(mean, log_std)
        alpha_raw, beta_raw = params
        return ScaledBeta(alpha_raw, beta_raw)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def sample(
        self, state: torch.Tensor, deterministic: bool = False
    ) -> tuple[SampledAction, torch.Tensor]:
        """Draw an action and compute V(s) in a single forward pass."""
        dist = self._make_dist(state)
        sampled = dist.sample(deterministic=deterministic)
        value = self.critic(state)
        return sampled, value

    def evaluate_actions(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate stored (state, action) pairs during PPO updates.

        Returns
        -------
        log_prob : (B,)        log π_θ(a | s)
        entropy : (B,)         H[π_θ(· | s)] (Gaussian entropy upper bound)
        value   : (B,)         V_φ(s)
        """
        dist = self._make_dist(state)
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        value = self.critic(state)
        return log_prob, entropy, value

    def value(self, state: torch.Tensor) -> torch.Tensor:
        return self.critic(state)
