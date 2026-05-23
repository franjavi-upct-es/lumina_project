# backend/cognition/policy/networks.py
"""Actor-Critic neural networks for the PPO/SAC agent.

This module defines two PyTorch nn.Module classes:

  * ``ActorNetwork``  — produces the parameters of the action distribution
                        (a 4-D continuous vector) plus a Monte-Carlo Dropout
                        layer that the Uncertainty Gate exploits.
  * ``CriticNetwork`` — V(s) value-function estimator. PPO uses a *clipped*
                        value loss; SAC uses two Q-networks instead, so we
                        also expose a ``QCriticNetwork`` with twin heads.

Architectural commitments (Lumina_V3_Deep_Fusion_Architecture.md §5)
--------------------------------------------------------------------
* Continuous 4-D action vector ``[direction, urgency, sizing, stop_distance]``
  in ``[-1, 1]``.
* MC-Dropout layers stay ON at *inference* time when the Uncertainty Gate
  collects N=10 stochastic forward passes (Gal & Ghahramani, 2016).
* Trunk size and depth chosen to keep the forward pass below 10 ms on a
  modern GPU; this is the budget assigned in the architecture spec to
  the "Reflex Arc" (§6).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from backend.config.constants import ACTION_DIM, NEXUS_OUTPUT_DIM


class _MLPTrunk(nn.Module):
    """Two-layer MLP with GELU + LayerNorm + Dropout. Shared by all heads.

    GELU is preferred over ReLU because the Cross-Modal Attention block in
    the fusion stage is also Transformer-flavoured; using a consistent
    nonlinearity makes loss-landscape behaviour more predictable.
    LayerNorm (instead of BatchNorm) avoids the dependence on batch
    statistics that breaks down at single-sample inference time.
    """

    def __init__(self, input_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ActorNetwork(nn.Module):
    """Policy network emitting Gaussian or Beta parameters for 4-D actions.

    Parameters
    ----------
    state_dim : int
        Dimension of the latent state coming from the Deep Fusion Nexus
        (default = ``NEXUS_OUTPUT_DIM`` = 256).
    hidden_dim : int
        Width of the MLP trunk. 256 is a good default; matches the latent
        state dimension to avoid information bottlenecks.
    action_dim : int
        4 by spec. Kept as a parameter only for unit-testing ergonomics.
    dropout : float
        Probability of MC-Dropout. Held at 0.2 because it is the value the
        Uncertainty Gate threshold (0.85) was calibrated against.
    log_std_init : float
        Initial value of the learnable per-dimension log-std. -0.5 maps to
        a std of ~0.61, which gives reasonable exploration at the start of
        training without exploding the policy gradient.
    distribution : {"gaussian", "beta"}
        Choice of action distribution. Beta needs *two* heads (α and β raw);
        Gaussian needs *one* (mean) plus a learnable log-std parameter.
    """

    def __init__(
        self,
        state_dim: int = NEXUS_OUTPUT_DIM,
        hidden_dim: int = 256,
        action_dim: int = ACTION_DIM,
        dropout: float = 0.2,
        log_std_init: float = -0.5,
        log_std_min: float = -5.0,
        log_std_max: float = 2.0,
        distribution: str = "gaussian",
    ):
        super().__init__()
        if distribution not in ("gaussian", "beta"):
            raise ValueError(f"distribution must be 'gaussian' or 'beta', got {distribution}")
        self.distribution = distribution
        self.action_dim = action_dim
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.trunk = _MLPTrunk(state_dim, hidden_dim, dropout)

        if distribution == "gaussian":
            self.mean_head = nn.Linear(hidden_dim, action_dim)
            self.log_std = nn.Parameter(torch.full((action_dim,), log_std_init))
        else:
            self.alpha_head = nn.Linear(hidden_dim, action_dim)
            self.beta_head = nn.Linear(hidden_dim, action_dim)

    def forward(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Run a forward pass and return distribution parameters.

        Returns
        -------
        For Gaussian: (mean, log_std_clamped)
        For Beta:    (alpha_raw, beta_raw)  — the consumer applies softplus+1.
        """
        h = self.trunk(state)
        if self.distribution == "gaussian":
            mean = self.mean_head(h)
            log_std = self.log_std.clamp(self.log_std_min, self.log_std_max)
            return mean, log_std
        return self.alpha_head(h), self.beta_head(h)


class CriticNetwork(nn.Module):
    """Plain V(s) value head used by PPO."""

    def __init__(
        self,
        state_dim: int = NEXUS_OUTPUT_DIM,
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.trunk = _MLPTrunk(state_dim, hidden_dim, dropout)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.head(self.trunk(state)).squeeze(-1)


class QCriticNetwork(nn.Module):
    """Twin Q-networks for SAC.

    SAC uses ``min(Q1, Q2)`` to mitigate over-estimation bias
    (Fujimoto et al., TD3, 2018). Each head receives the concatenation
    ``[state, action]`` so that Q : (S × A) → R.
    """

    def __init__(
        self,
        state_dim: int = NEXUS_OUTPUT_DIM,
        action_dim: int = ACTION_DIM,
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        in_dim = state_dim + action_dim
        self.q1 = nn.Sequential(_MLPTrunk(in_dim, hidden_dim, dropout), nn.Linear(hidden_dim, 1))
        self.q2 = nn.Sequential(_MLPTrunk(in_dim, hidden_dim, dropout), nn.Linear(hidden_dim, 1))

    def forward(
        self, state: torch.Tensor, action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sa = torch.cat([state, action], dim=-1)
        return self.q1(sa).squeeze(-1), self.q2(sa).squeeze(-1)
