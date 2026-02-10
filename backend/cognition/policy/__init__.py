# backend/cognition/policy/__init__.py
"""
Policy Networks and Distributions Module

This module contains the neural network architectures and probability distributions
used by RL agents for action selection.

Components:
- Networks: Actor-Critic architectures for PPO and SAC
- Distributions: Probability distributions for continuous actions

V3 Design:
All policies output continuous actions in bounded spaces using appropriate
probability distributions (Gaussian, Beta, Squashed Gaussian).
"""

from backend.cognition.policy.distributions import (
    BoundedNormal,
    DiagonalGaussian,
    SquashedGaussian,
)
from backend.cognition.policy.networks import (
    ActorCriticNetwork,
    SACActorNetwork,
    SACCriticNetwork,
)

__all__ = [
    # Distributions
    "BoundedNormal",
    "SquashedGaussian",
    "DiagonalGaussian",
    # Networks
    "ActorCriticNetwork",
    "SACActorNetwork",
    "SACCriticNetwork",
]
