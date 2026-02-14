# backend/cognition/__init__.py
"""
Cognition Layer - The Brain of Lumina V3

This module implements the cognitive core of the V3 architecture:
- RL Agents (PPO, SAC) for continuous action trading
- Policy networks and distributions
- Uncertainty estimation for safety
- Training infrastructure with curriculum learning

V3 Cognition Architecture:
Layer 3 of the Chimera system - receives fused multi-modal embeddings
from the perception layer and outputs continuous trading actions.

Components:
- agent/: RL agent implementations
- policy/: Neural network architectures and distributions
- training/: Training loops and curriculum learning

The cognition layer operates on a 224-dimensional fused state vector
combining temporal (TFT), semantic (BERT), and structural (GNN) embeddings.

Action Space (4D Continuous):
- Direction: [-1, 1] (Short to Long)
- Urgency: [0, 1] (Limit to Market order)
- Sizing: [0, 1] (Position size fraction)
- Stop-Distance: [0, 1] (Stop loss relative to ATR)
"""

from backend.cognition.agent import (
    MonteCarloDropout,
    PPOContinuousAgent,
    SACAgent,
    UncertaintyEstimator,
)
from backend.cognition.policy import (
    ActorCriticNetwork,
    BoundedNormal,
    DiagonalGaussian,
    SACActorNetwork,
    SACCriticNetwork,
    SquashedGaussian,
)
from backend.cognition.training import (
    CurriculumScheduler,
    RLTrainer,
    TrainingPhase,
)

__version__ = "3.0.0"

__all__ = [
    # Version
    "__version__",
    # Agents
    "PPOContinuousAgent",
    "SACAgent",
    # Uncertainty estimation
    "MonteCarloDropout",
    "UncertaintyEstimator",
    # Policy networks
    "ActorCriticNetwork",
    "SACActorNetwork",
    "SACCriticNetwork",
    # Distributions
    "BoundedNormal",
    "SquashedGaussian",
    "DiagonalGaussian",
    # Training
    "RLTrainer",
    "CurriculumScheduler",
    "TrainingPhase",
]
