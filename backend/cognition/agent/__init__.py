# backend/cognition/agent/__init__.py
"""
Reinforcement Learning Agent Module

This module contains the RL agents for the V3 Chimera architecture:
- PPO (Proximal Policy Optimization): Stable, general-purpose agent
- SAC (Soft Actor-Critic): Sample-efficient with entropy maximization
- Uncertainty estimation for epistemic risk

V3 Architecture:
All agents operate in a continuous action space with 4-dimensional output:
- Action[0] (Direction): -1.0 (Full Short) to 1.0 (Full Long)
- Action[1] (Urgency): 0.0 (Limit Order) to 1.0 (Market Order)
- Action[2] (Sizing): Position size as fraction of capital (0.0 to 1.0)
- Action[3] (Stop-Distance): Stop loss relative to ATR (0.0 to 1.0)

The agents are trained using multi-modal state representations from the
fusion layer (224-dim super-state combining TFT, BERT, and GNN embeddings).
"""

from backend.cognition.agent.ppo_continuous import PPOContinuousAgent
from backend.cognition.agent.sac_agent import SACAgent
from backend.cognition.agent.uncertainty import (
    DeepEnsemble,
    MonteCarloDropout,
    UncertaintyEstimator,
)

__all__ = [
    # Agents
    "PPOContinuousAgent",
    "SACAgent",
    # Uncertainty estimation
    "MonteCarloDropout",
    "UncertaintyEstimator",
    "DeepEnsemble",
]
