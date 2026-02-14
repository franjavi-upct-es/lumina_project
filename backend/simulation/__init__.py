# backend/simulation/__init__.py
"""
Simulation Module - The Gladiator Arena

Provides training environments and scenario generators for RL agents:
- Gymnasium-compatible trading environments
- Reward function implementations
- Adversarial scenario generators (Phase B training)
- Historical crash scenario replays
- Paper trading shadow environment

The simulation module is critical for the three-phase training curriculum:
- Phase A: Behavioral cloning on clean data
- Phase B: Domain randomization with nightmare scenarios
- Phase C: Pure RL on real + generated data

Architecture:
- environments/: Trading environments (Gymnasium API)
- generators/: Synthetic data and adversarial scenario generators
"""

from backend.simulation.environments import (
    CalmarReward,
    LiveShadowEnv,
    SharpeReward,
    SortinoReward,
    TradingEnv,
)
from backend.simulation.generators import (
    AdversarialScenarioGenerator,
    ScenarioLoader,
    SyntheticDataGenerator,
)

__all__ = [
    # Environments
    "TradingEnv",
    "LiveShadowEnv",
    # Reward Functions
    "SharpeReward",
    "SortinoReward",
    "CalmarReward",
    # Generators
    "SyntheticDataGenerator",
    "AdversarialScenarioGenerator",
    "ScenarioLoader",
]
