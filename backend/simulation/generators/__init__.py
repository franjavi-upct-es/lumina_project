# backend/simulation/generators/__init__.py
"""
Scenario Generators Module

Provides synthetic data generation and adversarial scenario creation
for robust RL training (Phase B - Domain Randomization).

Components:
- SyntheticDataGenerator: Generate synthetic OHLCV data with controllable properties
- AdversarialScenarioGenerator: Create "nightmare scenarios" to stress-test agents
- ScenarioLoader: Load and replay historical market crashes and anomalies

Phase B Training Philosophy:
"We don't just replay history. We warp it to create synthetic realities."
- Warp 1 (Volatility): 2x, 3x, 5x volatility multipliers
- Warp 2 (Noise): Spread widening, slippage spikes
- Warp 3 (Blackout): Data feed outages, missing candles

The agent learns that "stability is a privilege, not a right."
"""

from backend.simulation.generators.adversarial import (
    AdversarialScenarioGenerator,
    NightmareScenario,
    ScenarioType,
)
from backend.simulation.generators.scenario_loader import (
    HistoricalCrash,
    ScenarioLoader,
)
from backend.simulation.generators.synthetic_data import (
    GBMGenerator,
    JumpDiffusionGenerator,
    SyntheticDataGenerator,
)

__all__ = [
    # Synthetic Data
    "SyntheticDataGenerator",
    "GBMGenerator",
    "JumpDiffusionGenerator",
    # Adversarial
    "AdversarialScenarioGenerator",
    "NightmareScenario",
    "ScenarioType",
    # Historical
    "ScenarioLoader",
    "HistoricalCrash",
]
