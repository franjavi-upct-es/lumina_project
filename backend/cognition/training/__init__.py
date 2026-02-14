# backend/cognition/training/__init__.py
"""
RL Training Module

Implements the training infrastructure for V3 agents:
- Main training loop with episode management
- Curriculum learning (3-phase approach)
- MLflow experiment tracking
- Early stopping and checkpointing
- Evaluation and metrics

Training Philosophy:
Lumina V3 follows a "Gladiator School" approach to training:
1. Phase A (Apprentice): Behavioral cloning from V2 logic
2. Phase B (Matrix): Domain randomization with adversarial scenarios
3. Phase C (Master): Pure RL for Sharpe ratio maximization

Each phase builds on the previous, creating a robust agent that can
handle market conditions never seen in historical data.
"""

from backend.cognition.training.curriculum import (
    CurriculumScheduler,
    PhaseConfig,
    TrainingPhase,
)
from backend.cognition.training.trainer import (
    RLTrainer,
    TrainingConfig,
    TrainingMetrics,
)

__all__ = [
    # Curriculum learning
    "CurriculumScheduler",
    "TrainingPhase",
    "PhaseConfig",
    # Training
    "RLTrainer",
    "TrainingConfig",
    "TrainingMetrics",
]
