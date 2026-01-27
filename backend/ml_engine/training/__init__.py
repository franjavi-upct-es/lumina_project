# backend/ml_engine/training/__init__.py
"""
ML Training Module for Lumina Quant Lab

Provides training utilities and cross-validation strategies:

Trainer:
- Generic model training with callbacks
- Early stopping
- Learning rate scheduling
- Gradient clipping

WalkForwardValidator:
- Time-series aware cross-validation
- Expanding window validation
- Rolling window validation

PurgedKFold:
- Gap-aware cross-validation
- Prevents data leakage
- Embargo periods

HyperoptTuner:
- Bayesian hyperparameter optimization
- Grid search
- Random search
- Integration with Optuna

Usage:
    from backend.ml_engine.training import WalkForwardValidator

    validator = WalkForwardValidator(
        n_splits=5,
        gap=5,
        expanding=True
    )

    for train_idx, val_idx in validator.split(data):
        # Train and validate
        pass
"""

from backend.ml_engine.training.hyperopt_tuner import HyperoptTuner
from backend.ml_engine.training.purged_cv import PurgedKFold
from backend.ml_engine.training.trainer import Trainer
from backend.ml_engine.training.walk_forward import WalkForwardValidator

__all__ = [
    "Trainer",
    "WalkForwardValidator",
    "PurgedKFold",
    "HyperoptTuner",
]
