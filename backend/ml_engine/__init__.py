# backend/ml_engine/__init__.py
"""
Machine Learning Engine Module for Lumina Quant Lab

Provides comprehensive ML capabilities for quantitative finance:

Models:
- AdvancedLSTM: Multi-variate LSTM with attention mechanism
- TransformerModel: Temporal Transformer for sequence modeling
- XGBoostModel: Gradient boosting for tabular data
- EnsembleModel: Meta-learner combining multiple models

Training:
- LSTMTrainer: Training utilities for LSTM models
- WalkForwardValidator: Time-series cross-validation
- PurgedKFold: Gap-aware cross-validation
- HyperoptTuner: Bayesian hyperparameter optimization

Evaluation:
- ModelMetrics: Comprehensive evaluation metrics
- SHAPExplainer: Feature importance via SHAP
- ErrorAnalyzer: Error analysis by regime

Features:
- TechnicalFeatures: 100+ technical indicators
- SentimentFeatures: NLP-based features
- FundamentalFeatures: Company fundamentals
- MacroFeatures: Economic indicators

Usage:
    from backend.ml_engine import AdvancedLSTM, LSTMTrainer, TimeSeriesDataset

    # Create model
    model = AdvancedLSTM(
        input_dim=50,
        hidden_dim=128,
        num_layers=3,
        dropout=0.3
    )

    # Create trainer
    trainer = LSTMTrainer(model)

    # Train
    history = trainer.train(train_loader, val_loader, epochs=50)
"""

from importlib import import_module
from typing import Any

# Models
from backend.ml_engine.evaluation import ModelMetrics, SHAPExplainer
from backend.ml_engine.features import SentimentFeatures, TechnicalFeatures
from backend.ml_engine.models import (
    BaseModel,
    EnsembleModel,
    ModelMetadata,
    XGBoostModel,
)
from backend.ml_engine.training import PurgedKFold, WalkForwardValidator

_LAZY_EXPORTS = {
    "AdvancedLSTM": ("backend.ml_engine.models", "AdvancedLSTM"),
    "LSTMTrainer": ("backend.ml_engine.models", "LSTMTrainer"),
    "TimeSeriesDataset": ("backend.ml_engine.models", "TimeSeriesDataset"),
    "TransformerModel": ("backend.ml_engine.models", "TransformerModel"),
    "HyperoptTuner": ("backend.ml_engine.training", "HyperoptTuner"),
}


def __getattr__(name: str) -> Any:
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attribute_name = _LAZY_EXPORTS[name]
    value = getattr(import_module(module_name), attribute_name)
    globals()[name] = value
    return value


__all__ = [
    # Base
    "BaseModel",
    "ModelMetadata",
    # Models
    "AdvancedLSTM",
    "TransformerModel",
    "TimeSeriesDataset",
    "LSTMTrainer",
    "XGBoostModel",
    "EnsembleModel",
    # Training
    "WalkForwardValidator",
    "PurgedKFold",
    "HyperoptTuner",
    # Features
    "TechnicalFeatures",
    "SentimentFeatures",
    # Evaluation
    "ModelMetrics",
    "SHAPExplainer",
]
