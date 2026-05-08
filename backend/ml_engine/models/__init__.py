# backend/ml_engine/models/__init__.py
"""
ML Models Module for Lumina Quant Lab

Contains all machine learning model implementations:

BaseModel:
- Abstract base class for all models
- Common interface for training, prediction, evaluation
- Model serialization and loading
- MLflow integration

AdvancedLSTM:
- Multi-variate LSTM with attention mechanism
- Bidirectional processing
- Multi-task learning (price, volatility, regime)
- Residual connections

TransformerModel:
- Temporal Transformer architecture
- Multi-head self-attention
- Positional encoding for time series
- Layer normalization

XGBoostModel:
- Gradient boosting for feature-rich data
- Feature importance analysis
- Hyperparameter tuning support

EnsembleModel:
- Meta-learner combining multiple models
- Weighted averaging
- Stacking ensemble

Usage:
    from backend.ml_engine.models import AdvancedLSTM, LSTMTrainer

    model = AdvancedLSTM(
        input_dim=50,
        hidden_dim=128,
        num_layers=3
    )

    trainer = LSTMTrainer(model)
    trainer.train(train_loader, val_loader)
"""

from importlib import import_module
from typing import Any

from backend.ml_engine.models.base_model import BaseModel, ModelMetadata
from backend.ml_engine.models.ensemble import EnsembleModel
from backend.ml_engine.models.xgboost_model import XGBoostFinancialModel as XGBoostModel

_LAZY_EXPORTS = {
    "AdvancedLSTM": ("backend.ml_engine.models.lstm_advanced", "AdvancedLSTM"),
    "LSTMTrainer": ("backend.ml_engine.models.lstm_advanced", "LSTMTrainer"),
    "TimeSeriesDataset": ("backend.ml_engine.models.lstm_advanced", "TimeSeriesDataset"),
    "TimeSeriesTransformer": ("backend.ml_engine.models.transformer", "TimeSeriesTransformer"),
    "TransformerDataset": ("backend.ml_engine.models.transformer", "TransformerDataset"),
    "TransformerModel": ("backend.ml_engine.models.transformer", "TransformerFinancialModel"),
}


def __getattr__(name: str) -> Any:
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attribute_name = _LAZY_EXPORTS[name]
    try:
        value = getattr(import_module(module_name), attribute_name)
    except ModuleNotFoundError as exc:
        if exc.name == "torch":
            raise ModuleNotFoundError(
                f"{name} requires PyTorch. Install the ml dependency group with "
                "`uv sync --group ml`."
            ) from exc
        raise

    globals()[name] = value
    return value


__all__ = [
    # Base
    "BaseModel",
    "ModelMetadata",
    # LSTM
    "AdvancedLSTM",
    "LSTMTrainer",
    "TimeSeriesDataset",
    # Transformer
    "TransformerModel",
    "TimeSeriesTransformer",
    "TransformerDataset",
    "XGBoostModel",
    "EnsembleModel",
]
