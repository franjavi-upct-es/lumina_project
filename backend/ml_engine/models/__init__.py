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

from backend.ml_engine.models.base_model import BaseModel, ModelMetadata
from backend.ml_engine.models.ensemble import EnsembleModel
from backend.ml_engine.models.lstm_advanced import (
    AdvancedLSTM,
    LSTMTrainer,
    TimeSeriesDataset,
)
from backend.ml_engine.models.transformer import (
    TimeSeriesTransformer,
    TransformerDataset,
    TransformerModel,
)
from backend.ml_engine.models.xgboost_model import XGBoostModel

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
