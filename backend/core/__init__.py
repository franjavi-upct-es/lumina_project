# backend/core/__init__.py
"""
Core Integration Module for Lumina Quant Lab

This module provides a unified interface to all Lumina components.
Import from here for the most convenient access to all features.

Quick Start:
    from backend.core import (
        # Data Collection
        YFinanceCollector,
        # Feature Engineering
        FeatureEngineer,
        # Machine Learning
        AdvancedLSTM,
        LSTMTrainer,
        TransformerModel,
        # Configuration
        settings,
    )

    # Collect data
    collector = YFinanceCollector()
    data = await collector.collect_with_retry("AAPL", start_date, end_date)

    # Engineer features
    fe = FeatureEngineer()
    enriched_data = fe.create_all_features(data)

    # Train model
    model = AdvancedLSTM(input_dim=50, hidden_dim=128)
    trainer = LSTMTrainer(model)

Full Example Pipeline:
    from backend.core import create_pipeline, run_analysis

    # Run complete analysis pipeline
    results = await run_analysis(
        ticker="AAPL",
        model_type="lstm",
        backtest=True,
        risk_analysis=True
    )
"""

from typing import TYPE_CHECKING

# ============================================================================
# BACKTESTING
# ============================================================================
from backend.backtesting import (
    BaseStrategy,
    EventDrivenBacktest,
    MonteCarloConfig,
    MonteCarloSimulator,
    Portfolio,
    RSIStrategy,
)

# ============================================================================
# CONFIGURATION
# ============================================================================
from backend.config import get_settings, settings

# ============================================================================
# DATA ENGINE
# ============================================================================
from backend.data_engine import (
    BaseDataCollector,
    FeatureEngineer,
    YFinanceCollector,
)

# ============================================================================
# MACHINE LEARNING
# ============================================================================
from backend.ml_engine import (
    AdvancedLSTM,
    BaseModel,
    LSTMTrainer,
    ModelMetadata,
    TimeSeriesDataset,
    TransformerModel,
)

# ============================================================================
# TYPE CHECKING ONLY IMPORTS
# ============================================================================
if TYPE_CHECKING:
    from backend.db import BacktestResult, Feature, Model, PriceData
    from backend.workers import celery_app


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================


async def collect_data(
    ticker: str,
    start_date=None,
    end_date=None,
    with_features: bool = True,
):
    """
    Convenience function to collect and process market data.

    Args:
        ticker: Stock ticker symbol
        start_date: Start date for data collection
        end_date: End date for data collection
        with_features: Whether to compute features

    Returns:
        DataFrame with OHLCV data and optionally features
    """
    collector = YFinanceCollector()
    data = await collector.collect_with_retry(ticker, start_date, end_date)

    if data is None:
        return None

    if with_features:
        fe = FeatureEngineer()
        data = fe.create_all_features(data, add_lags=True, add_rolling=True)

    return data


def create_model(
    model_type: str = "lstm",
    input_dim: int = 50,
    **kwargs,
):
    """
    Factory function to create ML models.

    Args:
        model_type: Type of model ("lstm", "transformer", "xgboost")
        input_dim: Number of input features
        **kwargs: Additional model parameters

    Returns:
        Configured model instance
    """
    if model_type == "lstm":
        return AdvancedLSTM(
            input_dim=input_dim,
            hidden_dim=kwargs.get("hidden_dim", 128),
            num_layers=kwargs.get("num_layers", 3),
            dropout=kwargs.get("dropout", 0.3),
            output_horizon=kwargs.get("output_horizon", 5),
        )
    elif model_type == "transformer":
        return TransformerModel(
            model_name=kwargs.get("model_name", "transformer"),
            hyperparameters={
                "d_model": kwargs.get("d_model", 64),
                "nhead": kwargs.get("nhead", 4),
                "num_encoder_layers": kwargs.get("num_layers", 3),
            },
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def get_feature_list(category: str = None):
    """
    Get list of available features.

    Args:
        category: Optional category filter

    Returns:
        List of feature names
    """
    fe = FeatureEngineer()
    if category:
        return fe.get_feature_names_by_category(category)
    return fe.get_all_feature_names()


# ============================================================================
# VERSION INFO
# ============================================================================
from backend import __version__

__all__ = [
    # Version
    "__version__",
    # Configuration
    "settings",
    "get_settings",
    # Data Collection
    "BaseDataCollector",
    "YFinanceCollector",
    # Feature Engineering
    "FeatureEngineer",
    # Machine Learning
    "BaseModel",
    "ModelMetadata",
    "AdvancedLSTM",
    "LSTMTrainer",
    "TimeSeriesDataset",
    "TransformerModel",
    # Backtesting
    "EventDrivenBacktest",
    "Portfolio",
    "MonteCarloSimulator",
    "MonteCarloConfig",
    "BaseStrategy",
    "RSIStrategy",
    # Convenience functions
    "collect_data",
    "create_model",
    "get_feature_list",
]
