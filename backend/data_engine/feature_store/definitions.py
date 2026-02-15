# backend/data_engine/feature_store/definitions.py
"""
Feature Definitions and Metadata for Lumina V3
=============================================

Defines the taxonomy of features used by the Chimera architecture.
Features are categorized by their role in the perception layer:

Temporal Features:
    - Price action indicators (RSI, MACD, Bollinger Bands)
    - Volume dynamics
    - Volatility measures (ATR, historical vol)
    - Trend strength indicators

Semantic Features:
    - News sentiment scores
    - Social media sentiment
    - Earnings call transcripts analysis
    - SEC filing analysis

Structural Features:
    - Correlation matrices
    - Graph centrality measures
    - Sector rotation indicators
    - ETF flow dynamics

Macro Features:
    - Interest rates (FRED)
    - Economic indicators (GDP, CPI, Unemployment)
    - VIX and fear indices
    - Commodity prices

Author: Lumina Quant Lab
Version: 3.0.0
"""

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class FeatureCategory(str, Enum):
    """
    Feature taxonomy aligned with V3 perception encoders
    """

    # Temporal Fusion Transformer inputs
    TEMPORAL_PRICE = "temporal_price"  # OHLCV-derived
    TEMPORAL_VOLUME = "temporal_volume"  # Volume patterns
    TEMPORAL_VOLATILITY = "temporal_volatility"  # Volatility measures
    TEMPORAL_MOMENTUM = "temporal_momentum"  # Trend indicators

    # Semantic Encoder inputs
    SEMANTIC_NEWS = "semantic_news"  # News sentiment
    SEMANTIC_SOCIAL = "semantic_social"  # Reddit/Twitter sentiment
    SEMANTIC_FUNDAMENTAL = "semantic_fundamental"  # Earnings, filings

    # Graph Neural Network inputs
    STRUCTURAL_CORRELATION = "structural_correlation"  # Asset correlations
    STRUCTURAL_NETWORK = "structural_network"  # Graph metrics
    STRUCTURAL_SECTOR = "structural_sector"  # Sector dynamics

    # Macro context
    MACRO_RATES = "macro_rates"  # Interest rates
    MACRO_ECONOMIC = "macro_economic"  # Economic indicators
    MACRO_SENTIMENT = "macro_sentiment"  # VIX, fear indices

    # Derived/Engineered
    REGIME = "regime"  # Market regime labels
    INTERACTION = "interaction"  # Feature interactions
    EMBEDDING = "embedding"  # Pre-computed encoder outputs

    # Metadata
    META = "meta"  # Non-predictive metadata


class FeatureDefinition(BaseModel):
    """
    Definition of a single feature with metadata
    """

    name: str = Field(..., description="Feature name (e.g., 'rsi_14')")
    category: FeatureCategory = Field(..., description="Feature category")
    description: str = Field(..., description="Human-readable description")
    dtype: str = Field(default="float32", description="Data type")
    min_value: float | None = Field(None, description="Expected minimum value")
    max_value: float | None = Field(None, description="Expected maximum value")
    normalization: str | None = Field(
        None, description="Normalization method (zscore, minmax, robust, etc.)"
    )
    lookback_period: int | None = Field(None, description="Lookback window in minutes")
    dependencies: list[str] = Field(default_factory=list, description="Required input features")
    encoder_target: str | None = Field(None, description="Which encoder uses this (tft, llm, gnn)")
    is_embedding: bool = Field(False, description="True if this is a pre-computed embedding vector")
    embedding_dim: int | None = Field(None, description="Embedding vector dimension")
    version: str = Field(default="1.0.0", description="Feature version")
    created_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        json_schema_extra = {
            "example": {
                "name": "rsi_14",
                "category": "temporal_momentum",
                "description": "14-period Relative Strength Index",
                "dtype": "float32",
                "min_value": 0.0,
                "max_value": 100.0,
                "normalization": "none",
                "lookback_period": 14,
                "dependencies": ["close"],
                "encoder_target": "tft",
                "is_embedding": False,
                "version": "1.0.0",
            }
        }


class FeatureMetadata(BaseModel):
    """
    Metadata for a batch of features stored in the feature store
    """

    ticker: str = Field(..., description="Asset ticker")
    feature_count: int = Field(..., description="Number of features in batch")
    feature_names: list[str] = Field(..., description="List of feature names")
    categories: dict[str, str] = Field(..., description="Mapping of feature name to category")
    time_range: tuple[datetime, datetime] = Field(
        ..., description="(start_time, end_time) of features"
    )
    data_points: int = Field(..., description="Number of time steps")
    missing_ratio: float = Field(default=0.0, description="Ratio of missing values (0.0-1.0)")
    storage_location: str | None = Field(None, description="Path to offline storage (parquet file)")
    embeddings_available: list[str] = Field(
        default_factory=list, description="List of available embeddings (price, news, graph)"
    )
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    version: str = Field(default="3.0.0", description="Feature store version")

    class Config:
        json_schema_extra = {
            "example": {
                "ticker": "AAPL",
                "feature_count": 156,
                "feature_names": ["rsi_14", "macd", "bb_upper", "volume_sma_20"],
                "categories": {
                    "rsi_14": "temporal_momentum",
                    "macd": "temporal_momentum",
                    "bb_upper": "temporal_volatility",
                },
                "time_range": ("2024-01-01T00:00:00", "2024-01-31T23:59:00"),
                "data_points": 44640,  # 31 days * 24 hours * 60 minutes
                "missing_ratio": 0.002,
                "storage_location": "/features/AAPL/2024/01/features.parquet",
                "embeddings_available": ["price", "news", "graph"],
                "version": "3.0.0",
            }
        }


class EmbeddingVector(BaseModel):
    """
    A pre-computed embedding from a perception encoder
    """

    ticker: str = Field(..., description="Asset ticker")
    encoder: str = Field(..., description="Encoder name (tft, llm, gnn)")
    vector: list[float] = Field(..., description="Embedding vector")
    dimension: int = Field(..., description="Vector dimension")
    timestamp: datetime = Field(..., description="When embedding was computed")
    confidence: float | None = Field(None, description="Encoder confidence score (0.0-1.0)")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    class Config:
        json_schema_extra = {
            "example": {
                "ticker": "AAPL",
                "encoder": "tft",
                "vector": [0.123, -0.456, 0.789],  # Truncated for example
                "dimension": 128,
                "timestamp": "2024-02-09T15:30:00",
                "confidence": 0.95,
                "metadata": {"model_version": "tft_v3.1", "train_date": "2024-01-15"},
            }
        }


# ============================================================================
# FEATURE REGISTRY - Standard Features for V3
# ============================================================================

# Temporal features for TFT encoder
TEMPORAL_FEATURES: list[FeatureDefinition] = [
    FeatureDefinition(
        name="rsi_14",
        category=FeatureCategory.TEMPORAL_MOMENTUM,
        description="14-period Relative Strength Index",
        min_value=0.0,
        max_value=100.0,
        lookback_period=14,
        dependencies=["close"],
        encoder_target="tft",
    ),
    FeatureDefinition(
        name="macd",
        category=FeatureCategory.TEMPORAL_MOMENTUM,
        description="MACD line (12,26,9)",
        lookback_period=26,
        dependencies=["close"],
        encoder_target="tft",
    ),
    FeatureDefinition(
        name="atr_14",
        category=FeatureCategory.TEMPORAL_VOLATILITY,
        description="14-period Average True Range",
        min_value=0.0,
        lookback_period=14,
        dependencies=["high", "low", "close"],
        encoder_target="tft",
    ),
    FeatureDefinition(
        name="bb_width",
        category=FeatureCategory.TEMPORAL_VOLATILITY,
        description="Bollinger Band Width (normalized)",
        min_value=0.0,
        lookback_period=20,
        dependencies=["close"],
        encoder_target="tft",
    ),
]

# Semantic features for NLP encoder
SEMANTIC_FEATURES: list[FeatureDefinition] = [
    FeatureDefinition(
        name="news_sentiment_score",
        category=FeatureCategory.SEMANTIC_NEWS,
        description="Aggregated news sentiment (-1.0 to 1.0)",
        min_value=-1.0,
        max_value=1.0,
        encoder_target="llm",
    ),
    FeatureDefinition(
        name="social_sentiment_score",
        category=FeatureCategory.SEMANTIC_SOCIAL,
        description="Reddit/Twitter sentiment (-1.0 to 1.0)",
        min_value=-1.0,
        max_value=1.0,
        encoder_target="llm",
    ),
]

# Structural features for GNN encoder
STRUCTURAL_FEATURES: list[FeatureDefinition] = [
    FeatureDefinition(
        name="correlation_spy_30d",
        category=FeatureCategory.STRUCTURAL_CORRELATION,
        description="30-day rolling correlation with SPY",
        min_value=-1.0,
        max_value=1.0,
        lookback_period=30 * 24 * 60,  # 30 days in minutes
        encoder_target="gnn",
    ),
    FeatureDefinition(
        name="sector_momentum",
        category=FeatureCategory.STRUCTURAL_SECTOR,
        description="Sector-relative momentum score",
        encoder_target="gnn",
    ),
]

# Embedding features (pre-computed by encoders)
EMBEDDING_FEATURES: list[FeatureDefinition] = [
    FeatureDefinition(
        name="embedding_price",
        category=FeatureCategory.EMBEDDING,
        description="TFT price embedding vector",
        is_embedding=True,
        embedding_dim=128,
        encoder_target="tft",
    ),
    FeatureDefinition(
        name="embedding_news",
        category=FeatureCategory.EMBEDDING,
        description="LLM semantic embedding vector",
        is_embedding=True,
        embedding_dim=64,
        encoder_target="llm",
    ),
    FeatureDefinition(
        name="embedding_graph",
        category=FeatureCategory.EMBEDDING,
        description="GNN structural embedding vector",
        is_embedding=True,
        embedding_dim=32,
        encoder_target="gnn",
    ),
]

# Complete feature registry
ALL_FEATURES = TEMPORAL_FEATURES + SEMANTIC_FEATURES + STRUCTURAL_FEATURES + EMBEDDING_FEATURES


def get_features_by_category(category: FeatureCategory) -> list[FeatureDefinition]:
    """Get all features in a specific category"""
    return [f for f in ALL_FEATURES if f.category == category]


def get_features_by_encoder(encoder: str) -> list[FeatureDefinition]:
    """Get all features used by a specific encoder"""
    return [f for f in ALL_FEATURES if f.encoder_target == encoder]
