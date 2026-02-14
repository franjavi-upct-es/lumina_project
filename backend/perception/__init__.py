# backend/perception/__init__.py
"""
Perception Layer - Layer 1 of V3 Chimera Architecture ("The Eyes")

The perception engines compress raw, noisy, multi-modal data into dense
embeddings that capture the essence of market information.

Three Specialized Encoders:
1. Temporal (TFT): Price action → 128d embedding
2. Semantic (BERT): News/social → 64d embedding
3. Structural (GNN): Market graph → 32d embedding

Total: 224-dimensional raw super-vector fed to fusion layer

Philosophy:
"We do not feed raw data to the RL agent. Raw data is noisy, sparse,
and chemically different across modalities. Instead, we feed Compressed
Representations that capture the essence, stripping away noise."

Components:
- temporal/: Temporal Fusion Transformer for OHLCV time series
- semantic/: Distilled LLM for text embeddings
- structural/: Graph Neural Network for market structure

Example:
    >>> from backend.perception import TemporalEncoder, SemanticEncoder, StructuralEncoder
    >>>
    >>> # Initialize encoders
    >>> temporal_enc = TemporalEncoder()
    >>> semantic_enc = SemanticEncoder()
    >>> structural_enc = StructuralEncoder()
    >>>
    >>> # Generate embeddings
    >>> price_emb = temporal_enc.encode(ohlcv_data)      # 128d
    >>> text_emb = semantic_enc.encode(news_text)        # 64d
    >>> graph_emb = structural_enc.encode(market_graph)  # 32d
    >>>
    >>> # Feed to fusion layer
    >>> from backend.fusion import FusionStateBuilder
    >>> fusion = FusionStateBuilder()
    >>> state = fusion.build_state(price_emb, text_emb, graph_emb)  # 224d
"""

# Temporal Encoder
# Semantic Encoder
from backend.perception.semantic import (
    DistilledLLMEncoder,
    FinancialTokenizer,
    SemanticEncoder,
    TextVectorizer,
)

# Structural Encoder
from backend.perception.structural import (
    DynamicEdgeComputer,
    GraphAttentionNetwork,
    GraphBuilder,
    MarketGraph,
)
from backend.perception.temporal import (
    TemporalEncoder,
    TemporalPreprocessor,
    TFTConfig,
    create_temporal_features,
)

__all__ = [
    # Temporal
    "TemporalEncoder",
    "TFTConfig",
    "TemporalPreprocessor",
    "create_temporal_features",
    # Semantic
    "SemanticEncoder",
    "FinancialTokenizer",
    "TextVectorizer",
    "DistilledLLMEncoder",
    # Structural
    "GraphAttentionNetwork",
    "GraphBuilder",
    "DynamicEdgeComputer",
    "MarketGraph",
]
