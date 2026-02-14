# backend/perception/temporal/__init__.py
"""
Temporal Encoder - Layer 1.A of V3 Chimera Architecture

Temporal Fusion Transformer (TFT) for OHLCV time series encoding.
Replaces standard LSTM/GRU with self-attention mechanisms.

Problem Solved:
LSTMs suffer from vanishing gradients over long sequences and struggle
to integrate static metadata (sector, market cap) alongside dynamic data.

TFT Solution:
- Self-attention weights importance of different time steps dynamically
- Treats time as a map of relevant events, not just sequence
- Automatically selects which features matter for current regime

Architecture Components:
1. Variable Selection Networks (VSN): Auto-select relevant features
2. Static Covariate Encoders: Integrate metadata conditioning
3. Multi-head Attention: Identify long-term dependencies
4. Interpretability: Attention weights show which candles matter

Output: 128-dimensional temporal embedding

Mathematical Formulation:
    VSN: ξ_t = Softmax(W·x_t) ⊙ x_t  (feature selection)
    Attention: α_ij = softmax(Q_i·K_j^T / √d_k)
    Output: h_t = Σ α_ij·V_j → Dense(128)

Example:
    >>> from backend.perception.temporal import TemporalEncoder
    >>>
    >>> encoder = TemporalEncoder()
    >>> ohlcv_data = get_1min_candles(symbol='AAPL', bars=60)
    >>> embedding = encoder.encode(ohlcv_data)  # 128d
    >>>
    >>> # Embedding captures: trend, volatility, momentum, patterns
    >>> # with attention on relevant historical events

References:
- Lim et al. (2021): "Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting"
"""

from backend.perception.temporal.encoder import (
    TemporalEncoder,
    TFTConfig,
    VariableSelectionNetwork,
)
from backend.perception.temporal.inference import (
    InferenceConfig,
    TemporalInference,
    encode_real_time,
)
from backend.perception.temporal.preprocessor import (
    PreprocessorConfig,
    TemporalPreprocessor,
    create_temporal_features,
)

__all__ = [
    # Preprocessor
    "TemporalPreprocessor",
    "PreprocessorConfig",
    "create_temporal_features",
    # Encoder
    "TemporalEncoder",
    "TFTConfig",
    "VariableSelectionNetwork",
    # Inference
    "TemporalInference",
    "InferenceConfig",
    "encode_real_time",
]
