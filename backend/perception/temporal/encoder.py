# backend/perception/temporal/encoder.py
"""
Temporal Fusion Transformer (TFT) Encoder

Core temporal encoding model that transforms OHLCV time series into
128-dimensional embeddings using self-attention mechanisms.

Architecture:
1. Variable Selection Network (VSN) - Feature gating
2. Static Covariate Encoder - Metadata integration
3. LSTM Encoder - Sequential processing
4. Multi-head Self-Attention - Long-range dependencies
5. Projection - Output to 128d

Mathematical Formulation:
    VSN: ξ_t = Softmax(W·x_t) ⊙ x_t
    Attention: α = softmax(QK^T / √d_k)
    Output: h = LayerNorm(Attention(Q,K,V)) → Dense(128)

References:
- Lim et al. (2021): "Temporal Fusion Transformers"
"""

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from loguru import logger


@dataclass
class TFTConfig:
    """
    Configuration for Temporal Fusion Transformer.

    Attributes:
        input_dim: Number of input features
        hidden_dim: Hidden state dimension
        output_dim: Output embedding dimension (128 for V3)
        num_heads: Number of attention heads
        num_layers: Number of transformer layers
        dropout: Dropout rate
        lookback_window: Sequence length
    """

    input_dim: int = 30
    hidden_dim: int = 128
    output_dim: int = 128
    num_heads: int = 4
    num_layers: int = 2
    dropout: float = 0.1
    lookback_window: int = 60


class VariableSelectionNetwork(nn.Module):
    """
    Variable Selection Network - Feature Gating

    Automatically learns which features are important for current
    market regime. During low liquidity, might suppress volume features.

    Formula:
        weights = Softmax(GRN(concat(features, context)))
        selected = weights ⊙ features
    """

    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()

        self.grn = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim),
        )

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply feature selection.

        Args:
            x: Input features [batch, seq, input_dim]

        Returns:
            Selected features [batch, seq, input_dim]
        """
        # Compute selection weights
        weights = self.grn(x)
        weights = self.softmax(weights)

        # Apply gating
        return weights * x


class TemporalEncoder(nn.Module):
    """
    Complete Temporal Fusion Transformer encoder.

    Transforms OHLCV sequences into 128d embeddings that capture:
    - Price trends and patterns
    - Volatility regimes
    - Momentum indicators
    - Long-range dependencies

    Example:
        >>> config = TFTConfig(input_dim=30, output_dim=128)
        >>> encoder = TemporalEncoder(config)
        >>>
        >>> # 60 bars with 30 features each
        >>> ohlcv_features = torch.randn(1, 60, 30)
        >>> embedding = encoder(ohlcv_features)  # [1, 128]
    """

    def __init__(self, config: TFTConfig):
        """
        Initialize TFT encoder.

        Args:
            config: TFT configuration
        """
        super().__init__()

        self.config = config

        # Variable selection
        self.vsn = VariableSelectionNetwork(
            input_dim=config.input_dim, hidden_dim=config.hidden_dim, dropout=config.dropout
        )

        # Input projection
        self.input_proj = nn.Linear(config.input_dim, config.hidden_dim)

        # LSTM for sequential encoding
        self.lstm = nn.LSTM(
            input_size=config.hidden_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.num_layers,
            batch_first=True,
            dropout=config.dropout if config.num_layers > 1 else 0,
        )

        # Multi-head self-attention
        self.attention = nn.MultiheadAttention(
            embed_dim=config.hidden_dim,
            num_heads=config.num_heads,
            dropout=config.dropout,
            batch_first=True,
        )

        # Layer normalization
        self.layer_norm = nn.LayerNorm(config.hidden_dim)

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.output_dim),
            nn.LayerNorm(config.output_dim),
        )

        logger.info(f"TemporalEncoder initialized: {config.input_dim} → {config.output_dim}d")

    def forward(self, x: torch.Tensor, static_features: torch.Tensor | None = None) -> torch.Tensor:
        """
        Forward pass through TFT.

        Args:
            x: Input features [batch, seq_len, input_dim]
            static_features: Static covariates (unused for now)

        Returns:
            Temporal embedding [batch, output_dim]
        """
        # Variable selection
        x_selected = self.vsn(x)

        # Project to hidden dimension
        x_proj = self.input_proj(x_selected)

        # LSTM encoding
        lstm_out, (h_n, c_n) = self.lstm(x_proj)

        # Self-attention
        attn_out, attn_weights = self.attention(lstm_out, lstm_out, lstm_out)

        # Residual connection + layer norm
        x_combined = self.layer_norm(lstm_out + attn_out)

        # Pool across time dimension (use last timestep)
        x_pooled = x_combined[:, -1, :]

        # Project to output dimension
        embedding = self.output_proj(x_pooled)

        return embedding

    def encode(self, features: np.ndarray, static_features: np.ndarray | None = None) -> np.ndarray:
        """
        Encode to numpy (inference mode).

        Args:
            features: Input features [seq_len, input_dim]
            static_features: Static covariates

        Returns:
            Embedding [output_dim]
        """
        self.eval()

        with torch.no_grad():
            # Add batch dimension
            if features.ndim == 2:
                features = features[np.newaxis, :]

            # Convert to tensor
            x = torch.FloatTensor(features)

            # Forward
            embedding = self.forward(x)

            # Remove batch dimension and convert
            return embedding[0].cpu().numpy()


def create_temporal_encoder(
    input_dim: int = 30, output_dim: int = 128, lookback: int = 60
) -> TemporalEncoder:
    """
    Factory function to create temporal encoder.

    Args:
        input_dim: Number of features
        output_dim: Output dimension (128 for V3)
        lookback: Sequence length

    Returns:
        Configured TemporalEncoder
    """
    config = TFTConfig(input_dim=input_dim, output_dim=output_dim, lookback_window=lookback)

    return TemporalEncoder(config)
