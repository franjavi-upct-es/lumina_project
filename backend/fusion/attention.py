# backend/fusion/attention.py
"""
Cross-Modal Attention Mechanism

Implements Transformer-based attention that allows different modalities
to "talk" to each other and suppress noise via learned attention weights.

This is the core innovation of the V3 fusion layer - it enables the
system to dynamically prioritize relevant information based on market context.

Scenarios:
- Earnings Call: Semantic embedding dominates ⟶ suppress technical
- Flash Crash: Price embedding spikes ⟶ amplify despite no news
- Sector Rotation: Graph embedding activates ⟶ highlight structural

References:
- Vaswani et al. (2017): "Attention Is All You Need"
- Tsai et al. (2019): "Multimodal Transformer for Unaligned Multimodal Language Sequences"

Mathematical Formulation:
    Attention(Q, K, V) = softmax(QK^T / √d_k) V

    Where:
    - Q (Query): What information am I looking for?
    - K (Key): What information do I have?
    - V (Value): The actual information
    - d_k: Dimension of key vectors (for scaling)
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger


class CrossModalAttention(nn.Module):
    """
    Single-head cross-modal attention.

    Allows one modality (query) to attend to information from
    another modality (key/value).

    Example Use Case:
        Semantic modality (news) queries Temporal modality (price)
        to determine if price action confirms the news sentiment.
    """

    def __init__(
        self,
        query_dim: int,
        key_dim: int,
        value_dim: int,
        output_dim: int,
        dropout: float = 0.1,
    ):
        """
        Initialize cross-modal attention.

        Args:
            query_dim: Dimension of query modality
            key_dim: Dimension of key modality
            value_dim: Dimension of value modality
            output_dim: Output dimension
            dropout: Dropout probability
        """
        super().__init__()

        self.query_dim = query_dim
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.output_dim = output_dim

        # Projection matrices
        self.W_q = nn.Linear(query_dim, output_dim, bias=False)
        self.W_k = nn.Linear(key_dim, output_dim, bias=False)
        self.W_v = nn.Linear(value_dim, output_dim, bias=False)

        # Output projection
        self.W_o = nn.Linear(output_dim, output_dim)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Scaling factor
        self.scale = math.sqrt(output_dim)

        logger.debug(
            f"CrossModalAttention: query_dim={query_dim}, "
            f"key_dim={key_dim}, output_dim={output_dim}"
        )

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through attention mechanism.

        Args:
            query: Query tensor [batch_size, query_dim]
            key: Key tensor [batch_size, key_dim]
            value: Value tensor [batch_size, value_dim]
            mask: Optional attention mask

        Returns:
            output: Attended output [batch_size, output_dim]
            attention_weights: Attention weights [batch_size, 1]
        """
        # Project inputs
        Q = self.W_q(query)  # [batch_size, output_dim]
        K = self.W_k(key)  # [batch_size, output_dim]
        V = self.W_v(value)  # [batch_size, output_dim]

        # Compute attention scores
        # scores = QK^T / √d_k
        scores = torch.matmul(Q.unqueeze(1), K.unqueeze(2)) / self.scale
        scores = scores.squeeze(-1)  # [batch_size, 1]

        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # Compute attention weights
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # Apply attention to values
        output = attention_weights * V

        # Output projection
        output = self.W_o(output)

        return output, attention_weights


class MultiHeadCrossAttention(nn.Module):
    """
    Multi-Head cross-modal attention.

    Uses multiple attention heads to capture different types of
    relationships between modalities.

    Mathematical Formulation:
        MultiHead(Q, K, V) = COncat(head_1, ..., head_h) W^O

        where head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
    """

    def __init__(
        self, input_dim: int, num_heads: int = 4, head_dim: int = 64, dropout: float = 0.1
    ):
        """
        Initialize multi-head cross attention.

        Args:
            input_dim: Input dimension
            num_heads: Number of attention heads
            head_dim: Dimension per head
            dropout: Dropout probability
        """
        super().__init__()

        self.input_dim = input_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.total_dim = num_heads * head_dim

        # Multi-Head projections
        self.W_q = nn.Linear(input_dim, self.total_dim, bias=False)
        self.W_k = nn.Linear(input_dim, self.total_dim, bias=False)
        self.W_v = nn.Linear(input_dim, self.total_dim, bias=False)

        # Output projection
        self.W_o = nn.Linear(self.total_dim, input_dim)

        # Layer norm
        self.layer_norm = nn.LayerNorm(input_dim)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        self.scale = math.sqrt(head_dim)

        logger.debug(
            f"MutliHeadCrossAttention: {num_heads} heads, head_dim={head_dim}, total_dim={self.total_dim}"
        )

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through multi-head attention.

        Args:
            x: Input tensor [batch_size, input_dim]
            mask: Optional attention mask

        Returns:
            output: Attended output [batch_size, input_dim]
            attention_weights: Average attention weights [batch_size, num_heads]
        """
        batch_size = x.size(0)

        # Store resiudal
        residual = x

        # Project to Q, K, V
        Q = self.W_q(x)  # [batch_size, total_dim]
        K = self.W_k(x)
        V = self.W_v(x)

        # Reshape for multi-head attention
        # [batch_size, num_heads, head_dim]
        Q = Q.view(batch_size, self.num_heads, self.head_dim)
        K = K.view(batch_size, self.num_heads, self.head_dim)
        V = V.view(batch_size, self.num_heads, self.head_dim)

        # Compute attention scores for each head
        # [batch_size, num_heads, head_dim] @ [batch_size, num_heads, head_dim]
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # Compute attention weights
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # Apply attention to values
        attended = torch.matmul(attention_weights, V)

        # Reshape and concatenate heads
        attended = attended.view(batch_size, self.total_dim)

        # Ouput projection
        output = self.W_o(attended)
        output = self.dropout(output)

        # Residual connection + layer norm
        output = self.layer_norm(output + residual)

        # Average attention weights across heads
        avg_attention = attention_weights.mean(dim=1)

        return output, avg_attention


class ModalityFusion(nn.Module):
    """
    Complete fusion module using cross-modal attention.

    Fuses temporal, semantic, and structural embeddings using
    bidirectional cross-attention between all modality pairs.

    Architecture:
        1. Self-attention within each modality
        2. Cross-attention between modality pairs
        3. Fusion via weighted combination
        4. Final projection to unified representation
    """

    def __init__(
        self,
        temporal_dim: int = 128,
        semantic_dim: int = 64,
        structural_dim: int = 32,
        hidden_dim: int = 256,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        """
        Initialize modality fusion.

        Args:
            temporal_dim: Temporal embedding dimension
            semantic_dim: Semantic embedding dimension
            structural_dim: Structural embedding dimension
            hidden_dim: Hidden dimension for fusion
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()

        self.temporal_dim = temporal_dim
        self.semantic_dim = semantic_dim
        self.structural_dim = structural_dim
        self.hidden_dim = hidden_dim

        # Project each modality to common dimension
        self.temporal_proj = nn.Linear(temporal_dim, hidden_dim)
        self.semantic_proj = nn.Linear(semantic_dim, hidden_dim)
        self.structural_proj = nn.Linear(structural_dim, hidden_dim)

        # Self-attention for each modality
        self.temporal_self_attn = MultiHeadCrossAttention(
            hidden_dim, num_heads, hidden_dim // num_heads, dropout
        )
        self.semantic_self_attn = MultiHeadCrossAttention(
            hidden_dim, num_heads, hidden_dim // num_heads, dropout
        )
        self.structural_self_attn = MultiHeadCrossAttention(
            hidden_dim, num_heads, hidden_dim // num_heads, dropout
        )

        # Cross-attention between modalities
        self.cross_attn_t_s = CrossModalAttention(
            hidden_dim,
            hidden_dim,
            hidden_dim,
            hidden_dim,
            dropout,
        )
        self.cross_attn_t_g = CrossModalAttention(
            hidden_dim, hidden_dim, hidden_dim, hidden_dim, dropout
        )
        self.cross_attn_s_t = CrossModalAttention(
            hidden_dim, hidden_dim, hidden_dim, hidden_dim, dropout
        )
        self.cross_attn_s_g = CrossModalAttention(
            hidden_dim, hidden_dim, hidden_dim, hidden_dim, dropout
        )
        self.cross_attn_g_t = CrossModalAttention(
            hidden_dim, hidden_dim, hidden_dim, hidden_dim, dropout
        )
        self.cross_attn_g_s = CrossModalAttention(
            hidden_dim, hidden_dim, hidden_dim, hidden_dim, dropout
        )

        # Fusion network
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        logger.info(
            "ModalityFusion initialized: "
            f"temporal={temporal_dim}, semantic={semantic_dim}, "
            f"structural={structural_dim} ⟶ hidden={hidden_dim}"
        )

    def forward(
        self,
        temporal: torch.Tensor,
        semantic: torch.Tensor,
        structural: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through fusion module.

        Args:
            temporal: Temporal embedding [batch_size, temporal_dim]
            semantic: Semantic embedding [batch_size, semantic_dim]
            structural: Structural embedding [batch_size, structural_dim]

        Returns:
            fused: Fused representation [batch_size, hidden_dim]
            attention_weights: Dictionary of attention weights
        """
        # Project to common dimension
        t = self.temporal_proj(temporal)
        s = self.semantic_proj(semantic)
        g = self.structural_proj(structural)

        # Self-attention within modalities
        t_self, _ = self.temporal_self_attn(t)
        s_self, _ = self.semantic_self_attn(s)
        g_self, _ = self.structural_self_attn(g)

        # Cross-attention between modalitites
        t_from_s, w_t_s = self.cross_attn_t_s(t_self, s_self, s_self)
        t_from_g, w_t_g = self.cross_attn_t_g(t_self, g_self, g_self)

        s_from_t, w_s_t = self.cross_attn_s_t(s_self, t_self, t_self)
        s_from_g, w_s_g = self.cross_attn_s_g(s_self, g_self, g_self)

        g_from_t, w_g_t = self.cross_attn_g_t(g_self, t_self, t_self)
        g_from_s, w_g_s = self.cross_attn_g_s(g_self, s_self, s_self)

        # Combine cross-attended representation
        t_enhanced = t_self + t_from_s + t_from_g
        s_enhanced = s_self + s_from_t + s_from_g
        g_enhanced = g_self + g_from_t + g_from_s

        # Concatenate and fuse
        combined = torch.cat([t_enhanced, s_enhanced, g_enhanced], dim=1)
        fused = self.fusion(combined)

        # Collect attention weights for analysis
        attention_weights = {
            "temporal_to_semantic": w_t_s.detach(),
            "temporal_to_structural": w_t_g.detach(),
            "semantic_to_temporal": w_s_t.detach(),
            "semantic_to_structural": w_s_g.detach(),
            "structural_to_temporal": w_g_t.detach(),
            "structural_to_semantic": w_g_s.detach(),
        }

        return fused, attention_weights
