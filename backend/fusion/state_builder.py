# backend/fusion/state_builder.py
"""
Fusion State Builder

Orchestrates the complete fusion pipeline from multi-modal embeddings
to unified super-state representation.

This is the main interface for the fusion layer, combining:
1. Concatenation of raw embeddings
2. Cross-modal attention mechanisms
3. State refinement and output

The state builder acts as the "Thalamus" coordinator, deciding which
sensory information is relevant for the current market moment.
"""

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger

from backend.fusion.attention import ModalityFusion
from backend.fusion.concatenation import SimpleConcatenation


@dataclass
class ModalityInput:
    """
    Container for multi-modal inputs.

    Attributes:
        temporal: Temporal embedding from TFT (128d)
        semantic: Semantic embedding from BERT (64d)
        structural: Structural embedding from GNN (32d)
        metadata: Optional metadata dictionary
    """

    temporal: np.ndarray | None = None
    semantic: np.ndarray | None = None
    structural: np.ndarray | None = None
    metadata: dict | None = None

    def to_dict(self) -> dict[str, np.ndarray]:
        """Convert to dictionary format."""
        return {
            "temporal": self.temporal,
            "semantic": self.semantic,
            "structural": self.structural,
        }

    def validate(self) -> bool:
        """
        Validate that at least one modality is present.

        Returns:
            True is valid
        """
        return any(
            [self.temporal is not None, self.semantic is not None, self.structural is not None]
        )


@dataclass
class FusionConfig:
    """
    Configuration for fusion state builder.

    Attributes:
        use_attention: Enable cross-modal attention
        use_residual: Add residual connections
        dropout: Dropout probability
        hidden_dim: Hidden dimension for attention
        num_attention_heads: Number of attention heads
        normalize_output: L2-normalize output state
    """

    use_attention: bool = True
    use_residual: bool = True
    dropout: float = 0.1
    hidden_dim: int = 256
    num_attention_heads: int = 4
    normalize_output: bool = False


class FusionStateBuilder(nn.Module):
    """
    Main fusion state builder.

    Attributes Flow:
        1. Input validation and preprocessing
        2. Simple concatenation (224d)
        3. Cross-modal attention (if enabled)
        4. Optional residual connection
        5. Output normalization (if enabled)

    Example:
        >>> config = FusionConfig(use_attention=True, hidden_dim=256)
        >>> builder = FusionStateBuilder(config)
        >>>
        >>> inputs = ModalityInput(
        >>>     temporal=np.random.randn(128),
        >>>     semantic=np.random.randn(64),
        >>>     structural=np.random.randn(32)
        >>> )
        >>>
        >>> state, info = builder.build_state(inputs)
        >>> print(state.shape)  # (hidden_dim,) or (224,) if no attention
    """

    def __init__(
        self,
        config: FusionConfig | None = None,
        temporal_dim: int = 128,
        semantic_dim: int = 64,
        structural_dim: int = 32,
    ):
        """
        Initialize fusion state builder.

        Args:
            config: Fusion configuration
            temporal_dim: Temporal embedding dimension
            semantic_dim: Semantic embedding dimension
            structural_dim: Structural embedding dimension
        """
        super().__init__()

        self.config = config or FusionConfig()
        self.temporal_dim = temporal_dim
        self.semantic_dim = semantic_dim
        self.structural_dim = structural_dim

        # Concatenation module
        self.concatenation = SimpleConcatenation(
            modality_dims={
                "temporal": temporal_dim,
                "semantic": semantic_dim,
                "structural": structural_dim,
            },
            modality_order=["temporal", "semantic", "structural"],
        )

        self.raw_state_dim = self.concatenation.total_dim  # 224

        # Cross-modal attention (optional)
        if self.config.use_attention:
            self.attention = ModalityFusion(
                temporal_dim=temporal_dim,
                semantic_dim=semantic_dim,
                structural_dim=structural_dim,
                hidden_dim=self.config.hidden_dim,
                num_heads=self.config.num_attention_heads,
                dropout=self.config.dropout,
            )
            self.output_dim = self.config.hidden_dim
        else:
            self.attention = None
            self.output_dim = self.raw_state_dim

        # Optional output projection
        if self.config.use_residual and self.config.use_attention:
            # Project raw state to match attention output dim
            self.residual_proj = nn.Linear(self.raw_state_dim, self.config.hidden_dim)
        else:
            self.residual_proj = None

        logger.info(
            "FusionStateBuilder initialized: "
            f"input_dims=({temporal_dim}, {semantic_dim}, {structural_dim}), "
            f"output_dim={self.output_dim}, use_attention={self.config.use_attention}"
        )

    def build_state(
        self, inputs: ModalityInput, return_attention: bool = False
    ) -> tuple[np.ndarray, dict]:
        """
        Build unified state from multi-modal inputs.

        Args:
            inputs: Multi-modal embeddings
            return_attention: Return attention weights

        Returns:
            state: Unified state vector
            info: Information dictionary with metadata
        """
        # Validate inputs
        if not inputs.validate():
            raise ValueError("At least one modality must be provided")

        # Convert to dictionary
        embedding_dict = inputs.to_dict()

        # Remove None values
        embedding_dict = {k: v for k, v in embedding_dict.items() if v is not None}

        # Concatenate embeddings
        raw_state = self.concatenation.concatenate(embedding_dict, fill_missing=True)

        # If not attention, return raw concatenation
        if not self.config.use_attention:
            # Optionally normalize
            if self.config.normalize_output:
                norm = np.linalg.norm(raw_state)
                if norm > 0:
                    raw_state = raw_state / norm

            return raw_state, {
                "raw_state_dim": self.raw_state_dim,
                "output_dim": self.output_dim,
                "modalities_present": list(embedding_dict.keys()),
            }

        # Apply cross-modal attention
        with torch.no_grad():
            # Convert to tensors
            temporal_tensor = torch.FloatTensor(
                embedding_dict.get("temporal", np.zeros(self.temporal_dim))
            ).unsqueeze(0)
            semantic_tensor = torch.FloatTensor(
                embedding_dict.get("semantic", np.zeros(self.semantic_dim))
            ).unsqueeze(0)
            structural_tensor = torch.FloatTensor(
                embedding_dict.get("structural", np.zeros(self.structural_dim))
            ).unsqueeze(0)

            # Forward through attention
            fused_state, attention_weights = self.attention(
                temporal_tensor, semantic_tensor, structural_tensor
            )

            # Apply residual connection if enabled
            if self.config.use_residual and self.residual_proj is not None:
                raw_state_tensor = torch.FloatTensor(raw_state).unsqueeze(0)
                residual = self.residual_proj(raw_state_tensor)
                fused_state = fused_state + residual

            # Convert back to numpy
            state = fused_state.squeeze(0).numpy()

        # Optionally normalize
        if self.config.normalize_output:
            norm = np.linalg.norm(state)
            if norm > 0:
                state = state / norm

        # Build info dictionary
        info = {
            "raw_state_dim": self.raw_state_dim,
            "output_dim": self.output_dim,
            "modalities_present": list(embedding_dict.keys()),
        }

        if return_attention:
            # Convert attention weights to numpy
            info["attention_weights"] = {
                k: v.squeeze(0).numpy() for k, v in attention_weights.items()
            }

        return state, info

    def forward(
        self, temporal: torch.Tensor, semantic: torch.Tensor, structural: torch.Tensor
    ) -> tuple[torch.Tensor, dict]:
        """
        PyTorch forward pass (for training).

        Args:
            temporal: Temporal embedding [batch_size, temporal_dim]
            semantic: Semantic embedding [batch_size, semantic_dim]
            structural: Structural embedding [batch_size, structural_dim]

        Returns:
            state: Fused state [batch_size, output_dim]
            attention_weights: Attention weights dictionary
        """
        if not self.config.use_attention:
            # Simple concatenation
            state = torch.cat([temporal, semantic, structural], dim=-1)
            return state, {}

        # Cross-modal attention
        fused_state, attention_weights = self.attention(temporal, semantic, structural)

        # Apply residual if enabled
        if self.config.use_residual and self.residual_proj is not None:
            raw_state = torch.cat([temporal, semantic, structural], dim=-1)
            residual = self.residual_proj(raw_state)
            fused_state = fused_state + residual

        # Normalzie if enabled
        if self.config.normalize_output:
            fused_state = F.normalize(fused_state, p=2, dim=-1)

        return fused_state, attention_weights

    def get_attention_summary(self, inputs: ModalityInput) -> dict[str, float | str]:
        """
        Get summary statistics of attention weights.

        Useful for understanding which modalities are more important
        for the current market state.

        Args:
            inputs: Multi-modal inputs

        Returns:
            Dictionary of attention statistics
        """
        if not self.config.use_attention:
            return {"message": "Attention not enabled"}

        _, info = self.build_state(inputs, return_attention=True)

        if "attention_weights" not in info:
            return {"message": "No attention weights available"}

        weights = info["attention_weights"]

        # Calculate summary statistics
        summary = {}
        for key, weight_array in weights.items():
            summary[f"{key}_mean"] = float(np.mean(weight_array))
            summary[f"{key}_std"] = float(np.std(weight_array))

        return summary

    def save(self, path: str):
        """
        Save fusion model.

        Args:
            path: Save path
        """
        torch.save(
            {
                "config": self.config,
                "state_dict": self.state_dict(),
                "temporal_dim": self.temporal_dim,
                "semantic_dim": self.semantic_dim,
                "structural_dim": self.structural_dim,
            },
            path,
        )
        logger.success(f"Fusion model saved to {path}")

    def load(self, path: str):
        """
        Load fusion model.

        Args:
            path: Load path
        """
        checkpoint = torch.load(path)
        self.load_state_dict(checkpoint["state_dict"])
        logger.success(f"Fusion model loaded from {path}")


def create_default_fusion_builder() -> FusionStateBuilder:
    """
    Create default V3 fusion state builder.

    Returns:
        FusionStateBuilder with default V3 configuration
    """
    config = FusionConfig(
        use_attention=True,
        use_residual=True,
        dropout=0.1,
        hidden_dim=256,
        num_attention_heads=4,
        normalize_output=False,
    )

    return FusionStateBuilder(config=config, temporal_dim=128, semantic_dim=64, structural_dim=32)
