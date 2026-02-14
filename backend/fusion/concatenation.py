# backend/fusion/concatenation.py
"""
Embedding COncatenation Module

Implements simple concatenation of multi-modal embeddings and
optional gating mechanisms for modality selection.

V3 Architecture:
- Temporal Embedding (TFT): 128 dimensions
- Semantic Embedding (BERT): 64 dimensions
- Structural Embedding (GNN): 32 dimensions
- Total Fused Vector: 224 dimensions

The concatenation preserves all information from each modality
allowing the attention mechanism to learn relevance dynamically.
"""

import numpy as np
import torch
import torch.nn as nn
from loguru import logger


class SimpleConcatenation:
    """
    Simple concatenation of embeddings from multiple modalities.

    This is the first step in the fusion pipeline, creating the
    raw super-vector attention mechanisms refine it.

    Example:
        >>> concat = SimpleConcatenation(
        >>>     modality_dims={'temporal': 128, 'semantic': 64, 'structural': 32}
        >>> )
        >>> embeddings = {
        >>>     'temporal': np.random.randn(128),
        >>>     'semanctic': np.random.randn(64),
        >>>     'structural': np.random.randn(32)
        >>> }
        >>> fused = concat.concatenate(embeddings)
        >>> print(fused.shape)  # (224, )
    """

    def __init__(
        self,
        modality_dims: dict[str, int],
        modality_order: list[str] | None = None,
    ):
        """
        Initialize concatenation module.

        Args:
            modality_dims: Dictionary mapping modality names to dimensions
            modality_order: Order of modalities in output (optional)
        """
        self.modality_dims = modality_dims

        # Set modality order
        if modality_order is None:
            # Default order: temporal, semantic, structural
            self.modality_order = sorted(modality_dims.keys())
        else:
            self.modality_order = modality_order

        # Calculate total dimension
        self.total_dim = sum(modality_dims.values())

        # Calculate offsets for each modality
        self.modality_offsets = {}
        offset = 0
        for modality in self.modality_order:
            dim = modality_dims[modality]
            self.modality_offsets[modality] = (offset, offset + dim)
            offset += dim

        logger.debug(
            f"SimpleConcatenation initialized: {len(modality_dims)} modalities, "
            f"total_dim={self.total_dim}"
        )

    def concatenate(
        self, embeddings: dict[str, np.ndarray], fill_missing: bool = True
    ) -> np.ndarray:
        """
        Concatenate embeddings from multiple modalities.

        Args:
            embeddings: Dictionary mapping modality names to embedding vectors
            fill_missing: Fill missing modalities with zeros

        Returns:
            Concatenated embedding vector

        Raises:
            ValueError: If required modalities are missing and fill_missing=False
        """
        # Check for missing modalities
        missing = set(self.modality_order) - set(embeddings.keys())
        if missing and not fill_missing:
            raise ValueError(f"Missing required modalities: {missing}")

        # Initialize output vector
        fused = np.zeros(self.total_dim, dtype=np.float32)

        # Fill in each modality
        for modality in self.modality_order:
            start, end = self.modality_offsets[modality]

            if modality in embeddings:
                embedding = embeddings[modality]

                # Validate dimension
                expected_dim = self.modality_dims[modality]
                if embedding.shape[0] != expected_dim:
                    logger.warning(
                        f"Modality '{modality}' has incorrect dimension: "
                        f"expected {expected_dim}, got {embedding.shape[0]}"
                    )
                    # Resize if needed
                    if embedding.shape[0] < expected_dim:
                        # Pad with zeros
                        embedding = np.pad(
                            embedding, (0, expected_dim - embedding.shape[0]), mode="constant"
                        )
                    else:
                        # Truncate
                        embedding = embedding[:expected_dim]

                fused[start:end] = embedding
            else:
                # Missing modality - already filled with zeros
                if not fill_missing:
                    logger.warning(f"Modality '{modality}' missing, filling with zeros")

        return fused

    def split(self, fused_embedding: np.ndarray) -> dict[str, np.ndarray]:
        """
        Split fused embedding back into individual modalities.

        Useful for analysis and debugging.

        Args:
            fused_embedding: Concatenated embedding

        Returns:
            Dictionary of individual modality embeddings
        """
        if fused_embedding.shape[0] != self.total_dim:
            raise ValueError(f"Expected dimension {self.total_dim}, got {fused_embedding.shape[0]}")

        embeddings = {}
        for modality in self.modality_order:
            start, end = self.modality_offsets[modality]
            embeddings[modality] = fused_embedding[start:end]

        return embeddings

    def get_modality_slice(self, modality: str) -> tuple[int, int]:
        """
        Get slice indices for a specific modality.

        Args:
            modality: Modality name

        Returns:
            (start_index, end_index) tuple
        """
        if modality not in self.modality_offsets:
            raise ValueError(f"Unknown modality: {modality}")

        return self.modality_offsets[modality]


class ModalityGate(nn.Module):
    """
    Learnable gating mechanism for modality selection.

    Allows the network to dynamically select which modalities are
    relevant for the current market state.

    Uses sigmoid gating: g_i = σ(W·x + b)
    where g_i ∈ [0, 1] controls modality i importance
    """

    def __init__(
        self,
        input_dim: int,
        num_modalities: int,
        hidden_dim: int = 64,
    ):
        """
        Initialize modality gate.

        Args:
            input_dim: Input dimension (fused embedding)
            num_modalities: Number of modalities
            hidden_dim: Hidden layer dimension
        """
        super().__init__()

        self.input_dim = input_dim
        self.num_modalities = num_modalities

        # Gating network
        self.gate_network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_modalities),
            nn.Sigmoid(),  # Gate values in [0, 1]
        )

        logger.debug(
            f"ModalityGate initialized: {num_modalities} modalities, hidden_dim={hidden_dim}"
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through gating network.

        Args:
            x: Input tensor [batch_size, input_dim]

        Returns:
            gates: Gate values [batch_size, num_modalities]
            gated_x: Input modulated by gates
        """
        # Compute gate values
        gates = self.gate_network(x)

        # Apply gates to input
        # NOTE: This assumes each modality occupies contiguous dimensions
        # For more sophisticates gating, use attention mechanisms
        gated_x = x * gates.mean(dim=1, keepdim=True)

        return gates, gated_x

    def get_gate_statistics(self, x: torch.Tensor) -> dict[str, float]:
        """
        Get statistics about gate activations.

        Args:
            x: Input tensor

        Returns:
            Dictionary of gate statistics
        """
        with torch.no_grad():
            gates, _ = self.forward(x)

            stats = {
                f"gate_{i}_mean": gates[:, i].mean().item() for i in range(self.num_modalities)
            }

            stats["gate_variance"] = gates.var().item()
            stats["gate_min"] = gates.min().item()
            stats["gate_max"] = gates.max().item()

        return stats


def create_default_concatenation() -> SimpleConcatenation:
    """
    Create default V3 concatenation with standard dimensions.

    Returns:
        SimpleConcatenation configured for V3 architecture
    """
    return SimpleConcatenation(
        modality_dims={
            "temporal": 128,  # TFT output
            "semantic": 64,  # BERT output
            "structural": 32,  # GNN output
        },
        modality_order=["temporal", "semantic", "structural"],
    )
