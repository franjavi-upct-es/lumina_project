# backend/perception/structural/gnn_model.py
"""
Graph Neural Network Model - Graph Attention Network (GATv2)

Implements message passing on the market graph to generate structural
embeddings that capture "neighborhood stress" and market contagion effects.

Mathematical Formulation (GATv2):
    h_i' = σ(Σ_j α_ij W h_j)

    Where:
    - h_i = node i's feature vector
    - α_ij = attention coefficient from node j to i
    - W = learnable weight matrix
    - σ = activation function (LeakyReLU)

    Attention coefficients:
    α_ij = softmax_j(LeakyReLU(a^T [W h_i || W h_j]))

    Where || denotes concatenation and a is a learnable attention vector.

Key Innovation:
GATv2 computes attention after the linear transformation, allowing for
more dynamic and expressive attention patterns.

Example:
If semiconductor neighbors (NVDA, AMD) are crashing, the GNN embedding
for AAPL will reflect "neighborhood stress" even before AAPL drops.

References:
- Veličković et al. (2018): "Graph Attention Networks"
- Brody et al. (2021): "How Attentive are Graph Attention Networks?" (GATv2)
"""

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger


@dataclass
class GNNConfig:
    """
    Configuration for Graph Neural Network.

    Attributes:
        input_dim: Input node feature dimension
        hidden_dim: Hidden layer dimension
        output_dim: Output embedding dimension (32 for V3)
        num_layers: Number of GNN layers
        num_heads: Number of attention heads
        dropout: Dropout rate
        use_edge_features: Use edge weights in attention
        concat_heads: Concatenate attention heads vs average
    """

    input_dim: int = 64
    hidden_dim: int = 64
    output_dim: int = 32
    num_layers: int = 2
    num_heads: int = 4
    dropout: float = 0.2
    use_edge_features: bool = True
    concat_heads: bool = False


class GATv2Layer(nn.Module):
    """
    Single Graph Attention Network v2 layer.

    Performs one round of message passing with attention-based aggregation.

    GATv2 improvement: Computes attention after the linear transformation,
    allowing for more dynamic attention patterns.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_heads: int = 4,
        dropout: float = 0.2,
        concat: bool = True,
        use_edge_weights: bool = True,
    ):
        """
        Initialize GAT layer.

        Args:
            in_features: Input feature dimension
            out_features: Output feature dimension per head
            num_heads: Number of attention heads
            dropout: Dropout rate
            concat: Concatenate heads (True) or average (False)
            use_edge_weights: Incorporate edge weights
        """
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.num_heads = num_heads
        self.concat = concat
        self.use_edge_weights = use_edge_weights

        # Linear transformation for each head
        self.W = nn.ModuleList(
            [nn.Linear(in_features, out_features, bias=False) for _ in range(num_heads)]
        )

        # Attention mechanism (GATv2 style)
        self.a = nn.ParameterList(
            [nn.Parameter(torch.FloatTensor(2 * out_features, 1)) for _ in range(num_heads)]
        )

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Activation
        self.leakyrelu = nn.LeakyReLU(0.2)

        # Initialize parameters
        self._init_parameters()

        logger.debug(f"GATv2Layer: {num_heads} heads, in={in_features}, out={out_features}")

    def _init_parameters(self):
        """Initialize parameters using Xavier initialization."""
        for a_head in self.a:
            nn.init.xavier_uniform_(a_head)

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, edge_weights: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Forward pass through GAT layer.

        Args:
            x: Node features [num_nodes, in_features]
            edge_index: Edge connections [2, num_edges]
            edge_weights: Edge weights [num_edges] (optional)

        Returns:
            Updated node features [num_nodes, out_features * num_heads]
            or [num_nodes, out_features] if concat=False
        """
        num_nodes = x.size(0)

        # Multi-head attention
        head_outputs = []

        for head in range(self.num_heads):
            # Linear transformation
            h = self.W[head](x)  # [num_nodes, out_features]

            # Compute attention
            attention = self._compute_attention(h, edge_index, edge_weights, head)

            # Apply attention and aggregate
            h_prime = self._aggregate(h, edge_index, attention)

            head_outputs.append(h_prime)

        # Combine heads
        if self.concat:
            output = torch.cat(head_outputs, dim=-1)
        else:
            output = torch.mean(torch.stack(head_outputs), dim=0)

        return output

    def _compute_attention(
        self,
        h: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weights: torch.Tensor | None,
        head: int,
    ) -> torch.Tensor:
        """
        Compute attention coefficients (GATv2 style).

        Args:
            h: Transformed features [num_nodes, out_features]
            edge_index: Edge connections [2, num_edges]
            edge_weights: Edge weights [num_edges]
            head: Head index

        Returns:
            Attention coefficients [num_edges]
        """
        num_edges = edge_index.size(1)

        # Get source and target features
        source_idx = edge_index[0]  # [num_edges]
        target_idx = edge_index[1]  # [num_edges]

        h_source = h[source_idx]  # [num_edges, out_features]
        h_target = h[target_idx]  # [num_edges, out_features]

        # Concatenate source and target (GATv2)
        h_cat = torch.cat([h_source, h_target], dim=-1)  # [num_edges, 2*out_features]

        # Compute attention scores
        e = self.leakyrelu(torch.matmul(h_cat, self.a[head]))  # [num_edges, 1]
        e = e.squeeze(-1)  # [num_edges]

        # Incorporate edge weights if provided
        if self.use_edge_weights and edge_weights is not None:
            e = e * edge_weights

        # Apply softmax per target node
        attention = self._edge_softmax(e, target_idx, h.size(0))

        # Apply dropout
        attention = self.dropout(attention)

        return attention

    def _edge_softmax(
        self, scores: torch.Tensor, target_idx: torch.Tensor, num_nodes: int
    ) -> torch.Tensor:
        """
        Apply softmax per target node.

        Args:
            scores: Edge scores [num_edges]
            target_idx: Target node indices [num_edges]
            num_nodes: Total number of nodes

        Returns:
            Normalized attention [num_edges]
        """
        # Compute max per node for numerical stability
        max_scores = torch.full((num_nodes,), float("-inf"), device=scores.device)
        max_scores.scatter_reduce_(0, target_idx, scores, reduce="amax", include_self=False)

        # Subtract max and exponentiate
        scores_shifted = scores - max_scores[target_idx]
        exp_scores = torch.exp(scores_shifted)

        # Sum per node
        exp_sums = torch.zeros(num_nodes, device=scores.device)
        exp_sums.scatter_add_(0, target_idx, exp_scores)

        # Normalize
        attention = exp_scores / (exp_sums[target_idx] + 1e-16)

        return attention

    def _aggregate(
        self, h: torch.Tensor, edge_index: torch.Tensor, attention: torch.Tensor
    ) -> torch.Tensor:
        """
        Aggregate neighbor features using attention.

        Args:
            h: Node features [num_nodes, out_features]
            edge_index: Edge connections [2, num_edges]
            attention: Attention coefficients [num_edges]

        Returns:
            Aggregated features [num_nodes, out_features]
        """
        num_nodes = h.size(0)
        out_features = h.size(1)

        source_idx = edge_index[0]
        target_idx = edge_index[1]

        # Weight source features by attention
        weighted_features = h[source_idx] * attention.unsqueeze(-1)

        # Aggregate to target nodes
        aggregated = torch.zeros(num_nodes, out_features, device=h.device, dtype=h.dtype)
        aggregated.scatter_add_(
            0, target_idx.unsqueeze(-1).expand_as(weighted_features), weighted_features
        )

        return aggregated


class GraphAttentionNetwork(nn.Module):
    """
    Multi-layer Graph Attention Network.

    Stacks multiple GAT layers with residual connections and
    produces final 32-dimensional structural embedding.

    Example:
        >>> config = GNNConfig(input_dim=64, output_dim=32, num_layers=2)
        >>> gnn = GraphAttentionNetwork(config)
        >>>
        >>> # Node features (e.g., price returns, volatility, volume)
        >>> node_features = torch.randn(10, 64)  # 10 nodes
        >>> edge_index = torch.tensor([[0,1,2], [1,2,0]])  # 3 edges
        >>> edge_weights = torch.tensor([0.7, 0.8, 0.6])
        >>>
        >>> embedding = gnn(node_features, edge_index, edge_weights)
        >>> # embedding.shape = [10, 32]
    """

    def __init__(self, config: GNNConfig):
        """
        Initialize GNN.

        Args:
            config: GNN configuration
        """
        super().__init__()

        self.config = config

        # Input projection
        self.input_proj = nn.Linear(config.input_dim, config.hidden_dim)

        # GAT layers
        self.gat_layers = nn.ModuleList()

        for i in range(config.num_layers):
            # First layer
            if i == 0:
                in_dim = config.hidden_dim
            else:
                # After first layer, dimension depends on concat
                if config.concat_heads:
                    in_dim = config.hidden_dim * config.num_heads
                else:
                    in_dim = config.hidden_dim

            # Last layer outputs to output_dim
            if i == config.num_layers - 1:
                out_dim = (
                    config.output_dim // config.num_heads
                    if config.concat_heads
                    else config.output_dim
                )
                concat = config.concat_heads
            else:
                out_dim = config.hidden_dim
                concat = config.concat_heads

            layer = GATv2Layer(
                in_features=in_dim,
                out_features=out_dim,
                num_heads=config.num_heads,
                dropout=config.dropout,
                concat=concat,
                use_edge_weights=config.use_edge_features,
            )

            self.gat_layers.append(layer)

        # Layer normalization
        self.layer_norms = nn.ModuleList(
            [
                nn.LayerNorm(
                    config.hidden_dim * config.num_heads
                    if config.concat_heads
                    else config.hidden_dim
                )
                for _ in range(config.num_layers - 1)
            ]
        )

        # Output layer norm
        self.output_norm = nn.LayerNorm(config.output_dim)

        logger.info(
            f"GraphAttentionNetwork initialized: "
            f"{config.num_layers} layers, output_dim={config.output_dim}"
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weights: torch.Tensor | None = None,
        return_attention: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass through GNN.

        Args:
            x: Node features [num_nodes, input_dim]
            edge_index: Edge connections [2, num_edges]
            edge_weights: Edge weights [num_edges] (optional)
            return_attention: Return attention weights

        Returns:
            Node embeddings [num_nodes, output_dim]
        """
        # Input projection
        h = self.input_proj(x)
        h = F.relu(h)

        # Pass through GAT layers
        for i, gat_layer in enumerate(self.gat_layers):
            h_new = gat_layer(h, edge_index, edge_weights)

            # Residual connection (except first layer)
            if i > 0 and h.size(-1) == h_new.size(-1):
                h_new = h_new + h

            h = h_new

            # Layer norm and activation (except last layer)
            if i < len(self.gat_layers) - 1:
                h = self.layer_norms[i](h)
                h = F.elu(h)

        # Output normalization
        h = self.output_norm(h)

        return h

    def get_node_embedding(
        self,
        node_features: np.ndarray,
        edge_index: np.ndarray,
        edge_weights: np.ndarray | None = None,
        node_idx: int | None = None,
    ) -> np.ndarray:
        """
        Get embedding for specific node or all nodes.

        Args:
            node_features: Node features [num_nodes, input_dim]
            edge_index: Edge connections [2, num_edges]
            edge_weights: Edge weights [num_edges]
            node_idx: Specific node index (None = all nodes)

        Returns:
            Node embedding(s)
        """
        with torch.no_grad():
            # Convert to tensors
            x = torch.FloatTensor(node_features)
            edges = torch.LongTensor(edge_index)

            if edge_weights is not None:
                weights = torch.FloatTensor(edge_weights)
            else:
                weights = None

            # Forward pass
            embeddings = self.forward(x, edges, weights)

            # Convert back to numpy
            embeddings_np = embeddings.numpy()

            # Return specific node or all
            if node_idx is not None:
                return embeddings_np[node_idx]
            else:
                return embeddings_np
