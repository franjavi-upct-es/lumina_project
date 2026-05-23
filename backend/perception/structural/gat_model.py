# backend/perception/structural/gat_model.py
"""GATv2 over the financial correlation + supply-chain graph."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv

from backend.config.constants import DIM_GRAPH


class GraphEncoder(nn.Module):
    def __init__(
        self,
        node_feat_dim: int = 32,
        edge_feat_dim: int = 4,
        hidden_dim: int = 64,
        output_dim: int = DIM_GRAPH,
        heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.gat1 = GATv2Conv(
            node_feat_dim,
            hidden_dim,
            heads=heads,
            dropout=dropout,
            edge_dim=edge_feat_dim,
            add_self_loops=True,
        )
        self.gat2 = GATv2Conv(
            hidden_dim * heads,
            hidden_dim,
            heads=heads,
            dropout=dropout,
            edge_dim=edge_feat_dim,
        )
        self.gat3 = GATv2Conv(
            hidden_dim * heads,
            output_dim,
            heads=1,
            concat=False,
            dropout=dropout,
            edge_dim=edge_feat_dim,
        )
        self.norm1 = nn.LayerNorm(hidden_dim * heads)
        self.norm2 = nn.LayerNorm(hidden_dim * heads)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        return_attention_weights: bool = False,
    ):
        """Forward pass.

        Default returns the node embeddings tensor only. When
        ``return_attention_weights=True``, also returns the last layer's
        ``(alpha_edge_index, alpha)`` pair from ``GATv2Conv`` so the Spartan
        Arena attribution extractor can pick the top-K influential edges
        without a second forward pass.
        """
        h = self.gat1(x, edge_index, edge_attr=edge_attr)
        h = self.norm1(F.elu(h))
        h = self.gat2(h, edge_index, edge_attr=edge_attr)
        h = self.norm2(F.elu(h))
        if return_attention_weights:
            out, (alpha_edge_index, alpha) = self.gat3(
                h,
                edge_index,
                edge_attr=edge_attr,
                return_attention_weights=True,
            )
            # alpha has shape (E, heads); the last layer uses heads=1, so squeeze it.
            if alpha.dim() == 2 and alpha.size(-1) == 1:
                alpha = alpha.squeeze(-1)
            return out, alpha_edge_index, alpha
        return self.gat3(h, edge_index, edge_attr=edge_attr)
