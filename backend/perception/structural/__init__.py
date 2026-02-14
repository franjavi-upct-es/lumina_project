# backend/perception/structural/__init__.py
"""
Structural Encoder - Graph Neural Network (GNN)

Layer 1.C of the V3 Chimera Architecture - "Spatial Awareness"

Models the market as a graph where nodes are assets and edges are relationships.
Understands the "contagion" effect of the market web.

Problem Solved:
Standard models treat assets as Independent and Identically Distributed (I.I.D.)
islands. In reality, stocks don't move in a vacuum:
- If NVDA drops, AMD reacts (competitors - correlation)
- SMCI follows (supply chain partners - causality)

GNN Solution:
Graph Attention Network (GATv2) performs "Message Passing" to aggregate
information from neighbor nodes.

Example:
If "Semiconductor" neighbors are crashing, the GNN embedding for AAPL
will reflect "neighborhood stress" even if AAPL itself hasn't dropped yet.
It predicts the shockwave before it hits.

Architecture:
- Nodes: Assets (SPY, AAPL, MSFT, VIX, Oil, Gold, etc.)
- Edges:
  * Dynamic: Rolling 30-day price correlation (updated daily)
  * Static: Sector membership, supply chain relationships, ETF weightings
- Output: 32-dimensional embedding representing market structure pressure

Components:
- graph_builder: Constructs market graph from assets and relationships
- dynamic_edges: Computes rolling correlation edges
- gnn_model: Graph Attention Network (GATv2) for message passing
"""

from backend.perception.structural.dynamic_edges import (
    CorrelationConfig,
    DynamicEdgeComputer,
    compute_rolling_correlation,
)
from backend.perception.structural.gnn_model import (
    GATv2Layer,
    GNNConfig,
    GraphAttentionNetwork,
)
from backend.perception.structural.graph_builder import (
    AssetNode,
    EdgeType,
    GraphBuilder,
    MarketGraph,
)

__all__ = [
    # Graph Builder
    "MarketGraph",
    "GraphBuilder",
    "AssetNode",
    "EdgeType",
    # Dynamic Edges
    "DynamicEdgeComputer",
    "CorrelationConfig",
    "compute_rolling_correlation",
    # GNN Model
    "GraphAttentionNetwork",
    "GATv2Layer",
    "GNNConfig",
]
