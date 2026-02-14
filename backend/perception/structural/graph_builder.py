# backend/perception/structural/graph_builder.py
"""
Market Graph Builder

Constructs the market graph representation with nodes (assets) and edges
(relationships). Combines dynamic edges (correlations) with static edges
(sectors, supply chains, ETF holdings).

Graph Structure:
- Nodes: Individual assets (stocks, indices, commodities, VIX)
- Edges:
  * Dynamic: Rolling correlation (updated daily)
  * Static: Sector membership, supply chain, ETF weightings

Example Graph:
    AAPL ←→ MSFT (correlation: 0.7, same sector: Tech)
    AAPL ←→ NVDA (correlation: 0.6, supply chain: chips)
    NVDA ←→ AMD (correlation: 0.8, same sector: Semiconductors)
    SPY ←→ QQQ (correlation: 0.9, both indices)

Usage:
    >>> builder = GraphBuilder()
    >>> builder.add_asset('AAPL', sector='Technology')
    >>> builder.add_asset('MSFT', sector='Technology')
    >>> builder.add_correlation_edges(price_data)
    >>> graph = builder.build()
"""

from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import pandas as pd
from loguru import logger

from backend.perception.structural.dynamic_edges import (
    CorrelationConfig,
    DynamicEdgeComputer,
    create_sector_edges,
)


class EdgeType(Enum):
    """Types of edges in the market graph."""

    CORRELATION = "correlation"  # Dynamic price correlation
    SECTOR = "sector"  # Same sector membership
    SUPPLY_CHAIN = "supply_chain"  # Supply chain relationship
    ETF_HOLDING = "etf_holding"  # Both held in same ETF
    INDEX_CONSTITUENT = "index"  # Index membership
    COMPETITOR = "competitor"  # Direct competitors


@dataclass
class AssetNode:
    """
    Represents an asset node in the market graph.

    Attributes:
        symbol: Asset ticker symbol
        name: Full asset name
        sector: Sector classification (e.g., 'Technology', 'Energy')
        industry: Industry classification
        market_cap: Market capitalization category
        node_features: Additional node features for GNN
        metadata: Additional metadata
    """

    symbol: str
    name: str | None = None
    sector: str | None = None
    industry: str | None = None
    market_cap: str | None = None  # 'Large', 'Mid', 'Small'
    node_features: np.ndarray | None = None
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        """Initialize default values."""
        if self.name is None:
            self.name = self.symbol


@dataclass
class MarketGraph:
    """
    Complete market graph representation.

    Attributes:
        nodes: List of asset nodes
        edge_index: Edge connections [2, num_edges]
        edge_weights: Edge weights [num_edges]
        edge_types: Edge types [num_edges]
        adjacency_matrix: Full adjacency matrix [n_nodes, n_nodes]
        node_features: Node feature matrix [n_nodes, n_features]
    """

    nodes: list[AssetNode]
    edge_index: np.ndarray
    edge_weights: np.ndarray
    edge_types: list[EdgeType]
    adjacency_matrix: np.ndarray | None = None
    node_features: np.ndarray | None = None

    def get_node_index(self, symbol: str) -> int:
        """Get node index by symbol."""
        for i, node in enumerate(self.nodes):
            if node.symbol == symbol:
                return i
        raise ValueError(f"Asset {symbol} not found in graph")

    def get_neighbors(self, symbol: str) -> list[tuple[str, float, EdgeType]]:
        """
        Get neighbors of a node.

        Returns:
            List of (neighbor_symbol, edge_weight, edge_type) tuples
        """
        node_idx = self.get_node_index(symbol)
        neighbors = []

        for i in range(self.edge_index.shape[1]):
            source, target = self.edge_index[:, i]

            if source == node_idx:
                neighbor_symbol = self.nodes[target].symbol
                weight = self.edge_weights[i]
                edge_type = self.edge_types[i]
                neighbors.append((neighbor_symbol, weight, edge_type))

        return neighbors

    def get_subgraph(self, symbols: list[str]) -> "MarketGraph":
        """Extract subgraph containing only specified assets."""
        # Get indices
        indices = [self.get_node_index(s) for s in symbols]
        index_set = set(indices)

        # Filter nodes
        sub_nodes = [self.nodes[i] for i in indices]

        # Create index mapping
        old_to_new = {old_idx: new_idx for new_idx, old_idx in enumerate(indices)}

        # Filter edges
        sub_edges = []
        sub_weights = []
        sub_types = []

        for i in range(self.edge_index.shape[1]):
            source, target = self.edge_index[:, i]

            if source in index_set and target in index_set:
                new_source = old_to_new[source]
                new_target = old_to_new[target]
                sub_edges.append([new_source, new_target])
                sub_weights.append(self.edge_weights[i])
                sub_types.append(self.edge_types[i])

        if len(sub_edges) == 0:
            sub_edge_index = np.array([[], []], dtype=np.int64)
        else:
            sub_edge_index = np.array(sub_edges, dtype=np.int64).T

        return MarketGraph(
            nodes=sub_nodes,
            edge_index=sub_edge_index,
            edge_weights=np.array(sub_weights),
            edge_types=sub_types,
        )


class GraphBuilder:
    """
    Builds market graphs from assets and relationships.

    Supports incremental construction:
    1. Add asset nodes
    2. Add static edges (sectors, supply chains)
    3. Add dynamic edges (correlations)
    4. Build final graph

    Example:
        >>> builder = GraphBuilder()
        >>>
        >>> # Add tech stocks
        >>> builder.add_asset('AAPL', sector='Technology')
        >>> builder.add_asset('MSFT', sector='Technology')
        >>> builder.add_asset('NVDA', sector='Semiconductors')
        >>>
        >>> # Add relationships
        >>> builder.add_static_edge('AAPL', 'NVDA', EdgeType.SUPPLY_CHAIN, 0.8)
        >>>
        >>> # Add correlations from price data
        >>> builder.add_correlation_edges(price_df)
        >>>
        >>> # Build
        >>> graph = builder.build()
    """

    def __init__(self, correlation_config: CorrelationConfig | None = None):
        """
        Initialize graph builder.

        Args:
            correlation_config: Configuration for correlation edges
        """
        self.nodes: list[AssetNode] = []
        self.edges: list[tuple[int, int, float, EdgeType]] = []

        self.symbol_to_index: dict[str, int] = {}
        self.correlation_config = correlation_config or CorrelationConfig()

        logger.debug("GraphBuilder initialized")

    def add_asset(
        self,
        symbol: str,
        name: str | None = None,
        sector: str | None = None,
        industry: str | None = None,
        market_cap: str | None = None,
        **metadata,
    ) -> int:
        """
        Add asset node to graph.

        Args:
            symbol: Asset ticker
            name: Full name
            sector: Sector classification
            industry: Industry classification
            market_cap: Market cap category
            **metadata: Additional metadata

        Returns:
            Node index
        """
        if symbol in self.symbol_to_index:
            logger.warning(f"Asset {symbol} already exists, skipping")
            return self.symbol_to_index[symbol]

        node = AssetNode(
            symbol=symbol,
            name=name,
            sector=sector,
            industry=industry,
            market_cap=market_cap,
            metadata=metadata,
        )

        node_idx = len(self.nodes)
        self.nodes.append(node)
        self.symbol_to_index[symbol] = node_idx

        return node_idx

    def add_static_edge(self, source: str, target: str, edge_type: EdgeType, weight: float = 1.0):
        """
        Add static edge between assets.

        Args:
            source: Source asset symbol
            target: Target asset symbol
            edge_type: Type of relationship
            weight: Edge weight
        """
        if source not in self.symbol_to_index:
            raise ValueError(f"Source asset {source} not found")
        if target not in self.symbol_to_index:
            raise ValueError(f"Target asset {target} not found")

        source_idx = self.symbol_to_index[source]
        target_idx = self.symbol_to_index[target]

        # Add both directions for undirected graph
        self.edges.append((source_idx, target_idx, weight, edge_type))
        self.edges.append((target_idx, source_idx, weight, edge_type))

    def add_sector_edges(self, weight: float = 0.5):
        """
        Automatically add edges between assets in the same sector.

        Args:
            weight: Edge weight for sector connections
        """
        # Group by sector
        sector_groups: dict[str, list[int]] = {}

        for idx, node in enumerate(self.nodes):
            if node.sector is not None:
                if node.sector not in sector_groups:
                    sector_groups[node.sector] = []
                sector_groups[node.sector].append(idx)

        # Connect within sectors
        for sector, indices in sector_groups.items():
            for i in range(len(indices)):
                for j in range(i + 1, len(indices)):
                    idx_i = indices[i]
                    idx_j = indices[j]

                    # Add both directions
                    self.edges.append((idx_i, idx_j, weight, EdgeType.SECTOR))
                    self.edges.append((idx_j, idx_i, weight, EdgeType.SECTOR))

        logger.info(f"Added sector edges for {len(sector_groups)} sectors")

    def add_correlation_edges(
        self,
        prices: pd.DataFrame,
        timestamp: pd.Timestamp | None = None,
        weight_multiplier: float = 1.0,
    ):
        """
        Add dynamic correlation edges from price data.

        Args:
            prices: DataFrame with asset prices
            timestamp: Timestamp to compute correlations (latest if None)
            weight_multiplier: Multiply correlation weights
        """
        # Get assets present in both graph and price data
        graph_symbols = set(self.symbol_to_index.keys())
        price_symbols = set(prices.columns)
        common_symbols = graph_symbols.intersection(price_symbols)

        if len(common_symbols) == 0:
            logger.warning("No common assets between graph and price data")
            return

        # Filter price data to common assets
        filtered_prices = prices[list(common_symbols)]

        # Compute correlation edges
        computer = DynamicEdgeComputer(self.correlation_config)
        edge_index, edge_weights = computer.compute_edges(filtered_prices, timestamp)

        # Map to graph indices
        symbol_list = list(common_symbols)
        symbol_to_local = {s: i for i, s in enumerate(symbol_list)}

        for i in range(edge_index.shape[1]):
            local_source, local_target = edge_index[:, i]

            source_symbol = symbol_list[local_source]
            target_symbol = symbol_list[local_target]

            source_idx = self.symbol_to_index[source_symbol]
            target_idx = self.symbol_to_index[target_symbol]

            weight = edge_weights[i] * weight_multiplier

            self.edges.append((source_idx, target_idx, weight, EdgeType.CORRELATION))

        logger.info(f"Added {edge_index.shape[1]} correlation edges")

    def build(self) -> MarketGraph:
        """
        Build final market graph.

        Returns:
            Complete MarketGraph object
        """
        if len(self.nodes) == 0:
            raise ValueError("No nodes in graph")

        # Convert edges to arrays
        if len(self.edges) == 0:
            edge_index = np.array([[], []], dtype=np.int64)
            edge_weights = np.array([])
            edge_types = []
        else:
            edge_list = [(e[0], e[1]) for e in self.edges]
            edge_index = np.array(edge_list, dtype=np.int64).T
            edge_weights = np.array([e[2] for e in self.edges], dtype=np.float32)
            edge_types = [e[3] for e in self.edges]

        # Build adjacency matrix
        n_nodes = len(self.nodes)
        adj_matrix = np.zeros((n_nodes, n_nodes))

        for i in range(edge_index.shape[1]):
            source, target = edge_index[:, i]
            adj_matrix[source, target] = edge_weights[i]

        graph = MarketGraph(
            nodes=self.nodes,
            edge_index=edge_index,
            edge_weights=edge_weights,
            edge_types=edge_types,
            adjacency_matrix=adj_matrix,
        )

        logger.success(f"Built market graph: {len(self.nodes)} nodes, {edge_index.shape[1]} edges")

        return graph

    def clear(self):
        """Clear all nodes and edges."""
        self.nodes = []
        self.edges = []
        self.symbol_to_index = {}


def create_default_market_graph(
    assets: list[str], sectors: dict[str, str] | None = None
) -> MarketGraph:
    """
    Create default market graph for common assets.

    Args:
        assets: List of asset symbols
        sectors: Optional sector mapping

    Returns:
        MarketGraph with sector connections
    """
    builder = GraphBuilder()

    # Add assets
    for asset in assets:
        sector = sectors.get(asset) if sectors else None
        builder.add_asset(asset, sector=sector)

    # Add sector edges if sectors provided
    if sectors:
        builder.add_sector_edges(weight=0.5)

    return builder.build()
