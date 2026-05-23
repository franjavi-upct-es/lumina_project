# backend/feature_store/definitions.py
"""Formal feature definitions. Single source of truth for the Feature Store."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from backend.config.constants import DIM_GRAPH, DIM_PRICE, DIM_SEMANTIC, OHLCV_WINDOW_MINUTES

FeatureSource = Literal["timescale", "redis", "computed"]
FeatureDType = Literal["float32", "int64", "string"]
UpdateFrequency = Literal["tick", "1min", "1hour", "daily", "weekly"]


class FeatureDef(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    source: FeatureSource
    dim: int = Field(gt=0)
    dtype: FeatureDType
    ttl_seconds: int | None = None
    update_frequency: UpdateFrequency
    description: str

    @property
    def is_hot(self) -> bool:
        return self.source == "redis"

    @property
    def is_cold(self) -> bool:
        return self.source == "timescale"


PRICE_EMBEDDING = FeatureDef(
    name="price_emb",
    source="redis",
    dim=DIM_PRICE,
    dtype="float32",
    ttl_seconds=5 * 60,
    update_frequency="1min",
    description="Output of the Temporal Fusion Transformer encoder for a ticker.",
)
SEMANTIC_EMBEDDING = FeatureDef(
    name="semantic_emb",
    source="redis",
    dim=DIM_SEMANTIC,
    dtype="float32",
    ttl_seconds=24 * 3600,
    update_frequency="1hour",
    description="Output of the distilled LLM encoder over recent news for a ticker.",
)
GRAPH_EMBEDDING = FeatureDef(
    name="graph_emb",
    source="redis",
    dim=DIM_GRAPH,
    dtype="float32",
    ttl_seconds=3600,
    update_frequency="daily",
    description="Node embedding from the GATv2 over the supply-chain graph.",
)
OHLCV_WINDOW = FeatureDef(
    name="ohlcv_window",
    source="timescale",
    dim=OHLCV_WINDOW_MINUTES * 5,
    dtype="float32",
    ttl_seconds=None,
    update_frequency="1min",
    description="240 minutes × 5 channels (open, high, low, close, volume) window.",
)
NEWS_WINDOW = FeatureDef(
    name="news_window",
    source="timescale",
    dim=512,
    dtype="float32",
    ttl_seconds=None,
    update_frequency="1hour",
    description="Concatenated last-24h news tokens (padded/truncated to 512).",
)

_ALL: list[FeatureDef] = [
    PRICE_EMBEDDING,
    SEMANTIC_EMBEDDING,
    GRAPH_EMBEDDING,
    OHLCV_WINDOW,
    NEWS_WINDOW,
]
FEATURE_REGISTRY: dict[str, FeatureDef] = {f.name: f for f in _ALL}


def get_feature(name: str) -> FeatureDef:
    if name not in FEATURE_REGISTRY:
        raise KeyError(f"Unknown feature '{name}'. Declared: {sorted(FEATURE_REGISTRY)}")
    return FEATURE_REGISTRY[name]


def features_by_source(source: FeatureSource) -> list[FeatureDef]:
    return [f for f in _ALL if f.source == source]
