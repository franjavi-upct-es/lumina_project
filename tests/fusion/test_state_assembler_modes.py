"""Regression tests for the StateAssembler mode invariants.

These lock in the contract introduced by the v3 hardening pass:

* An assembler constructed with ``arena_mode=False`` (the live default)
  must reject ``build()`` and ``attach_encoders()`` — otherwise the
  capture-attribution overhead could silently leak into the live reflex
  arc.
* An assembler constructed with ``arena_mode=True`` must reject
  ``run()`` — that's the live loop and has no business running on a
  simulation instance.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import torch

from backend.data_engine.storage.redis_cache import RedisCache
from backend.feature_store.client import FeatureStoreClient
from backend.fusion.nexus import DeepFusionNexus
from backend.fusion.state_assembler import RawAttributionTensors, StateAssembler


def _build_assembler(*, arena_mode: bool) -> StateAssembler:
    """Build an assembler with mocked Redis / feature-client dependencies."""
    redis = MagicMock(spec=RedisCache)
    feature_client = MagicMock(spec=FeatureStoreClient)
    feature_client.mode = "online"
    return StateAssembler(
        model=DeepFusionNexus(),
        redis=redis,
        feature_client=feature_client,
        device="cpu",
        arena_mode=arena_mode,
    )


def test_live_assembler_rejects_build() -> None:
    """build() must not be callable on a live (arena_mode=False) instance."""
    assembler = _build_assembler(arena_mode=False)
    dummy = torch.zeros(1, 1, 1)
    with pytest.raises(RuntimeError, match="arena-only"):
        assembler.build(
            "AAPL",
            price_window=dummy,
            news_input_ids=dummy,
            news_attention_mask=dummy,
            graph_x=dummy,
            graph_edge_index=dummy,
            graph_edge_attr=dummy,
            capture_attribution=False,
        )


def test_live_assembler_rejects_attach_encoders() -> None:
    """attach_encoders() must not be callable on a live instance."""
    assembler = _build_assembler(arena_mode=False)
    with pytest.raises(RuntimeError, match="arena-only"):
        assembler.attach_encoders(
            tft=MagicMock(),
            llm=MagicMock(),
            gat=MagicMock(),
            ticker_list=["AAPL"],
        )


@pytest.mark.asyncio
async def test_arena_assembler_rejects_run() -> None:
    """run() must not be callable on an arena (arena_mode=True) instance."""
    assembler = _build_assembler(arena_mode=True)
    with pytest.raises(RuntimeError, match="live loop"):
        await assembler.run(tickers=["AAPL"])


def test_raw_attribution_tensors_is_frozen_dataclass() -> None:
    """RawAttributionTensors must be immutable so consumers can't accidentally
    mutate the byproducts of a forward pass after capture."""
    raw = RawAttributionTensors(
        cross_modal_weights=torch.tensor([0.3, 0.4, 0.3]),
        vsn_weights_by_feature={"rsi_14": torch.tensor([0.5])},
        gat_edge_index=torch.tensor([[0], [1]]),
        gat_alpha=torch.tensor([0.7]),
        ticker_list=("AAPL", "MSFT"),
    )
    with pytest.raises((AttributeError, Exception)):
        raw.cross_modal_weights = torch.tensor([1.0, 0.0, 0.0])  # type: ignore[misc]
