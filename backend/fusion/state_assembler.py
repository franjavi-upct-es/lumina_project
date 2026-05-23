# backend/fusion/state_assembler.py
"""Live state assembler: Redis embeddings -> DeepFusionNexus -> Redis market state.

The class also exposes an inline ``build()`` path used by the Spartan Arena
(Phase X.3 of the Arena roadmap). When the arena attaches the three
encoders directly to the assembler, ``build()`` runs the full
TFT + LLM + GAT + Nexus pipeline inline and returns both the 256-d
market-state vector and an optional :class:`AttributionPayload`
containing the VSN / GAT / cross-modal weights collected during that
same forward pass — i.e. no extra inference cost.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import numpy as np
import torch
from loguru import logger
from prometheus_client import Counter, Histogram

from backend.config.constants import TARGET_TICKERS
from backend.data_engine.storage.redis_cache import RedisCache
from backend.feature_store.client import FeatureStoreClient
from backend.fusion.nexus import DeepFusionNexus

if TYPE_CHECKING:
    from backend.perception.semantic.distilled_llm import DistilledFinancialEncoder
    from backend.perception.structural.gat_model import GraphEncoder
    from backend.perception.temporal.tft_model import TemporalFusionTransformer
    from backend.simulation.arena.schemas import AttributionPayload

STATES_ASSEMBLED = Counter(
    "fusion_states_assembled_total", "Market states written", labelnames=("ticker",)
)
STATE_ASSEMBLY_LATENCY = Histogram(
    "fusion_state_latency_seconds",
    "Time from embeddings read to state written",
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5),
)
INCOMPLETE_BUNDLES = Counter(
    "fusion_incomplete_bundles_total",
    "Tickers skipped for missing embeddings",
    labelnames=("missing",),
)

CHANNEL_STATE_UPDATES = "channel:state.updates"


def _k_market_state(ticker: str) -> str:
    return f"state:market:{ticker}"


def _k_state_uncertainty(ticker: str) -> str:
    return f"state:uncertainty:{ticker}"


def _default_feature_state_labeler(_name: str, _value: float) -> str:
    """Conservative fallback when no labeler is wired by the caller."""
    return "neutral"


class StateAssembler:
    def __init__(
        self,
        model: DeepFusionNexus,
        redis: RedisCache,
        feature_client: FeatureStoreClient,
        device: str = "cuda",
        interval_s: float = 1.0,
        uncertainty_samples: int = 20,
        compute_uncertainty_every_n: int = 5,
    ):
        self.model = model.to(device).eval()
        self.device = device
        self.redis = redis
        self.client = feature_client
        self.interval_s = interval_s
        self.uncertainty_samples = uncertainty_samples
        self.compute_uncertainty_every_n = compute_uncertainty_every_n
        self._running = False
        self._tick_counter = 0
        # Inline encoders (Arena mode). Populated via :meth:`attach_encoders`.
        self._tft: TemporalFusionTransformer | None = None
        self._llm: DistilledFinancialEncoder | None = None
        self._gat: GraphEncoder | None = None
        self._ticker_list: list[str] | None = None
        if feature_client.mode != "online":
            raise ValueError("StateAssembler requires FeatureStoreClient in online mode")

    # ------------------------------------------------------------------
    # Arena-mode helpers (Phase X.3 / X.5 of the Spartan Arena roadmap)
    # ------------------------------------------------------------------
    def attach_encoders(
        self,
        tft: TemporalFusionTransformer,
        llm: DistilledFinancialEncoder,
        gat: GraphEncoder,
        ticker_list: list[str],
    ) -> None:
        """Wire the three encoders for inline ``build()`` calls.

        Live deployments do not call this — the encoders run as separate
        services that publish to Redis. The Spartan Arena, however, runs
        the entire pipeline inline so it can capture attribution data
        without an extra forward pass.
        """
        self._tft = tft.to(self.device).eval()
        self._llm = llm.to(self.device).eval()
        self._gat = gat.to(self.device).eval()
        self._ticker_list = list(ticker_list)

    def build(
        self,
        ticker: str,
        *,
        price_window: torch.Tensor,
        news_input_ids: torch.Tensor,
        news_attention_mask: torch.Tensor,
        graph_x: torch.Tensor,
        graph_edge_index: torch.Tensor,
        graph_edge_attr: torch.Tensor,
        feature_state_labeler: Callable[[str, float], str] | None = None,
        capture_attribution: bool = False,
    ) -> tuple[torch.Tensor, AttributionPayload | None]:
        """Run the full Chimera pipeline for a single ticker.

        Parameters
        ----------
        ticker
            Symbol whose row in ``ticker_list`` (attached at construction)
            corresponds to the node we want to extract from the GAT output.
        price_window
            ``(1, T, F)`` tensor — the TFT input.
        news_input_ids, news_attention_mask
            ``(1, S)`` tensors — the LLM input.
        graph_x, graph_edge_index, graph_edge_attr
            The graph batch — node features, edge indices, edge features.
        feature_state_labeler
            Optional callable used by the attribution extractor to label
            VSN features (e.g. ``("rsi_14", 72.3) -> "overbought"``).
        capture_attribution
            When ``True``, also computes and returns the
            :class:`AttributionPayload` for this decision step.
        """
        if self._tft is None or self._llm is None or self._gat is None:
            raise RuntimeError(
                "StateAssembler.build requires inline encoders. "
                "Call attach_encoders(...) before invoking build()."
            )
        assert self._ticker_list is not None
        ticker_list = self._ticker_list
        try:
            node_idx = ticker_list.index(ticker)
        except ValueError as exc:
            raise ValueError(
                f"Ticker {ticker!r} not in attached ticker_list (len={len(ticker_list)})"
            ) from exc

        with torch.no_grad():
            # --- TFT (capture VSN weights iff requested) ----------------
            if capture_attribution:
                price_emb_full, vsn_weights = self._tft(price_window, return_vsn_weights=True)
            else:
                price_emb_full, _ = self._tft(price_window)
                vsn_weights = None
            # --- LLM ---------------------------------------------------
            semantic_emb_full, _ = self._llm(news_input_ids, news_attention_mask)
            # --- GAT (capture per-edge attention iff requested) --------
            if capture_attribution:
                gat_out, alpha_edge_index, alpha = self._gat(
                    graph_x,
                    graph_edge_index,
                    graph_edge_attr,
                    return_attention_weights=True,
                )
            else:
                gat_out = self._gat(graph_x, graph_edge_index, graph_edge_attr)
                alpha_edge_index = None
                alpha = None
            graph_emb_full = gat_out[node_idx : node_idx + 1]
            # --- Nexus (capture cross-modal weights iff requested) -----
            if capture_attribution:
                fused, modality_weights = self.model.cross_attention(
                    price_emb_full,
                    semantic_emb_full,
                    graph_emb_full,
                    return_modality_weights=True,
                )
                gate = self.model.gate(fused)
                gated = fused * gate
                state = self.model.head(gated)
            else:
                out = self.model(price_emb_full, semantic_emb_full, graph_emb_full)
                state = out["market_state"]
                modality_weights = None

        if not capture_attribution:
            return state, None

        # Defer the heavy import to avoid the circular module-load risk.
        from backend.simulation.xai.attribution_extractor import extract_attribution

        attribution = extract_attribution(
            cross_modal_weights_tensor=modality_weights.squeeze(0),
            vsn_weights_by_feature={
                name: weights.squeeze(0) for name, weights in (vsn_weights or {}).items()
            },
            gat_edge_index=alpha_edge_index if alpha_edge_index is not None else graph_edge_index,
            gat_alpha=alpha if alpha is not None else torch.zeros(graph_edge_index.size(1)),
            ticker_list=ticker_list,
            feature_state_labeler=feature_state_labeler or _default_feature_state_labeler,
            top_k=5,
        )
        return state, attribution

    async def run(self, tickers: list[str] | None = None) -> None:
        self._running = True
        tickers = tickers or list(TARGET_TICKERS)
        logger.info(f"StateAssembler started on {len(tickers)} tickers")
        while self._running:
            await self._cycle(tickers)
            await asyncio.sleep(self.interval_s)

    async def stop(self) -> None:
        self._running = False

    async def _cycle(self, tickers: list[str]) -> None:
        self._tick_counter += 1
        compute_unc = self._tick_counter % self.compute_uncertainty_every_n == 0
        bundles = await asyncio.gather(
            *[self.client.get_bundle(t) for t in tickers],
            return_exceptions=True,
        )
        ready_tickers: list[str] = []
        price_list, semantic_list, graph_list = [], [], []
        for ticker, bundle in zip(tickers, bundles, strict=True):
            if isinstance(bundle, BaseException):
                logger.error(f"Bundle fetch failed for {ticker}: {bundle}")
                continue
            # mypy: after the isinstance guard, `bundle` is the dict.
            missing = [k for k in ("price_emb", "semantic_emb", "graph_emb") if k not in bundle]
            if missing:
                for m in missing:
                    INCOMPLETE_BUNDLES.labels(missing=m).inc()
                continue
            ready_tickers.append(ticker)
            price_list.append(bundle["price_emb"])
            semantic_list.append(bundle["semantic_emb"])
            graph_list.append(bundle["graph_emb"])
        if not ready_tickers:
            return
        with STATE_ASSEMBLY_LATENCY.time():
            price_t = torch.from_numpy(np.stack(price_list)).to(self.device)
            semantic_t = torch.from_numpy(np.stack(semantic_list)).to(self.device)
            graph_t = torch.from_numpy(np.stack(graph_list)).to(self.device)
            with torch.no_grad():
                out = self.model(price_t, semantic_t, graph_t)
            states = out["market_state"].cpu().numpy().astype(np.float32)
            uncertainties: np.ndarray | None = None
            if compute_unc:
                _, std = self.model.encode_with_uncertainty(
                    price_t,
                    semantic_t,
                    graph_t,
                    n_samples=self.uncertainty_samples,
                )
                uncertainties = std.mean(dim=-1).cpu().numpy().astype(np.float32)
        now = datetime.now(UTC).isoformat()
        pipe = self.redis.client.pipeline()
        for i, ticker in enumerate(ready_tickers):
            pipe.set(_k_market_state(ticker), states[i].tobytes(), ex=60)
            if uncertainties is not None:
                pipe.set(_k_state_uncertainty(ticker), float(uncertainties[i]), ex=300)
            STATES_ASSEMBLED.labels(ticker=ticker).inc()
        await pipe.execute()
        await self.redis.client.publish(
            CHANNEL_STATE_UPDATES,
            f'{{"n": {len(ready_tickers)}, "ts": "{now}"}}',
        )
