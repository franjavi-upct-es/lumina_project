# backend/fusion/state_assembler.py
"""Live state assembler: Redis embeddings -> DeepFusionNexus -> Redis market state.

The class also exposes an inline ``build()`` path used by the Spartan Arena
(Phase X.3 of the Arena roadmap). When the arena attaches the three
encoders directly to the assembler, ``build()`` runs the full
TFT + LLM + GAT + Nexus pipeline inline and returns the market-state
vector plus a :class:`RawAttributionTensors` payload — the raw byproducts
of the same forward pass. The on-the-wire ``AttributionPayload`` schema is
intentionally *not* materialised here: the fusion layer owns tensors, and
the consumer (the arena) is responsible for reducing them to its schema.
This keeps the dependency direction clean (fusion has no knowledge of
simulation) and lets the live path skip the schema work entirely.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
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


@dataclass(frozen=True)
class RawAttributionTensors:
    """Raw byproducts of an attribution-capturing forward pass.

    Fusion owns this dataclass on purpose — the live execution layer must
    not have to import simulation schemas just to type a return value.
    Consumers (e.g. the arena runner) reduce these tensors into their own
    on-the-wire schema via ``backend.simulation.xai.attribution_extractor``.

    Attributes
    ----------
    cross_modal_weights
        ``(3,)`` softmaxed tensor — price/news/graph weights.
    vsn_weights_by_feature
        Mapping ``feature_name -> (T,)`` 1-D tensor (TFT VSN weights).
    gat_edge_index
        ``(2, E)`` long tensor of source/target node indices.
    gat_alpha
        ``(E,)`` tensor of last-layer GAT attention coefficients.
    ticker_list
        Tuple aligning GAT node indices to ticker symbols.
    """

    cross_modal_weights: torch.Tensor
    vsn_weights_by_feature: dict[str, torch.Tensor]
    gat_edge_index: torch.Tensor
    gat_alpha: torch.Tensor
    ticker_list: tuple[str, ...]


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


class StateAssembler:
    """Two-mode fusion driver.

    ``arena_mode=False`` (default): live operation. ``run()`` consumes
    embeddings from Redis on a fixed cadence, produces market states, and
    publishes them back to Redis. Calling ``build()`` in this mode raises.

    ``arena_mode=True``: offline simulation. ``build()`` is callable with
    inline encoders attached via ``attach_encoders``. Calling ``run()`` in
    this mode raises. This invariant is enforced at construction time so
    no single instance can ever serve both paths — that prevents the
    capture-attribution overhead from ever leaking into the live reflex
    arc by accident.
    """

    def __init__(
        self,
        model: DeepFusionNexus,
        redis: RedisCache,
        feature_client: FeatureStoreClient,
        device: str = "cuda",
        interval_s: float = 1.0,
        uncertainty_samples: int = 20,
        compute_uncertainty_every_n: int = 5,
        *,
        arena_mode: bool = False,
    ):
        self.model = model.to(device).eval()
        self.device = device
        self.redis = redis
        self.client = feature_client
        self.interval_s = interval_s
        self.uncertainty_samples = uncertainty_samples
        self.compute_uncertainty_every_n = compute_uncertainty_every_n
        self.arena_mode = bool(arena_mode)
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
        without an extra forward pass. Only callable when ``arena_mode``
        was set at construction.
        """
        if not self.arena_mode:
            raise RuntimeError(
                "attach_encoders() is arena-only. Construct StateAssembler with "
                "arena_mode=True for offline / simulation use."
            )
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
        capture_attribution: bool = False,
    ) -> tuple[torch.Tensor, RawAttributionTensors | None]:
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
        capture_attribution
            When ``True``, also returns the :class:`RawAttributionTensors`
            captured from this forward pass. Schema reduction is done by
            the caller — fusion intentionally does not know about
            ``AttributionPayload``.
        """
        if not self.arena_mode:
            raise RuntimeError(
                "StateAssembler.build() is arena-only. The live ``_cycle`` path "
                "runs without attribution capture by design — accidentally calling "
                "build() from the live loop would re-introduce the latency we "
                "explicitly engineered out. Construct with arena_mode=True for "
                "simulation."
            )
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

        # When capture_attribution is True the three branches above always
        # populated these; assert so the type checker can see it.
        assert modality_weights is not None
        assert vsn_weights is not None
        assert alpha is not None and alpha_edge_index is not None

        raw = RawAttributionTensors(
            cross_modal_weights=modality_weights.squeeze(0),
            vsn_weights_by_feature={
                name: weights.squeeze(0) for name, weights in vsn_weights.items()
            },
            gat_edge_index=alpha_edge_index,
            gat_alpha=alpha,
            ticker_list=tuple(ticker_list),
        )
        return state, raw

    async def run(self, tickers: list[str] | None = None) -> None:
        if self.arena_mode:
            raise RuntimeError(
                "StateAssembler.run() is the live loop; do not call it on an "
                "arena_mode=True instance. Use build() instead, or construct a "
                "separate assembler with arena_mode=False for live operation."
            )
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
                # return_attention=True yields (B, Heads, 3, 3) cross-modal attention
                out = self.model(price_t, semantic_t, graph_t, return_attention=True)
            states = out["market_state"].cpu().numpy().astype(np.float32)
            
            # Collapse to (B, 3) modality weights for the dashboard.
            # out["attention_weights"] is (B, H, 3, 3). 
            # We take mean over heads (dim 1) and mean over queries (dim 2) to get 
            # the importance of each modality as a key (dim 3).
            attn_weights = out["attention_weights"].mean(dim=1).mean(dim=1).cpu().numpy().astype(np.float32)

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
            pipe.set(_k_state_attention(ticker), attn_weights[i].tobytes(), ex=60)
            if uncertainties is not None:
                pipe.set(_k_state_uncertainty(ticker), float(uncertainties[i]), ex=300)
            STATES_ASSEMBLED.labels(ticker=ticker).inc()
        await pipe.execute()
        await self.redis.client.publish(
            CHANNEL_STATE_UPDATES,
            f'{{"n": {len(ready_tickers)}, "ts": "{now}"}}',
        )
