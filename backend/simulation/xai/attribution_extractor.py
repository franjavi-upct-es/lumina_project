# backend/simulation/xai/attribution_extractor.py
"""Pure helper: tensor -> :class:`AttributionPayload`.

Reads attention/weight tensors produced as byproducts of the encoder
forward passes (see Phase X.3 of the Arena roadmap) and converts them
to the schema the rest of the system speaks. Performs no model calls
and allocates no tensors of its own — everything is detach + cpu +
top-K selection.
"""

from __future__ import annotations

from collections.abc import Callable

import torch

from backend.simulation.arena.schemas import (
    AttributionPayload,
    CrossModalWeights,
    GATEdgeCoefficient,
    VSNWeight,
)

# A small registry of feature->state labelers. The arena runner can extend
# or replace it via the ``feature_state_labeler`` argument.
_DEFAULT_LABELERS: dict[str, Callable[[float], str]] = {
    "rsi_14": lambda v: "overbought" if v > 70.0 else ("oversold" if v < 30.0 else "neutral"),
    "rsi": lambda v: "overbought" if v > 70.0 else ("oversold" if v < 30.0 else "neutral"),
    "close": lambda _v: "neutral",
    "open": lambda _v: "neutral",
    "high": lambda _v: "neutral",
    "low": lambda _v: "neutral",
    "volume": lambda v: "high" if v > 1.5 else ("low" if v < 0.5 else "normal"),
}


def default_feature_state_labeler(name: str, value: float) -> str:
    """Look up the state label for a feature; falls back to ``"neutral"``."""
    fn = _DEFAULT_LABELERS.get(name)
    if fn is None:
        return "neutral"
    try:
        return fn(value)
    except Exception:
        return "neutral"


def extract_attribution(
    *,
    cross_modal_weights_tensor: torch.Tensor,
    vsn_weights_by_feature: dict[str, torch.Tensor],
    gat_edge_index: torch.Tensor,
    gat_alpha: torch.Tensor,
    ticker_list: list[str],
    feature_state_labeler: Callable[[str, float], str] | None = None,
    top_k: int = 5,
) -> AttributionPayload:
    """Reduce raw weight tensors to the on-the-wire schema.

    Parameters
    ----------
    cross_modal_weights_tensor
        Shape ``(3,)`` softmaxed tensor. ``[0]=price, [1]=news (semantic),
        [2]=graph``.
    vsn_weights_by_feature
        Mapping ``feature_name -> (T,)`` 1-D tensor (one weight per
        timestep). The function reduces each to a scalar by ``.mean()``
        before ranking.
    gat_edge_index
        Shape ``(2, E)`` long tensor with source/target node indices.
    gat_alpha
        Shape ``(E,)`` tensor of last-layer attention coefficients.
    ticker_list
        Maps node index to ticker symbol — supplied by the GAT builder.
    feature_state_labeler
        Callable used to label a VSN feature; defaults to a small
        registry shipped in this module.
    top_k
        Maximum entries per ranked list.
    """
    labeler = feature_state_labeler or default_feature_state_labeler

    # --- Cross-modal weights -------------------------------------------------
    cmw = cross_modal_weights_tensor.detach().cpu().tolist()
    if len(cmw) != 3:
        raise ValueError(f"cross_modal_weights_tensor must have length 3, got {len(cmw)}")
    # Clamp to [0, 1] to satisfy the schema even if upstream had tiny FP drift.
    price = max(0.0, min(1.0, float(cmw[0])))
    news = max(0.0, min(1.0, float(cmw[1])))
    graph = max(0.0, min(1.0, float(cmw[2])))
    cross_modal = CrossModalWeights(price=price, news=news, graph=graph)

    # --- TFT VSN top-K --------------------------------------------------------
    vsn_items: list[VSNWeight] = []
    if vsn_weights_by_feature:
        scored: list[tuple[str, float]] = []
        for name, weights_tensor in vsn_weights_by_feature.items():
            if weights_tensor.numel() == 0:
                continue
            scored.append((name, float(weights_tensor.detach().mean().item())))
        scored.sort(key=lambda p: p[1], reverse=True)
        for name, weight in scored[:top_k]:
            clamped = max(0.0, min(1.0, weight))
            vsn_items.append(VSNWeight(feature=name, weight=clamped, state=labeler(name, weight)))

    # --- GAT edge top-K -------------------------------------------------------
    gat_items: list[GATEdgeCoefficient] = []
    if gat_alpha.numel() > 0 and gat_edge_index.numel() > 0:
        alpha = gat_alpha.detach()
        if alpha.dim() > 1:
            alpha = alpha.mean(dim=-1)
        edge_index = gat_edge_index.detach().cpu()
        # Rank by |alpha|; keep sign-free since the schema bounds [0, 1].
        magnitudes = alpha.abs().cpu()
        n_edges = magnitudes.size(0)
        k = min(top_k, n_edges)
        if k > 0:
            top_values, top_idx = torch.topk(magnitudes, k)
            for v, i in zip(top_values.tolist(), top_idx.tolist(), strict=True):
                src = int(edge_index[0, i].item())
                tgt = int(edge_index[1, i].item())
                if src < 0 or src >= len(ticker_list) or tgt < 0 or tgt >= len(ticker_list):
                    continue
                gat_items.append(
                    GATEdgeCoefficient(
                        source_ticker=ticker_list[src],
                        target_ticker=ticker_list[tgt],
                        coefficient=max(0.0, min(1.0, float(v))),
                    )
                )

    return AttributionPayload(
        cross_modal=cross_modal,
        tft_vsn_top=vsn_items,
        gat_edges_top=gat_items,
        llm_top_tokens=None,
    )
