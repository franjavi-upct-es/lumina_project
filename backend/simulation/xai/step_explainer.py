# backend/simulation/xai/step_explainer.py
"""Render a one-block terminal-style explanation per decision.

The renderer is intentionally a pure template — it never calls a model
and does not import torch. This keeps it cheap (one template per
decision = thousands per arena run) and lets it run inside an async
hot path without backpressure.
"""

from __future__ import annotations

import math

from backend.config.constants import ARENA_DIVERGENCE_HORIZON_BARS
from backend.simulation.arena.schemas import (
    DecisionRecord,
    DivergencePoint,
    StepExplanation,
)


def format_decision(
    record: DecisionRecord,
    divergence: DivergencePoint | None = None,
) -> StepExplanation:
    """Build a :class:`StepExplanation` for one decision.

    Output layout::

        [<sim_timestamp>]  <ACTION> <size>%  |  conf=0.78  unc=0.22
          WHY: <top_vsn_feature> <state> (w=0.41) · <top_edge_target> contagion via GATv2 (edge=0.62)
               Dominant modality: news (0.51) > price (0.32) > graph (0.17)
          VS:  T2 chose SELL → +0.84 Sharpe delta over next 30 bars
    """
    cm = record.attribution.cross_modal
    modality_order = sorted(
        [("price", cm.price), ("news", cm.news), ("graph", cm.graph)],
        key=lambda kv: kv[1],
        reverse=True,
    )

    size_pct = _action_size_pct(record.action_vector)
    header = (
        f"[{record.sim_timestamp.isoformat()}]  "
        f"{record.action_kind.value} {size_pct:d}%  |  "
        f"conf={record.confidence:.2f}  unc={record.uncertainty:.2f}"
    )

    vsn_top = record.attribution.tft_vsn_top[0] if record.attribution.tft_vsn_top else None
    gat_top = record.attribution.gat_edges_top[0] if record.attribution.gat_edges_top else None
    why_left = (
        f"{vsn_top.feature} {vsn_top.state} (w={vsn_top.weight:.2f})"
        if vsn_top is not None
        else "no dominant feature"
    )
    why_right = (
        f"{gat_top.target_ticker} contagion via GATv2 (edge={gat_top.coefficient:.2f})"
        if gat_top is not None
        else "no graph contagion"
    )
    why_line = f"  WHY: {why_left} · {why_right}"

    dominant_line = "       Dominant modality: " + " > ".join(
        f"{name} ({value:.2f})" for name, value in modality_order
    )

    lines = [header, why_line, dominant_line]
    if divergence is not None:
        peer = (
            divergence.best_trajectory_id
            if divergence.best_trajectory_id != record.trajectory_id
            else divergence.worst_trajectory_id
        )
        peer_kind = _peer_action_kind(divergence, peer)
        delta = divergence.sharpe_delta
        sign = "+" if delta >= 0 else ""
        lines.append(
            f"  VS:  T{peer} chose {peer_kind} -> {sign}{delta:.1f} Sharpe delta over next "
            f"{ARENA_DIVERGENCE_HORIZON_BARS} bars"
        )

    tags = _build_tags(record)
    return StepExplanation(record_id=record.record_id, text="\n".join(lines), tags=tags)


# ----------------------------------------------------------------------
def _action_size_pct(action_vector: list[float]) -> int:
    """L2 norm of the action vector, scaled into [0, 100]."""
    norm = math.sqrt(sum(a * a for a in action_vector))
    return round(min(100.0, max(0.0, norm * 100.0)))


def _peer_action_kind(divergence: DivergencePoint, peer_trajectory_id: int) -> str:
    """Sketch the peer's action sign for the VS: line.

    We don't carry the peer's ``ActionKind`` in the divergence record,
    only their raw action vector. We re-derive the direction from
    ``action_vector[0]`` (the canonical direction slot).
    """
    if peer_trajectory_id == divergence.best_trajectory_id:
        direction = divergence.best_action_vector[0]
    else:
        direction = divergence.worst_action_vector[0]
    if direction > 0.05:
        return "BUY"
    if direction < -0.05:
        return "SELL"
    return "HOLD"


def _build_tags(record: DecisionRecord) -> list:
    """Structured tag index — used by the dashboard explanation panel filter."""
    tags: list[str] = []
    for vsn in record.attribution.tft_vsn_top:
        if vsn.state == "overbought" and "overbought" not in tags:
            tags.append("overbought")
        if vsn.state == "oversold" and "oversold" not in tags:
            tags.append("oversold")
    for edge in record.attribution.gat_edges_top:
        if edge.coefficient > 0.7 and "contagion" not in tags:
            tags.append("contagion")
            break
    if record.uncertainty > 0.5 and "low_confidence" not in tags:
        tags.append("low_confidence")
    if (
        record.confidence > 0.85
        and record.attribution.cross_modal.news > 0.5
        and "panic" not in tags
    ):
        tags.append("panic")
    return tags
