# backend/simulation/xai/run_summarizer.py
"""End-of-run narrative generator for the Spartan Arena.

Default mode is a deterministic template. The optional SLM mode is
loaded lazily and gated by ``Settings.arena.enable_run_summarizer_llm``.
On any SLM failure (file missing, OOM, timeout) we fall back to the
template summary and stamp ``summary_method = "template"`` on the result.
"""

from __future__ import annotations

import math
import statistics
from collections import Counter
from typing import Any, Literal

from loguru import logger

from backend.config.constants import ARENA_PIVOTAL_SHARPE_DELTA
from backend.config.settings import get_settings
from backend.simulation.arena.schemas import (
    ArenaRunMetadata,
    DecisionRecord,
    DivergencePoint,
    RunSummary,
)

_SLM_MAX_TOKENS: int = 200
_SLM_CACHE: dict[str, Any] = {}


def summarize_run(
    metadata: ArenaRunMetadata,
    decisions_by_trajectory: dict[int, list[DecisionRecord]],
    divergences: list[DivergencePoint],
    *,
    use_slm: bool | None = None,
) -> RunSummary:
    """Produce a :class:`RunSummary`.

    The SLM path is opt-in. When ``use_slm`` is ``None`` (the default),
    we consult ``Settings.arena.enable_run_summarizer_llm``.
    """
    if use_slm is None:
        use_slm = get_settings().arena.enable_run_summarizer_llm

    pivotal_count = sum(1 for d in divergences if d.sharpe_delta >= ARENA_PIVOTAL_SHARPE_DELTA)

    per_trajectory_sharpe = _compute_per_trajectory_sharpes(decisions_by_trajectory)
    if per_trajectory_sharpe:
        best_tid = max(per_trajectory_sharpe, key=lambda t: per_trajectory_sharpe[t])
        worst_tid = min(per_trajectory_sharpe, key=lambda t: per_trajectory_sharpe[t])
    else:
        # Pathological: no decisions recorded. Use the first/last seed indices.
        best_tid = 0
        worst_tid = max(0, metadata.n_trajectories - 1)

    template_narrative = _build_template_narrative(
        metadata=metadata,
        decisions_by_trajectory=decisions_by_trajectory,
        divergences=divergences,
        per_trajectory_sharpe=per_trajectory_sharpe,
        best_tid=best_tid,
        worst_tid=worst_tid,
        pivotal_count=pivotal_count,
    )

    summary_method: Literal["template", "slm"] = "template"
    narrative = template_narrative
    if use_slm:
        try:
            slm_text = _summarize_with_slm(
                metadata, decisions_by_trajectory, divergences, template_narrative
            )
            if slm_text:
                narrative = slm_text
                summary_method = "slm"
        except Exception as exc:
            logger.warning("SLM run-summarizer failed ({}). Falling back to template.", exc)

    return RunSummary(
        run_id=metadata.run_id,
        narrative=narrative,
        best_trajectory_id=best_tid,
        worst_trajectory_id=worst_tid,
        n_divergences=len(divergences),
        n_pivotal_divergences=pivotal_count,
        summary_method=summary_method,
    )


# ----------------------------------------------------------------------
def _compute_per_trajectory_sharpes(
    decisions_by_trajectory: dict[int, list[DecisionRecord]],
) -> dict[int, float]:
    """Annualised Sharpe per trajectory from the recorded rewards.

    Mirrors the formula in ``divergence_analyzer._sharpe_ratio`` (1-minute
    bars, 252 trading days, 390 bars/day). When the realized rewards have
    not been written back (Timescale-less unit tests), we return zeros.
    """
    annualizer = math.sqrt(252.0 * 390.0)
    result: dict[int, float] = {}
    for tid, decisions in decisions_by_trajectory.items():
        rewards = [d.realized_reward for d in decisions if d.realized_reward is not None]
        if len(rewards) < 2:
            result[tid] = 0.0
            continue
        mean = statistics.mean(rewards)
        std = statistics.pstdev(rewards) or 1e-9
        result[tid] = (mean / std) * annualizer
    return result


def _build_template_narrative(
    *,
    metadata: ArenaRunMetadata,
    decisions_by_trajectory: dict[int, list[DecisionRecord]],
    divergences: list[DivergencePoint],
    per_trajectory_sharpe: dict[int, float],
    best_tid: int,
    worst_tid: int,
    pivotal_count: int,
) -> str:
    header = (
        f"Run {metadata.run_id} | ticker={metadata.ticker} | "
        f"{metadata.start_date.date()} -> {metadata.end_date.date()} | "
        f"N={metadata.n_trajectories} trajectories"
    )

    if per_trajectory_sharpe:
        sharpes = list(per_trajectory_sharpe.values())
        perf_line = (
            f"Performance: best=T{best_tid} ({per_trajectory_sharpe[best_tid]:+.2f}) | "
            f"worst=T{worst_tid} ({per_trajectory_sharpe[worst_tid]:+.2f}) | "
            f"mean={statistics.mean(sharpes):+.2f} | "
            f"std={(statistics.pstdev(sharpes) if len(sharpes) > 1 else 0.0):.2f}"
        )
    else:
        perf_line = "Performance: no decisions recorded (run was empty)."

    div_line = f"Divergences: total={len(divergences)} | pivotal={pivotal_count}"
    top_div_lines: list[str] = []
    top_three = sorted(divergences, key=lambda d: -d.sharpe_delta)[:3]
    for d in top_three:
        top_div_lines.append(
            f"  - {d.sim_timestamp.isoformat()}  "
            f"T{d.best_trajectory_id} vs T{d.worst_trajectory_id}  "
            f"L2={d.action_l2_distance:.2f}  delta_sharpe={d.sharpe_delta:+.2f}"
        )

    modality_counter: Counter[str] = Counter()
    modality_avg: dict[str, float] = {"price": 0.0, "news": 0.0, "graph": 0.0}
    total = 0
    for decisions in decisions_by_trajectory.values():
        for record in decisions:
            cm = record.attribution.cross_modal
            modality_avg["price"] += cm.price
            modality_avg["news"] += cm.news
            modality_avg["graph"] += cm.graph
            total += 1
            dominant = max(
                ("price", cm.price), ("news", cm.news), ("graph", cm.graph), key=lambda kv: kv[1]
            )
            modality_counter[dominant[0]] += 1
    if total:
        for k in modality_avg:
            modality_avg[k] /= total
        dominant = modality_counter.most_common(1)[0]
        lean_line = (
            f"Modality lean: most-frequent dominant = {dominant[0]} "
            f"({dominant[1]}/{total} steps) | "
            f"avg weights = price {modality_avg['price']:.2f}, "
            f"news {modality_avg['news']:.2f}, graph {modality_avg['graph']:.2f}"
        )
    else:
        lean_line = "Modality lean: not enough data."

    parts = [header, perf_line, div_line]
    if top_div_lines:
        parts.append("Top pivotal divergences:")
        parts.extend(top_div_lines)
    parts.append(lean_line)
    return "\n".join(parts)


def _summarize_with_slm(
    metadata: ArenaRunMetadata,
    decisions_by_trajectory: dict[int, list[DecisionRecord]],
    divergences: list[DivergencePoint],
    template_narrative: str,
) -> str | None:
    """Optional Phi-3-mini-class summariser. Imports are lazy."""
    model_path = get_settings().arena.run_summarizer_model_path
    if model_path is None:
        return None
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as exc:
        logger.warning("transformers not importable for SLM summariser: {}", exc)
        return None

    cache_key = str(model_path)
    if cache_key not in _SLM_CACHE:
        _SLM_CACHE[cache_key] = {
            "tokenizer": AutoTokenizer.from_pretrained(str(model_path)),
            "model": AutoModelForCausalLM.from_pretrained(str(model_path)).eval(),
        }
    tokenizer = _SLM_CACHE[cache_key]["tokenizer"]
    model = _SLM_CACHE[cache_key]["model"]

    prompt = _build_slm_prompt(metadata, divergences, template_narrative)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    out = model.generate(
        **inputs,
        max_new_tokens=_SLM_MAX_TOKENS,
        do_sample=False,
    )
    text = tokenizer.decode(out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)
    return text.strip()


def _build_slm_prompt(
    metadata: ArenaRunMetadata,
    divergences: list[DivergencePoint],
    template_narrative: str,
) -> str:
    """Build the prompt sent to the SLM — terse, structured, JSON-flavored."""
    top = sorted(divergences, key=lambda d: -d.sharpe_delta)[:3]
    top_str = "\n".join(
        f"- {d.sim_timestamp.isoformat()} T{d.best_trajectory_id} vs T{d.worst_trajectory_id} "
        f"sharpe_delta={d.sharpe_delta:+.2f}"
        for d in top
    )
    return (
        "You are summarising one arena run for a trading-system operator.\n"
        "Statistics (template):\n"
        f"{template_narrative}\n\n"
        f"Top pivotal divergences:\n{top_str}\n\n"
        "Write 3-5 sentences. Plain English, no Markdown."
    )
