"""Run the article-ready daily universe sweep simulation bundle."""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from loguru import logger

from backend.config.logging import configure_logging
from backend.config.settings import get_settings
from backend.simulation.article_simulation import (
    ArticleSimulationConfig,
    DailyMarketData,
    load_market_data_from_timescale,
    run_article_pipeline,
    synthetic_market_data,
)

_COMPOSE_TIMESCALE_HOST = "timescale"
_LOCAL_TIMESCALE_HOST = "localhost"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Lumina article simulations.")
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/article_sims"))
    parser.add_argument("--checkpoint-root", type=Path, default=Path("models"))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--nexus-epochs", type=int, default=50)
    parser.add_argument("--hard-epochs", type=int, default=50)
    parser.add_argument("--hard-early-stop-patience", type=int, default=8)
    parser.add_argument("--hard-min-delta", type=float, default=1e-6)
    parser.add_argument("--policy-bc-epochs", type=int, default=80)
    parser.add_argument("--feedback-policy-epochs", type=int, default=60)
    parser.add_argument("--n-trajectories", type=int, default=8)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-eval-samples", type=int, default=0)
    parser.add_argument("--max-policy-samples", type=int, default=8192)
    parser.add_argument("--extra-test-scenarios", type=int, default=0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--seeds",
        default="",
        help="Comma/range seed sweep, for example '1,2,4-10'. Overrides --seed.",
    )
    parser.add_argument(
        "--write-sweep-checkpoints",
        action="store_true",
        help="Allow checkpoint writes during --seeds sweeps. Disabled by default for sweeps.",
    )
    parser.add_argument(
        "--no-checkpoint-overwrite",
        action="store_true",
        help="Produce the report bundle without writing models/* checkpoints.",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Small CPU-friendly smoke settings for local validation.",
    )
    parser.add_argument(
        "--data-source",
        choices=("auto", "timescale", "synthetic"),
        default="auto",
        help=(
            "Market data source. auto tries Timescale via a host-compatible URL "
            "and falls back to deterministic synthetic data when Timescale is unavailable."
        ),
    )
    parser.add_argument(
        "--synthetic-tickers",
        type=int,
        default=8,
        help="Ticker count used by --data-source synthetic or auto fallback.",
    )
    parser.add_argument(
        "--synthetic-days",
        type=int,
        default=13_500,
        help="Daily rows per synthetic ticker used by --data-source synthetic or auto fallback.",
    )
    return parser.parse_args()


def _config_from_args(
    args: argparse.Namespace, *, seed: int | None = None
) -> ArticleSimulationConfig:
    config = ArticleSimulationConfig(
        output_root=args.output_root,
        checkpoint_root=args.checkpoint_root,
        device=args.device,
        nexus_epochs=args.nexus_epochs,
        hard_epochs=args.hard_epochs,
        hard_early_stop_patience=args.hard_early_stop_patience,
        hard_min_delta=args.hard_min_delta,
        policy_bc_epochs=args.policy_bc_epochs,
        feedback_policy_epochs=args.feedback_policy_epochs,
        n_trajectories=args.n_trajectories,
        max_train_samples=args.max_train_samples,
        max_eval_samples=args.max_eval_samples,
        max_policy_samples=args.max_policy_samples,
        extra_test_scenarios=args.extra_test_scenarios,
        seed=args.seed if seed is None else seed,
        write_checkpoints=not args.no_checkpoint_overwrite,
    )
    if args.fast:
        config.nexus_epochs = 1
        config.hard_epochs = 1
        config.policy_bc_epochs = 1
        config.feedback_policy_epochs = 1
        config.n_trajectories = 3
        config.scenario_window_days = 60
        config.scenario_step_days = 12
        config.max_train_samples = 256
        config.max_eval_samples = 128
        config.max_policy_samples = 256
        config.batch_size = 64
    return config


def _parse_seed_spec(spec: str) -> list[int]:
    seeds: list[int] = []
    for part in (chunk.strip() for chunk in spec.split(",")):
        if not part:
            continue
        if "-" in part:
            start_s, end_s = part.split("-", 1)
            start = int(start_s)
            end = int(end_s)
            step = 1 if end >= start else -1
            seeds.extend(range(start, end + step, step))
        else:
            seeds.append(int(part))
    return list(dict.fromkeys(seeds))


def _summarize_run(out_dir: Path) -> dict[str, Any]:
    nexus = json.loads((out_dir / "nexus_metrics.json").read_text(encoding="utf-8"))
    arena = json.loads((out_dir / "arena_metrics.json").read_text(encoding="utf-8"))
    feedback = json.loads((out_dir / "feedback_summary.json").read_text(encoding="utf-8"))
    manifest = json.loads((out_dir / "run_manifest.json").read_text(encoding="utf-8"))
    initial = nexus["initial"]
    hard = nexus["hard_example"]
    base = {row["scenario"]: row for row in arena if row["phase"] == "baseline_eval"}
    post = {row["scenario"]: row for row in arena if row["phase"] == "post_feedback"}
    valtest = [
        post[name]["mean_sharpe"] - base[name]["mean_sharpe"]
        for name in post
        if name in base and post[name]["split"] in {"val", "test"}
    ]
    repeat_base = [
        base[name]["bad_action_repeat_rate"]
        for name in post
        if name in base and post[name]["split"] in {"val", "test"}
    ]
    repeat_post = [
        post[name]["bad_action_repeat_rate"]
        for name in post
        if name in base and post[name]["split"] in {"val", "test"}
    ]
    bad_repeat = [p - b for p, b in zip(repeat_post, repeat_base, strict=True)]
    diagnosis = feedback.get("diagnosis", {})
    return {
        "run_id": out_dir.name,
        "seed": manifest["config"]["seed"],
        "status": manifest["status"],
        "val_loss_delta_pct": _pct_delta(initial["val"]["loss"], hard["val"]["loss"]),
        "test_directional_edge": (
            hard["test"]["directional_accuracy"] - hard["test"]["naive_directional_accuracy"]
        ),
        "arena_valtest_sharpe_improved": sum(delta > 0 for delta in valtest),
        "arena_valtest_scenarios": len(valtest),
        "arena_avg_valtest_dsharpe": sum(valtest) / len(valtest) if valtest else 0.0,
        "arena_repeat_rate_baseline": sum(repeat_base) / len(repeat_base) if repeat_base else 0.0,
        "arena_repeat_rate_post": sum(repeat_post) / len(repeat_post) if repeat_post else 0.0,
        "arena_avg_bad_repeat_delta": sum(bad_repeat) / len(bad_repeat) if bad_repeat else 0.0,
        "feedback_verdict": diagnosis.get("verdict", ""),
        "counterfactual_pairs": feedback.get("counterfactual_pairs", 0),
        "bc_accuracy": feedback.get("behavioral_cloning", {}).get("accuracy", 0.0),
        "output_dir": str(out_dir),
    }


def _pct_delta(before: float, after: float) -> float:
    return ((after / before) - 1.0) * 100.0 if before else 0.0


def _write_sweep_summary(output_root: Path, rows: list[dict[str, Any]]) -> tuple[Path, Path]:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    summary_dir = output_root / "seed_sweeps"
    summary_dir.mkdir(parents=True, exist_ok=True)
    csv_path = summary_dir / f"{stamp}_summary.csv"
    json_path = summary_dir / f"{stamp}_summary.json"
    keys = sorted({key for row in rows for key in row})
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return csv_path, json_path


def _host_compatible_timescale_url(url: str) -> str:
    parts = urlsplit(url)
    if parts.hostname != _COMPOSE_TIMESCALE_HOST:
        return url
    old_host_port = parts.hostname
    new_host_port = _LOCAL_TIMESCALE_HOST
    if parts.port is not None:
        old_host_port = f"{old_host_port}:{parts.port}"
        new_host_port = f"{new_host_port}:{parts.port}"
    return url.replace(old_host_port, new_host_port, 1)


def _synthetic_data_for_run(
    args: argparse.Namespace,
    config: ArticleSimulationConfig,
) -> DailyMarketData:
    return synthetic_market_data(
        n_tickers=args.synthetic_tickers,
        n_days=args.synthetic_days,
        seed=config.seed,
        start=datetime(1989, 1, 1, tzinfo=UTC),
    )


async def _load_market_data_for_run(
    args: argparse.Namespace,
    config: ArticleSimulationConfig,
) -> tuple[DailyMarketData, dict[str, Any] | None]:
    if args.data_source == "synthetic":
        logger.info("Using deterministic synthetic market data for article run")
        return _synthetic_data_for_run(args, config), None

    settings = get_settings()
    original_url = settings.TIMESCALE_URL
    settings.TIMESCALE_URL = _host_compatible_timescale_url(original_url)
    if original_url != settings.TIMESCALE_URL:
        logger.info("Using localhost Timescale URL for standalone article run")

    try:
        market_data, inventory = await load_market_data_from_timescale()
        if not market_data.tickers:
            raise RuntimeError("Timescale returned no OHLCV rows for the article universe")
        return market_data, inventory
    except Exception as exc:
        if args.data_source == "timescale":
            raise RuntimeError(
                "Unable to load article market data from Timescale. "
                "Start the Timescale service, set TIMESCALE_URL to a host-reachable DSN, "
                "or rerun with --data-source synthetic."
            ) from exc
        logger.warning(
            "Timescale market data is unavailable for the article run ({}); "
            "falling back to deterministic synthetic data.",
            exc,
        )
        return _synthetic_data_for_run(args, config), None


async def _run_one(args: argparse.Namespace, config: ArticleSimulationConfig) -> Path:
    market_data, inventory = await _load_market_data_for_run(args, config)
    return await run_article_pipeline(config, market_data=market_data, inventory=inventory)


async def _amain() -> int:
    args = _parse_args()
    seeds = _parse_seed_spec(args.seeds) if args.seeds else []
    if seeds:
        summaries: list[dict[str, Any]] = []
        for seed in seeds:
            config = _config_from_args(args, seed=seed)
            if not args.write_sweep_checkpoints:
                config.write_checkpoints = False
            out_dir = await _run_one(args, config)
            print(f"article simulation bundle: {out_dir}")
            summaries.append(_summarize_run(out_dir))
        csv_path, json_path = _write_sweep_summary(args.output_root, summaries)
        print(f"seed sweep summary: {csv_path}")
        print(f"seed sweep summary json: {json_path}")
    else:
        config = _config_from_args(args)
        out_dir = await _run_one(args, config)
        print(f"article simulation bundle: {out_dir}")
    return 0


def main() -> int:
    configure_logging()
    return asyncio.run(_amain())


if __name__ == "__main__":
    raise SystemExit(main())
