# scripts/run_arena.py
"""CLI launcher for the Spartan Arena.

This is the development / CI-smoke entry point. The production
deployment fires runs via ``POST /arena/run`` and a Celery worker picks
them up; this script runs a self-contained arena in-process against
synthetic data and an untrained policy, primarily so reviewers can
exercise the pipeline end-to-end without standing up the whole
infrastructure.

Example
-------
    python scripts/run_arena.py \\
        --ticker AAPL \\
        --start 2024-01-01 \\
        --end  2024-01-05 \\
        --n-trajectories 4 \\
        --playback-multiplier 1.0 \\
        --output-dir ./artifacts/arena
"""

from __future__ import annotations

import argparse
import asyncio
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from loguru import logger

from backend.cognition.agent.policy_network import PolicyNetwork
from backend.cognition.agent.ppo_agent import PPOAgent
from backend.cognition.agent.uncertainty_gate import UncertaintyGate
from backend.config.constants import NEXUS_OUTPUT_DIM
from backend.simulation.arena.runner import ArenaRunner, make_random_seeds
from backend.simulation.arena.schemas import ArenaRunMetadata
from backend.simulation.environments.base_env import LuminaTradingEnv
from backend.simulation.generators.synthetic_data import jump_diffusion_episode


def _build_env_factory(n_steps: int):
    """Closure that yields a fresh env per Monte Carlo seed."""

    def factory(seed: int) -> LuminaTradingEnv:
        rng = np.random.default_rng(seed)

        class _OneShotGenerator:
            def __iter__(self):
                yield jump_diffusion_episode(n_steps, rng=rng)

        return LuminaTradingEnv(_OneShotGenerator())

    return factory


def _build_agent(device: str = "cpu") -> PPOAgent:
    policy = PolicyNetwork(state_dim=NEXUS_OUTPUT_DIM, action_dim=4)
    gate = UncertaintyGate()
    return PPOAgent(policy=policy, uncertainty_gate=gate, device=device)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a Spartan Arena execution.")
    parser.add_argument("--ticker", required=True)
    parser.add_argument(
        "--start",
        required=True,
        type=lambda s: datetime.fromisoformat(s).replace(tzinfo=UTC),
    )
    parser.add_argument(
        "--end",
        required=True,
        type=lambda s: datetime.fromisoformat(s).replace(tzinfo=UTC),
    )
    parser.add_argument("--n-trajectories", type=int, default=8)
    parser.add_argument("--playback-multiplier", type=float, default=1.0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./artifacts/arena"),
        help="Override ARENA_ARTIFACT_DIR for this run.",
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=200,
        help="Synthetic episode length when running without real OHLCV.",
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default="cpu",
    )
    return parser.parse_args()


async def _main_async(args: argparse.Namespace) -> int:
    # Honor the CLI override by mutating the cached settings instance.
    from backend.config.settings import get_settings

    settings = get_settings()
    settings.arena.artifact_dir = args.output_dir
    args.output_dir.mkdir(parents=True, exist_ok=True)

    seeds = make_random_seeds(args.n_trajectories)
    metadata = ArenaRunMetadata(
        ticker=args.ticker,
        start_date=args.start,
        end_date=args.end,
        n_trajectories=args.n_trajectories,
        mc_seeds=seeds,
        playback_multiplier=args.playback_multiplier,
    )
    logger.info(
        "Starting arena run {} ticker={} steps={} N={}",
        metadata.run_id,
        args.ticker,
        args.n_steps,
        args.n_trajectories,
    )

    agent = _build_agent(device=args.device)
    env_factory = _build_env_factory(args.n_steps)
    runner = ArenaRunner(
        run_metadata=metadata,
        agent=agent,
        env_factory=env_factory,
        timescale=None,  # smoke-test mode: JSONL-only
    )

    result = await runner.run()
    summary_path = args.output_dir / f"{result.run_id}.summary.txt"
    summary_path.write_text(
        f"run_id={result.run_id}\nstatus={result.status.value}\n"
        f"n_divergences={len(runner.divergence_analyzer.all_divergences())}\n",
        encoding="utf-8",
    )

    # stdout for the operator. `print` is allowed in scripts (ruff override).
    print(f"run_id           : {result.run_id}")
    print(f"final status     : {result.status.value}")
    print(f"divergences seen : {len(runner.divergence_analyzer.all_divergences())}")
    print(f"avg step (ms)    : {runner.time_controller.average_duration_ms:.2f}")
    print(f"p95 step (ms)    : {runner.time_controller.p95_duration_ms:.2f}")
    print(f"artifact dir     : {args.output_dir}")
    return 0 if result.status.value == "COMPLETED" else 1


def main() -> int:
    args = _parse_args()
    try:
        return asyncio.run(_main_async(args))
    except KeyboardInterrupt:
        logger.warning("Arena CLI interrupted")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
