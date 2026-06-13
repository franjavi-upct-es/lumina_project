# scripts/run_arena.py
"""CLI launcher for the Spartan Arena."""

from __future__ import annotations

import argparse
import asyncio
import os
from datetime import UTC, datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from loguru import logger

from backend.cognition.agent.policy_network import PolicyNetwork
from backend.cognition.agent.ppo_agent import PPOAgent
from backend.cognition.agent.uncertainty_gate import UncertaintyGate
from backend.config.constants import ACTION_DIM, NEXUS_OUTPUT_DIM
from backend.simulation.arena.runner import ArenaRunner, make_random_seeds
from backend.simulation.arena.schemas import ArenaRunMetadata
from backend.simulation.environments.base_env import LuminaTradingEnv
from backend.simulation.feedback.counterfactual_pairs import build_pairs
from backend.simulation.feedback.replay_buffer_writer import BCDatasetWriter
from backend.simulation.generators.synthetic_data import jump_diffusion_episode

_DOCKER_DEFAULT_MLFLOW_URI = "http://mlflow:5000"


def _build_env_factory(n_steps: int):
    """Closure that yields a fresh env per Monte Carlo seed."""

    def factory(seed: int) -> LuminaTradingEnv:
        rng = np.random.default_rng(seed)

        class _OneShotGenerator:
            def __iter__(self):
                yield jump_diffusion_episode(n_steps, rng=rng)

        return LuminaTradingEnv(_OneShotGenerator())

    return factory


def _build_agent(state_dim: int, checkpoint: Path | None = None, device: str = "cpu") -> PPOAgent:
    policy = PolicyNetwork(state_dim=state_dim, action_dim=ACTION_DIM)
    if checkpoint and checkpoint.exists():
        logger.info("Loading policy weights from {}", checkpoint)
        policy.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
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
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="Path to a policy .pt checkpoint to load.",
    )
    parser.add_argument(
        "--state-dim",
        type=int,
        default=NEXUS_OUTPUT_DIM + 4,
        help="State dimension for the policy network (256 + 4 portfolio features).",
    )
    parser.add_argument(
        "--action-boost",
        type=float,
        default=1.0,
        help="Multiplier for the agent's action vector to overcome execution thresholds.",
    )
    parser.add_argument(
        "--mlflow-tracking-uri",
        help=(
            "MLflow tracking URI. Defaults to a local SQLite DB under --output-dir; "
            "set this to http://localhost:5000 or another server URI to log remotely."
        ),
    )
    return parser.parse_args()


def _resolve_mlflow_tracking_uri(output_dir: Path, requested_uri: str | None) -> str:
    if requested_uri:
        return requested_uri.strip()

    env_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if env_uri:
        env_uri = env_uri.strip()
        if env_uri and env_uri.rstrip("/") != _DOCKER_DEFAULT_MLFLOW_URI:
            return env_uri

    return f"sqlite:///{(output_dir / 'mlflow.db').resolve()}"


class BoostedAgent:
    def __init__(self, base_agent: PPOAgent, boost: float):
        self.base_agent = base_agent
        self.boost = boost

    def act(self, state: np.ndarray, deterministic: bool = False):
        action, log_prob, value, uncertainty, vetoed = self.base_agent.act(state, deterministic)
        # Apply boost to direction and sizing (indices 0 and 2)
        boosted_action = action.copy()
        boosted_action[0] = np.clip(boosted_action[0] * self.boost, -1, 1)
        boosted_action[2] = np.clip(boosted_action[2] * self.boost, -1, 1)
        return boosted_action, log_prob, value, uncertainty, vetoed


async def _main_async(args: argparse.Namespace) -> int:
    # Honor the CLI override by mutating the cached settings instance.
    from backend.config.settings import get_settings

    settings = get_settings()
    settings.arena.artifact_dir = args.output_dir
    args.output_dir.mkdir(parents=True, exist_ok=True)
    mlflow_tracking_uri = _resolve_mlflow_tracking_uri(
        args.output_dir,
        args.mlflow_tracking_uri,
    )

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

    agent = _build_agent(state_dim=args.state_dim, checkpoint=args.checkpoint, device=args.device)
    if args.action_boost != 1.0:
        agent = BoostedAgent(agent, args.action_boost)

    env_factory = _build_env_factory(args.n_steps)
    runner = ArenaRunner(
        run_metadata=metadata,
        agent=agent,
        env_factory=env_factory,
        timescale=None,  # smoke-test mode: JSONL-only
        policy_uses_full_observation=(args.state_dim == 260),
        mlflow_tracking_uri=mlflow_tracking_uri,
    )

    result = await runner.run()

    # Generate feedback artifacts (BC dataset)
    divergences = runner.divergence_analyzer.all_divergences()
    pairs = build_pairs(result.run_id, divergences, runner._records_by_trajectory)
    if pairs:
        bc_writer = BCDatasetWriter(args.output_dir)
        bc_writer.append_pairs(pairs, args.output_dir)
        logger.info("BC dataset updated with {} pairs", len(pairs))

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

    _save_arena_plot(runner, args.output_dir, result.run_id)
    return 0 if result.status.value == "COMPLETED" else 1


def _save_arena_plot(runner: ArenaRunner, output_dir: Path, run_id: str) -> None:
    plt.figure(figsize=(12, 6))

    # Plot equity for each trajectory
    for tid, records in runner._records_by_trajectory.items():
        steps = [r.step_index for r in records]
        # Simpler: Plot the cumulative rewards as a proxy for PnL
        rewards = np.array(
            [r.realized_reward if r.realized_reward is not None else 0.0 for r in records]
        )
        cum_returns = np.cumsum(rewards / 100.0)  # Back to fractional
        plt.plot(steps, 1.0 + cum_returns, alpha=0.6, label=f"Seed {runner.metadata.mc_seeds[tid]}")

    plt.axhline(y=1.0, color="black", linestyle="--", alpha=0.3)
    plt.title(
        f"Spartan Arena: {runner.metadata.ticker} Performance across {len(runner._envs)} Seeds"
    )
    plt.xlabel("Step")
    plt.ylabel("Normalized Equity (1.0 = Initial)")
    plt.grid(True, alpha=0.3)
    if len(runner._envs) <= 8:
        plt.legend(loc="upper left", fontsize="small", ncol=2)

    plot_path = output_dir / f"arena_{run_id}_equity.png"
    plt.savefig(plot_path, dpi=300)
    logger.success(f"Arena visualization saved to {plot_path}")


def main() -> int:
    args = _parse_args()
    try:
        return asyncio.run(_main_async(args))
    except KeyboardInterrupt:
        logger.warning("Arena CLI interrupted")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
