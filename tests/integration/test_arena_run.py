# tests/integration/test_arena_run.py
"""Integration tests for the full ArenaRunner pipeline.

These tests run the runner against a synthetic episode generator and an
untrained policy. They use a tmp_path artifact dir and no Timescale
backend — the goal is to validate orchestration, not training quality.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

from backend.cognition.agent.policy_network import PolicyNetwork
from backend.cognition.agent.ppo_agent import PPOAgent
from backend.cognition.agent.uncertainty_gate import UncertaintyGate
from backend.simulation.arena.runner import ArenaRunner
from backend.simulation.arena.schemas import ArenaRunMetadata, ArenaRunStatus
from backend.simulation.environments.base_env import LuminaTradingEnv
from backend.simulation.generators.synthetic_data import jump_diffusion_episode


def _build_factory(n_steps: int):
    def factory(seed: int) -> LuminaTradingEnv:
        rng = np.random.default_rng(seed)

        class _Gen:
            def __iter__(self):
                yield jump_diffusion_episode(n_steps, rng=rng)

        return LuminaTradingEnv(_Gen())

    return factory


def _build_agent() -> PPOAgent:
    policy = PolicyNetwork()
    gate = UncertaintyGate()
    return PPOAgent(policy=policy, uncertainty_gate=gate, device="cpu")


@pytest.mark.integration
@pytest.mark.asyncio
async def test_three_trajectory_short_run(tmp_path: Path) -> None:
    metadata = ArenaRunMetadata(
        ticker="AAPL",
        start_date=datetime(2024, 1, 1, tzinfo=UTC),
        end_date=datetime(2024, 1, 2, tzinfo=UTC),
        n_trajectories=3,
        mc_seeds=[1, 2, 3],
    )
    from backend.config.settings import get_settings

    get_settings().arena.artifact_dir = tmp_path

    runner = ArenaRunner(
        run_metadata=metadata,
        agent=_build_agent(),
        env_factory=_build_factory(n_steps=200),
        timescale=None,
    )
    result = await runner.run()
    assert result.status == ArenaRunStatus.COMPLETED
    # Each trajectory should have produced at least ~40 records (we allow
    # for early termination via drawdown).
    jsonl = tmp_path / str(metadata.run_id) / "decisions.jsonl"
    n_records = len(jsonl.read_text().splitlines())
    assert n_records >= 50, f"expected >= 50 records, got {n_records}"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_arena_cancel(tmp_path: Path) -> None:
    metadata = ArenaRunMetadata(
        ticker="AAPL",
        start_date=datetime(2024, 1, 1, tzinfo=UTC),
        end_date=datetime(2024, 1, 2, tzinfo=UTC),
        n_trajectories=3,
        mc_seeds=[10, 20, 30],
    )
    from backend.config.settings import get_settings

    get_settings().arena.artifact_dir = tmp_path

    runner = ArenaRunner(
        run_metadata=metadata,
        agent=_build_agent(),
        env_factory=_build_factory(n_steps=500),
        timescale=None,
    )
    run_task = asyncio.create_task(runner.run())
    await asyncio.sleep(0.2)  # let a few steps run
    await runner.cancel()
    result = await run_task
    assert result.status in {ArenaRunStatus.CANCELLED, ArenaRunStatus.COMPLETED}


@pytest.mark.integration
@pytest.mark.asyncio
async def test_divergences_emit_when_perturbations_aggressive(tmp_path: Path) -> None:
    """An untrained policy on 3 different seeds is already noisy enough to
    produce divergences over 200 steps. We assert at least one is detected."""
    metadata = ArenaRunMetadata(
        ticker="AAPL",
        start_date=datetime(2024, 1, 1, tzinfo=UTC),
        end_date=datetime(2024, 1, 2, tzinfo=UTC),
        n_trajectories=3,
        mc_seeds=[111, 222, 333],
    )
    from backend.config.settings import get_settings

    get_settings().arena.artifact_dir = tmp_path

    runner = ArenaRunner(
        run_metadata=metadata,
        agent=_build_agent(),
        env_factory=_build_factory(n_steps=300),
        timescale=None,
    )
    await runner.run()
    divergences = runner.divergence_analyzer.all_divergences()
    assert len(divergences) >= 1
