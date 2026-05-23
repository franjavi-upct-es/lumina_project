"""The arena step loop must never await the user-supplied explanation sink.

These tests assert the decoupling contract introduced by the v3 hardening
pass:

* A slow sink does not slow down the runner.
* When the bounded queue saturates, explanations are dropped and the
  ``arena_explanations_dropped_total`` counter is incremented — the run
  itself is not blocked or aborted.
* On clean shutdown, items that *did* make it into the queue are drained
  before the run returns (modulo a hard timeout).
"""

from __future__ import annotations

import asyncio
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

from backend.cognition.agent.policy_network import PolicyNetwork
from backend.cognition.agent.ppo_agent import PPOAgent
from backend.cognition.agent.uncertainty_gate import UncertaintyGate
from backend.simulation.arena import runner as runner_module
from backend.simulation.arena.runner import ArenaRunner
from backend.simulation.arena.schemas import ArenaRunMetadata, ArenaRunStatus, StepExplanation
from backend.simulation.environments.base_env import LuminaTradingEnv
from backend.simulation.generators.synthetic_data import jump_diffusion_episode


def _factory(n_steps: int):
    def factory(seed: int) -> LuminaTradingEnv:
        rng = np.random.default_rng(seed)

        class _Gen:
            def __iter__(self):
                yield jump_diffusion_episode(n_steps, rng=rng)

        return LuminaTradingEnv(_Gen())

    return factory


def _agent() -> PPOAgent:
    return PPOAgent(
        policy=PolicyNetwork(),
        uncertainty_gate=UncertaintyGate(),
        device="cpu",
    )


@pytest.mark.asyncio
async def test_slow_sink_does_not_block_step_loop(tmp_path: Path) -> None:
    """A sink that sleeps 100ms per call must not slow the runner anywhere
    near 100ms per step. Wall time of the run is bounded by the step loop
    itself, not the sink."""
    sink_calls: list[float] = []

    async def slow_sink(_: StepExplanation) -> None:
        sink_calls.append(time.monotonic())
        await asyncio.sleep(0.1)

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
        agent=_agent(),
        env_factory=_factory(n_steps=50),
        timescale=None,
        explanation_sink=slow_sink,
    )
    t0 = time.monotonic()
    result = await runner.run()
    elapsed = time.monotonic() - t0

    # If the sink were on the critical path, 50 steps * 3 trajectories *
    # 0.1s = 15 seconds floor. Decoupled, the run is dominated by the
    # synthetic env, which is much faster.
    assert elapsed < 8.0, (
        f"runner appears to be awaiting the sink — elapsed={elapsed:.2f}s, "
        f"sink_calls={len(sink_calls)}"
    )
    assert result.status == ArenaRunStatus.COMPLETED


@pytest.mark.asyncio
async def test_overflow_drops_are_counted(tmp_path: Path, monkeypatch) -> None:
    """Shrink the queue to size 1 and use a sink that never drains in time.
    The runner must keep running and the dropped counter must climb."""
    # Squeeze the queue size to make overflow easy to trigger.
    monkeypatch.setattr(runner_module, "ARENA_EXPLANATION_QUEUE_SIZE", 1)

    block = asyncio.Event()

    async def stuck_sink(_: StepExplanation) -> None:
        # Hold the consumer task on its first item until the run is over.
        await block.wait()

    metadata = ArenaRunMetadata(
        ticker="AAPL",
        start_date=datetime(2024, 1, 1, tzinfo=UTC),
        end_date=datetime(2024, 1, 2, tzinfo=UTC),
        n_trajectories=3,
        mc_seeds=[1, 2, 3],
    )
    from backend.config.settings import get_settings

    get_settings().arena.artifact_dir = tmp_path

    # Snapshot the dropped counter for this run_id before we start.
    label = str(metadata.run_id)
    before = runner_module.ARENA_EXPLANATIONS_DROPPED.labels(run_id=label)._value.get()

    runner = ArenaRunner(
        run_metadata=metadata,
        agent=_agent(),
        env_factory=_factory(n_steps=80),
        timescale=None,
        explanation_sink=stuck_sink,
    )
    # Release the consumer at shutdown so the runner doesn't hang on drain.
    run_task = asyncio.create_task(runner.run())
    # Give the run a moment to fill the queue and start dropping.
    await asyncio.sleep(0.5)
    block.set()
    result = await run_task

    after = runner_module.ARENA_EXPLANATIONS_DROPPED.labels(run_id=label)._value.get()
    assert after - before >= 1, (
        f"expected at least one dropped explanation; counter delta={after - before}"
    )
    assert result.status == ArenaRunStatus.COMPLETED
