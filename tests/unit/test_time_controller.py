# tests/unit/test_time_controller.py
"""Unit tests for the AdaptiveStepController."""

from __future__ import annotations

import asyncio
import time

import pytest

from backend.simulation.arena.time_controller import AdaptiveStepController


@pytest.mark.asyncio
async def test_basic_timing() -> None:
    controller = AdaptiveStepController(playback_multiplier=1.0)
    for i in range(10):
        async with controller.step(i):
            await asyncio.sleep(0.01)
    avg = controller.average_duration_ms
    assert 9.0 <= avg <= 25.0, f"expected ~10 ms, got {avg}"


@pytest.mark.asyncio
async def test_playback_multiplier() -> None:
    controller = AdaptiveStepController(playback_multiplier=2.0)
    started = time.monotonic()
    for i in range(5):
        async with controller.step(i):
            await asyncio.sleep(0.01)
    elapsed_ms = (time.monotonic() - started) * 1000.0
    # 5 steps of ~10ms each, doubled = ~100ms (give or take async overhead).
    assert 90.0 <= elapsed_ms <= 200.0, f"got {elapsed_ms}"


@pytest.mark.asyncio
async def test_hard_ceiling_warning(caplog: pytest.LogCaptureFixture) -> None:
    controller = AdaptiveStepController(playback_multiplier=1.0)
    # The hook uses loguru; intercept that into the standard logging system.
    from loguru import logger

    sink_msgs: list[str] = []
    sink_id = logger.add(lambda record: sink_msgs.append(record), level="WARNING")
    try:
        async with controller.step(0):
            await asyncio.sleep(6.0)  # exceeds the 5000 ms ceiling
    finally:
        logger.remove(sink_id)
    joined = "\n".join(sink_msgs)
    assert "exceeded hard ceiling" in joined


@pytest.mark.asyncio
async def test_p95_with_few_samples() -> None:
    controller = AdaptiveStepController(playback_multiplier=1.0)
    for i in range(3):
        async with controller.step(i):
            await asyncio.sleep(0.005)
    # Fewer than 20 samples → return max recorded duration.
    assert controller.p95_duration_ms == pytest.approx(
        max(controller._durations),
        rel=1e-6,
    )


@pytest.mark.asyncio
async def test_reset_clears_window() -> None:
    controller = AdaptiveStepController(playback_multiplier=1.0)
    for i in range(5):
        async with controller.step(i):
            await asyncio.sleep(0.005)
    assert controller.average_duration_ms > 0.0
    controller.reset()
    assert controller.average_duration_ms == 0.0
    assert controller.p95_duration_ms == 0.0
