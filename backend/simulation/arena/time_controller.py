# backend/simulation/arena/time_controller.py
"""Adaptive step pacing for arena trajectories.

The controller measures the wall-clock duration of the full Chimera
inference pipeline (TFT + LLM + GAT + Nexus + agent + env.step) and,
when the user is watching the run live, optionally slows playback by
inserting an extra sleep proportional to the actual duration. There is
no fixed step delay — the underlying pipeline already takes a variable
amount of time, and pretending otherwise produces misleading visuals.
"""

from __future__ import annotations

import asyncio
import contextlib
import statistics
import time
from collections import deque
from collections.abc import AsyncIterator
from dataclasses import dataclass

from loguru import logger

from backend.config.constants import (
    ARENA_STEP_HARD_CEILING_MS,
    ARENA_TIMING_WINDOW_SIZE,
)


@dataclass(frozen=True)
class StepTiming:
    """Recorded duration of a single arena step."""

    step_id: int
    started_at: float
    completed_at: float
    duration_ms: float


class AdaptiveStepController:
    """Async-context-manager based pacing utility.

    Usage
    -----
        controller = AdaptiveStepController(playback_multiplier=1.0)
        async with controller.step(step_id=42) as timing:
            await chimera_forward_pass()
        # `timing.duration_ms` is now populated.

    The context manager yields a *mutable* timing dict-like view; on
    exit, the controller fills in the timing fields, records the
    duration in the rolling window, applies the playback multiplier
    (sleep extra time), and warns if the duration exceeded the hard
    ceiling. None of those post-actions can interrupt the body.
    """

    def __init__(self, playback_multiplier: float = 1.0) -> None:
        if playback_multiplier < 1.0:
            raise ValueError(f"playback_multiplier must be >= 1.0, got {playback_multiplier}")
        self.playback_multiplier = float(playback_multiplier)
        self._durations: deque[float] = deque(maxlen=ARENA_TIMING_WINDOW_SIZE)
        self._last_timing: StepTiming | None = None

    @contextlib.asynccontextmanager
    async def step(self, step_id: int) -> AsyncIterator[StepTiming]:
        """Enter, run the body, exit; on exit emit a :class:`StepTiming`.

        The yielded object is a placeholder with zero durations; the real
        values are written into ``self._last_timing`` after the body
        completes. Consumers that need the final values should read
        ``controller.last_timing`` after the ``async with`` block.
        """
        started = time.monotonic()
        # We yield a placeholder so the caller can reference `timing.step_id`
        # inside the block (the durations are zero until the block exits).
        placeholder = StepTiming(
            step_id=step_id, started_at=started, completed_at=started, duration_ms=0.0
        )
        try:
            yield placeholder
        finally:
            completed = time.monotonic()
            duration_ms = (completed - started) * 1000.0
            final = StepTiming(
                step_id=step_id,
                started_at=started,
                completed_at=completed,
                duration_ms=duration_ms,
            )
            self._durations.append(duration_ms)
            self._last_timing = final

            if duration_ms > ARENA_STEP_HARD_CEILING_MS:
                logger.warning(
                    "Arena step {} exceeded hard ceiling: {:.1f} ms > {:.1f} ms",
                    step_id,
                    duration_ms,
                    ARENA_STEP_HARD_CEILING_MS,
                )

            if self.playback_multiplier > 1.0:
                extra_ms = (self.playback_multiplier - 1.0) * duration_ms
                if extra_ms > 0.0:
                    await asyncio.sleep(extra_ms / 1000.0)

    @property
    def last_timing(self) -> StepTiming | None:
        return self._last_timing

    @property
    def average_duration_ms(self) -> float:
        if not self._durations:
            return 0.0
        return sum(self._durations) / len(self._durations)

    @property
    def p95_duration_ms(self) -> float:
        n = len(self._durations)
        if n == 0:
            return 0.0
        if n < 20:
            return max(self._durations)
        # quantiles(n=20)[18] = boundary between the 19th and 20th twentile = p95.
        return statistics.quantiles(self._durations, n=20)[18]

    def reset(self) -> None:
        self._durations.clear()
        self._last_timing = None
