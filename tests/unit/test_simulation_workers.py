from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np

from backend.config.constants import NEXUS_OUTPUT_DIM
from backend.simulation import backtest_worker
from backend.simulation.arena import worker as arena_worker


class HoldAgent:
    def act(self, state: np.ndarray, deterministic: bool = False):
        assert state.shape == (NEXUS_OUTPUT_DIM + 4,)
        return np.array([0.0, 0.0, -1.0, 0.0], dtype=np.float32), 0.0, 0.0, 0.2, False


def _rows(n: int) -> list[dict]:
    start = datetime(2024, 1, 1, tzinfo=UTC)
    rows = []
    for i in range(n):
        close = 100.0 + i * 0.25
        rows.append(
            {
                "time": start + timedelta(days=i),
                "open": close - 0.1,
                "high": close + 0.5,
                "low": close - 0.5,
                "close": close,
                "volume": 1_000 + i,
            }
        )
    return rows


def test_backtest_episode_from_rows_and_run_episode() -> None:
    episode = backtest_worker._episode_from_rows("AAPL", _rows(10))

    assert episode is not None
    assert episode["prices"].shape == (10,)
    assert episode["market_states"].shape == (10, NEXUS_OUTPUT_DIM)
    assert episode["synthetic"] is False

    metrics = backtest_worker._run_episode(
        ticker="AAPL",
        episode=episode,
        agent=HoldAgent(),  # type: ignore[arg-type]
        initial_capital=100_000.0,
    )

    assert metrics.ticker == "AAPL"
    assert metrics.steps == 9
    assert metrics.synthetic is False
    assert metrics.max_drawdown >= 0.0


def test_arena_historical_episode_requires_enough_rows_and_perturbs() -> None:
    assert arena_worker._episode_from_rows("AAPL", _rows(10)) is None
    episode = arena_worker._episode_from_rows("AAPL", _rows(80))

    assert episode is not None
    assert episode["prices"].shape == (80,)
    assert episode["market_states"].shape == (80, NEXUS_OUTPUT_DIM)

    perturbed = arena_worker._perturb_episode(episode, seed=123)
    assert perturbed["prices"].shape == episode["prices"].shape
    assert perturbed["market_states"].shape == episode["market_states"].shape
    assert not np.allclose(perturbed["prices"], episode["prices"])
