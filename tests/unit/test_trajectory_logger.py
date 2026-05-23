# tests/unit/test_trajectory_logger.py
"""Unit tests for the TrajectoryLogger.

Most of the logger's behaviour is exercised via a stubbed Timescale
store. We don't need a real TimescaleDB instance — the test reads
back the captured INSERT records.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import pytest
import torch

from backend.simulation.arena.schemas import (
    ActionKind,
    AttributionPayload,
    CrossModalWeights,
    DecisionRecord,
)
from backend.simulation.arena.trajectory_logger import TrajectoryLogger


class _FakeConn:
    def __init__(self, captured: list[tuple]) -> None:
        self._captured = captured

    async def executemany(self, _query: str, records: list[tuple]) -> None:
        self._captured.extend(records)

    async def execute(self, _query: str, *_args: Any) -> None:
        pass


class _FakeTimescale:
    def __init__(self) -> None:
        self.batches: list[list[tuple]] = []
        self.captured: list[tuple] = []

    def _conn(self):
        captured = self.captured
        batches = self.batches

        class _Ctx:
            async def __aenter__(self) -> _FakeConn:
                return _FakeConn(captured)

            async def __aexit__(self, *_exc: Any) -> None:
                # Snapshot per-batch view so tests can count flushes.
                batches.append(list(captured[len(captured) - len(captured) :]))

        return _Ctx()


def _make_record(traj_id: int, step: int, run_id: Any) -> DecisionRecord:
    return DecisionRecord(
        run_id=run_id,
        trajectory_id=traj_id,
        step_index=step,
        sim_timestamp=datetime.now(UTC),
        wall_timestamp=datetime.now(UTC),
        ticker="AAPL",
        ohlcv={"open": 1, "high": 1, "low": 1, "close": 1, "volume": 0},
        action_kind=ActionKind.BUY,
        action_vector=[0.1, 0.0, 0.0, 0.0],
        confidence=0.6,
        uncertainty=0.4,
        state_artifact_path="",
        attribution=AttributionPayload(
            cross_modal=CrossModalWeights(price=1 / 3, news=1 / 3, graph=1 / 3),
        ),
        mc_seed=42,
    )


@pytest.mark.asyncio
async def test_log_decision_persists_to_db(tmp_path: Path) -> None:
    run_id = uuid4()
    ts = _FakeTimescale()
    logger = TrajectoryLogger(run_id=run_id, artifact_root=tmp_path, timescale=ts)  # type: ignore[arg-type]
    rec = _make_record(0, 0, run_id)
    await logger.log_decision(rec, torch.zeros(224))
    await logger.finalize()
    assert len(ts.captured) == 1


@pytest.mark.asyncio
async def test_log_decision_writes_npy_file(tmp_path: Path) -> None:
    run_id = uuid4()
    logger = TrajectoryLogger(run_id=run_id, artifact_root=tmp_path)
    rec = _make_record(2, 7, run_id)
    state = torch.arange(224).float()
    final = await logger.log_decision(rec, state)
    await logger.finalize()
    npy_path = tmp_path / final.state_artifact_path
    assert npy_path.exists()
    loaded = np.load(npy_path)
    assert loaded.shape == (224,)


@pytest.mark.asyncio
async def test_batched_inserts_flush_on_size(tmp_path: Path) -> None:
    run_id = uuid4()
    ts = _FakeTimescale()
    logger = TrajectoryLogger(run_id=run_id, artifact_root=tmp_path, timescale=ts)  # type: ignore[arg-type]
    for i in range(51):
        await logger.log_decision(_make_record(0, i, run_id), torch.zeros(224))
    # Yield to the event loop so the background flusher can act.
    await asyncio.sleep(0.05)
    await logger.finalize()
    assert len(ts.captured) == 51


@pytest.mark.asyncio
async def test_batched_inserts_flush_on_timer(tmp_path: Path) -> None:
    run_id = uuid4()
    ts = _FakeTimescale()
    logger = TrajectoryLogger(run_id=run_id, artifact_root=tmp_path, timescale=ts)  # type: ignore[arg-type]
    for i in range(5):
        await logger.log_decision(_make_record(0, i, run_id), torch.zeros(224))
    await asyncio.sleep(2.5)
    await logger.finalize()
    assert len(ts.captured) == 5


@pytest.mark.asyncio
async def test_update_realized_reward(tmp_path: Path) -> None:
    run_id = uuid4()

    captured_updates: list[tuple[str, float, Any]] = []

    class _RewardConn:
        async def execute(self, query: str, reward: float, record_id: Any) -> None:
            captured_updates.append((query, reward, record_id))

        async def executemany(self, *_args: Any, **_kw: Any) -> None:
            pass

    class _RewardTimescale:
        def _conn(self):
            class _Ctx:
                async def __aenter__(self) -> _RewardConn:
                    return _RewardConn()

                async def __aexit__(self, *_exc: Any) -> None:
                    pass

            return _Ctx()

    logger = TrajectoryLogger(run_id=run_id, artifact_root=tmp_path, timescale=_RewardTimescale())  # type: ignore[arg-type]
    rec = _make_record(0, 0, run_id)
    final = await logger.log_decision(rec, torch.zeros(224))
    await logger.update_realized_reward(final.record_id, 0.42)
    await logger.finalize()
    assert any(round(reward, 4) == 0.42 for _q, reward, _rid in captured_updates)
