# backend/simulation/arena/trajectory_logger.py
"""Persist arena decisions to TimescaleDB + JSONL artifacts.

Every :class:`DecisionRecord` is written twice: once into the
``arena_decision_records`` hypertable (batched, transactional) and once
into a per-run JSONL file under ``ARENA_ARTIFACT_DIR``. The 224-d
super-state vector is saved out-of-band as ``.npy``; the database only
keeps the path.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from datetime import datetime
from pathlib import Path
from uuid import UUID

import numpy as np
import torch
from loguru import logger

from backend.data_engine.storage.timescale import TimescaleStore
from backend.simulation.arena.schemas import DecisionRecord

_FLUSH_INTERVAL_S: float = 2.0
_FLUSH_BATCH_SIZE: int = 50


class TrajectoryLogger:
    """Buffered writer for one arena run.

    The logger owns one background task that flushes the in-memory
    queue every :data:`_FLUSH_INTERVAL_S` seconds or whenever the queue
    grows to :data:`_FLUSH_BATCH_SIZE` records. Callers must invoke
    :meth:`finalize` before the process exits to flush any pending
    writes and close file handles.
    """

    def __init__(
        self,
        run_id: UUID,
        artifact_root: Path,
        timescale: TimescaleStore | None = None,
    ) -> None:
        self.run_id = run_id
        self.artifact_root = Path(artifact_root)
        self.run_dir = self.artifact_root / str(run_id)
        self.state_dir = self.run_dir / "states"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self._jsonl_path = self.run_dir / "decisions.jsonl"
        self._jsonl_lock = asyncio.Lock()

        self._timescale = timescale
        self._pending: list[DecisionRecord] = []
        self._pending_lock = asyncio.Lock()
        self._flush_event = asyncio.Event()
        self._stop_event = asyncio.Event()
        self._flusher_task: asyncio.Task | None = None

    async def start(self) -> None:
        """Spawn the background flusher. Idempotent."""
        if self._flusher_task is None or self._flusher_task.done():
            self._flusher_task = asyncio.create_task(self._flusher_loop())

    async def log_decision(
        self,
        record: DecisionRecord,
        super_state: torch.Tensor,
    ) -> DecisionRecord:
        """Persist one decision.

        The ``state_artifact_path`` field of the input ``record`` is ignored
        and rewritten by this method to point at the ``.npy`` file we just
        wrote. The returned record is a frozen copy with the correct path.
        """
        await self.start()
        relative_path = f"{self.run_id}/states/{record.trajectory_id}/{record.step_index}.npy"
        absolute_path = self.artifact_root / relative_path
        absolute_path.parent.mkdir(parents=True, exist_ok=True)
        await asyncio.to_thread(_save_npy_sync, absolute_path, super_state.detach().cpu().numpy())

        final_record = record.model_copy(update={"state_artifact_path": relative_path})

        async with self._pending_lock:
            self._pending.append(final_record)
            if len(self._pending) >= _FLUSH_BATCH_SIZE:
                self._flush_event.set()
        return final_record

    async def update_realized_reward(self, record_id: UUID, reward: float) -> None:
        """Patch the reward column of a previously-logged decision."""
        if self._timescale is None:
            return
        async with self._timescale._conn() as conn:
            await conn.execute(
                "UPDATE arena_decision_records SET realized_reward = $1 WHERE record_id = $2",
                float(reward),
                record_id,
            )

    async def finalize(self) -> None:
        """Drain pending writes and stop the background flusher."""
        self._stop_event.set()
        self._flush_event.set()
        if self._flusher_task is not None:
            with contextlib.suppress(asyncio.CancelledError):
                await self._flusher_task
        await self._flush_once(force=True)

    # ------------------------------------------------------------------
    async def _flusher_loop(self) -> None:
        try:
            while not self._stop_event.is_set():
                with contextlib.suppress(TimeoutError):
                    await asyncio.wait_for(self._flush_event.wait(), timeout=_FLUSH_INTERVAL_S)
                self._flush_event.clear()
                await self._flush_once()
        except Exception:
            logger.exception("TrajectoryLogger flusher crashed")

    async def _flush_once(self, force: bool = False) -> None:
        async with self._pending_lock:
            if not self._pending:
                return
            batch = self._pending
            self._pending = []

        if self._timescale is not None:
            try:
                await self._db_insert_batch(batch)
            except Exception:
                logger.exception(
                    "TrajectoryLogger DB insert failed for run {} (n={})",
                    self.run_id,
                    len(batch),
                )

        try:
            await self._jsonl_append(batch)
        except Exception:
            logger.exception(
                "TrajectoryLogger JSONL append failed for run {} (n={})",
                self.run_id,
                len(batch),
            )

    async def _db_insert_batch(self, batch: list[DecisionRecord]) -> None:
        assert self._timescale is not None
        records = [
            (
                r.record_id,
                r.run_id,
                r.trajectory_id,
                r.step_index,
                r.sim_timestamp,
                r.wall_timestamp,
                r.ticker,
                json.dumps(r.ohlcv),
                r.action_kind.value,
                list(r.action_vector),
                float(r.confidence),
                float(r.uncertainty),
                None if r.realized_reward is None else float(r.realized_reward),
                r.state_artifact_path,
                json.dumps(r.attribution.model_dump(mode="json")),
                int(r.mc_seed),
            )
            for r in batch
        ]
        query = """
            INSERT INTO arena_decision_records (
                record_id, run_id, trajectory_id, step_index,
                sim_timestamp, wall_timestamp, ticker, ohlcv,
                action_kind, action_vector, confidence, uncertainty,
                realized_reward, state_artifact_path, attribution, mc_seed
            ) VALUES (
                $1, $2, $3, $4, $5, $6, $7, $8::jsonb,
                $9, $10, $11, $12, $13, $14, $15::jsonb, $16
            )
            ON CONFLICT (record_id, sim_timestamp) DO NOTHING
        """
        async with self._timescale._conn() as conn:
            await conn.executemany(query, records)

    async def _jsonl_append(self, batch: list[DecisionRecord]) -> None:
        # Single open() per flush, multiple lines written.
        async with self._jsonl_lock:
            await asyncio.to_thread(_append_jsonl_sync, self._jsonl_path, batch)


def _save_npy_sync(path: Path, array: np.ndarray) -> None:
    """Synchronous .npy writer for use inside ``asyncio.to_thread``."""
    np.save(path, array.astype(np.float32))


def _append_jsonl_sync(path: Path, batch: list[DecisionRecord]) -> None:
    """Synchronous append writer for use inside ``asyncio.to_thread``."""
    with path.open("a", encoding="utf-8") as fh:
        for record in batch:
            fh.write(json.dumps(record.model_dump(mode="json"), default=_json_fallback))
            fh.write("\n")


def _json_fallback(value: object) -> str:
    if isinstance(value, UUID):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")
