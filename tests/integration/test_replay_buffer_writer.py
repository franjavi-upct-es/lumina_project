# tests/integration/test_replay_buffer_writer.py
"""Integration tests for the BCDatasetWriter."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import numpy as np
import pytest

from backend.simulation.arena.schemas import CounterfactualPair
from backend.simulation.feedback.replay_buffer_writer import BCDatasetWriter


def _write_state(artifact_root: Path, relative: str, value: float = 1.0) -> None:
    abs_path = artifact_root / relative
    abs_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(abs_path, np.full(224, value, dtype=np.float32))


def _pair(confidence: float) -> CounterfactualPair:
    return CounterfactualPair(
        pair_id=uuid4(),
        run_id=uuid4(),
        divergence_step_index=10,
        sim_timestamp=datetime.now(UTC),
        state_artifact_path=f"states/{confidence:.2f}.npy",
        good_action_vector=[0.5, 0.0, 0.0, 0.0],
        bad_action_vector=[-0.5, 0.0, 0.0, 0.0],
        good_outcome_sharpe=1.0,
        bad_outcome_sharpe=0.0,
        confidence_score=confidence,
    )


@pytest.mark.integration
def test_append_pairs_creates_npz(tmp_path: Path) -> None:
    artifact_root = tmp_path / "artifacts"
    pairs: list[CounterfactualPair] = []
    for c in (0.2, 0.5, 0.8):
        p = _pair(c)
        _write_state(artifact_root, p.state_artifact_path)
        pairs.append(p)

    writer = BCDatasetWriter(tmp_path / "bc")
    added = writer.append_pairs(pairs, artifact_root)
    out_path = writer.finalize()
    assert added > 0
    assert out_path.exists()
    data = np.load(out_path)
    assert data["states"].shape[1] == 224
    assert data["actions"].shape[1] == 4
    assert data["states"].shape[0] == data["actions"].shape[0]


@pytest.mark.integration
def test_high_confidence_pairs_weighted(tmp_path: Path) -> None:
    artifact_root = tmp_path / "artifacts"
    low = _pair(0.0)
    high = _pair(1.0)
    _write_state(artifact_root, low.state_artifact_path)
    _write_state(artifact_root, high.state_artifact_path)

    writer = BCDatasetWriter(tmp_path / "bc")
    n_low = writer.append_pairs([low], artifact_root)
    n_high_writer = BCDatasetWriter(tmp_path / "bc_high")
    n_high = n_high_writer.append_pairs([high], artifact_root)

    assert n_low == 1, "confidence=0.0 should yield exactly 1 sample"
    assert n_high == 1, "confidence=1.0 should yield exactly 1 sample"

    data_low = np.load(writer.output_path)
    data_high = np.load(n_high_writer.output_path)
    
    assert data_low["weights"][0] == 1.0, "confidence=0.0 should have weight 1.0"
    assert data_high["weights"][0] == 5.0, "confidence=1.0 should have weight 5.0"


@pytest.mark.integration
def test_writer_preserves_full_observation_state_dim(tmp_path: Path) -> None:
    artifact_root = tmp_path / "artifacts"
    pair = _pair(0.5)
    abs_path = artifact_root / pair.state_artifact_path
    abs_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(abs_path, np.ones(260, dtype=np.float32))

    writer = BCDatasetWriter(tmp_path / "bc_full_obs")
    writer.append_pairs([pair], artifact_root)
    data = np.load(writer.finalize())

    assert data["states"].shape[1] == 260
