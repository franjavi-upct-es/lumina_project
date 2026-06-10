# backend/simulation/feedback/replay_buffer_writer.py
"""Build a Behavioral-Cloning dataset from CounterfactualPairs.

The output is a single ``.npz`` file with ``states`` and ``actions``
arrays. The Lumina BC trainer (``backend/cognition/training/...``)
consumes it during the next phase-A run. High-confidence pairs are
weight-by-duplication so they receive proportionally more gradient.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from loguru import logger

from backend.simulation.arena.schemas import CounterfactualPair

_DATASET_FILENAME: str = "bc_dataset.npz"


class BCDatasetWriter:
    """Append-style writer that round-trips an .npz between calls.

    The cumulative size of an arena's BC dataset is small (< 1 MB even
    for thousands of pairs), so we re-write the file on every append.
    Simpler than a chunked store and still well within memory.
    """

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.output_path = self.output_dir / _DATASET_FILENAME

    def append_pairs(self, pairs: list[CounterfactualPair], artifact_root: Path) -> int:
        """Load super-states for each pair and append to the on-disk dataset.

        Returns the number of pairs added. Each sample is stored once with
        an associated weight in [1.0, 5.0] derived from confidence_score.
        """
        if not pairs:
            return 0

        new_states: list[np.ndarray] = []
        new_actions: list[np.ndarray] = []
        new_weights: list[float] = []
        added_count = 0
        for pair in pairs:
            state_path = Path(artifact_root) / pair.state_artifact_path
            if not state_path.exists():
                logger.warning("BC: super-state {} not found for pair {}", state_path, pair.pair_id)
                continue
            try:
                state = np.load(state_path).astype(np.float32).reshape(-1)
            except Exception as exc:
                logger.warning("BC: failed to load state {} ({})", state_path, exc)
                continue
            action = np.asarray(pair.good_action_vector, dtype=np.float32).reshape(-1)

            # Weight scaling: confidence_score is in [0, 1].
            # Final weight is in [1.0, 5.0].
            weight = 1.0 + 4.0 * float(pair.confidence_score)

            new_states.append(state)
            new_actions.append(action)
            new_weights.append(weight)
            added_count += 1

        if not new_states:
            return 0

        states_arr = np.stack(new_states)
        actions_arr = np.stack(new_actions)
        weights_arr = np.array(new_weights, dtype=np.float32)

        if self.output_path.exists():
            existing = np.load(self.output_path)
            e_states = existing["states"]

            # Handle dimension mismatch by padding with zeros
            if e_states.shape[1] != states_arr.shape[1]:
                max_dim = max(e_states.shape[1], states_arr.shape[1])
                logger.info(
                    "BC: dimension mismatch ({} vs {}). Padding to {}.",
                    e_states.shape[1],
                    states_arr.shape[1],
                    max_dim,
                )

                if e_states.shape[1] < max_dim:
                    pad = np.zeros(
                        (e_states.shape[0], max_dim - e_states.shape[1]), dtype=e_states.dtype
                    )
                    e_states = np.concatenate([e_states, pad], axis=1)

                if states_arr.shape[1] < max_dim:
                    pad = np.zeros(
                        (states_arr.shape[0], max_dim - states_arr.shape[1]), dtype=states_arr.dtype
                    )
                    states_arr = np.concatenate([states_arr, pad], axis=1)

            states_arr = np.concatenate([e_states, states_arr], axis=0)
            actions_arr = np.concatenate([existing["actions"], actions_arr], axis=0)
            # Support backward compatibility if weights weren't there before
            if "weights" in existing:
                weights_arr = np.concatenate([existing["weights"], weights_arr], axis=0)
            else:
                old_weights = np.ones(existing["states"].shape[0], dtype=np.float32)
                weights_arr = np.concatenate([old_weights, weights_arr], axis=0)

        np.savez_compressed(
            self.output_path,
            states=states_arr,
            actions=actions_arr,
            weights=weights_arr,
        )
        return added_count

    def finalize(self) -> Path:
        return self.output_path
