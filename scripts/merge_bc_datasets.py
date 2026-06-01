# scripts/merge_bc_datasets.py
"""Consolidate all BC datasets from Arena and Article runs into one master file.

This script searches for all `bc_dataset.npz` files under the artifacts 
directory and merges them into a single global replay buffer.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from loguru import logger

from backend.config.settings import get_settings


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge all BC datasets.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/master_bc_dataset.npz"),
        help="Path to the merged master dataset.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("artifacts"),
        help="Root directory to search for .npz files.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    
    all_states = []
    all_actions = []
    all_weights = []
    
    # Find all bc_dataset.npz files recursively
    dataset_files = list(args.root.rglob("bc_dataset.npz"))
    
    if not dataset_files:
        logger.warning(f"No bc_dataset.npz files found under {args.root}")
        return 0

    logger.info(f"Found {len(dataset_files)} datasets to merge.")

    for df in dataset_files:
        if df.resolve() == args.output.resolve():
            continue
            
        try:
            data = np.load(df)
            states = data["states"]
            actions = data["actions"]
            
            # Use weights if they exist, else default to 1.0
            if "weights" in data:
                weights = data["weights"]
            else:
                weights = np.ones(states.shape[0], dtype=np.float32)
                
            all_states.append(states)
            all_actions.append(actions)
            all_weights.append(weights)
            
            logger.info(f"  Loaded {states.shape[0]} samples from {df}")
        except Exception as e:
            logger.error(f"  Failed to load {df}: {e}")

    if not all_states:
        logger.warning("No samples were successfully loaded.")
        return 1

    # Resolve dimension mismatch: pad smaller states with zeros
    max_dim = max(s.shape[1] for s in all_states)
    padded_states = []
    for s in all_states:
        if s.shape[1] < max_dim:
            padding = np.zeros((s.shape[0], max_dim - s.shape[1]), dtype=s.dtype)
            padded_states.append(np.concatenate([s, padding], axis=1))
        else:
            padded_states.append(s)

    # Concatenate and save
    master_states = np.concatenate(padded_states, axis=0)
    master_actions = np.concatenate(all_actions, axis=0)
    master_weights = np.concatenate(all_weights, axis=0)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        states=master_states,
        actions=master_actions,
        weights=master_weights,
    )

    logger.success(f"Merged dataset saved to {args.output}")
    logger.info(f"Total samples: {master_states.shape[0]}")
    logger.info(f"State dim    : {master_states.shape[1]}")
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
