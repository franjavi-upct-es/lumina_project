# scripts/train_from_feedback.py
"""Retrain the policy network using the unified master BC dataset."""

from __future__ import annotations

import torch
import numpy as np
from pathlib import Path
from loguru import logger

from backend.cognition.agent.policy_network import PolicyNetwork
from backend.cognition.training.behavioral_cloning import BehavioralCloningTrainer
from backend.config.constants import NEXUS_OUTPUT_DIM, ACTION_DIM

def main():
    dataset_path = Path("artifacts/master_bc_dataset.npz")
    checkpoint_path = Path("models/policy_v3_feedback.pt")
    
    if not dataset_path.exists():
        logger.error(f"Dataset not found at {dataset_path}")
        return

    logger.info(f"Loading master dataset from {dataset_path}...")
    data = np.load(dataset_path)
    states = data["states"]
    actions = data["actions"]
    weights = data.get("weights")

    # The dataset dim is 260 (Nexus 256 + Portfolio 4)
    state_dim = states.shape[1]
    logger.info(f"Training on {states.shape[0]} samples (state_dim={state_dim})")

    policy = PolicyNetwork(state_dim=state_dim, action_dim=ACTION_DIM)
    
    # Optional: Load existing weights if you want to continue training
    # policy_path = Path("models/policy_v3_latest.pt")
    # if policy_path.exists():
    #     policy.load_state_dict(torch.load(policy_path, map_state_dict='cpu'))

    trainer = BehavioralCloningTrainer(
        states,
        actions,
        expert_weights=weights,
        device="cuda" if torch.cuda.is_available() else "cpu",
        val_fraction=0.15
    )

    metrics = trainer.fit(policy, epochs=50)
    
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(policy.state_dict(), checkpoint_path)
    logger.success(f"Retrained policy saved to {checkpoint_path}")
    logger.info(f"Final Accuracy: {metrics['accuracy']:.2%}")

if __name__ == "__main__":
    main()
