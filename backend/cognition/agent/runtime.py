"""Runtime helpers for loading the deployed PPO agent."""

from __future__ import annotations

from pathlib import Path

import torch
from loguru import logger

from backend.cognition.agent.policy_network import PolicyNetwork
from backend.cognition.agent.ppo_agent import PPOAgent
from backend.cognition.agent.uncertainty_gate import UncertaintyGate
from backend.config.constants import ACTION_DIM, NEXUS_OUTPUT_DIM
from backend.config.settings import Settings

AGENT_CHECKPOINT = Path("models/agent/final.pt")
AGENT_STATE_DIM = NEXUS_OUTPUT_DIM + 4


def pick_device() -> str:
    """Return ``cuda`` only when a minimal CUDA probe succeeds."""
    if not torch.cuda.is_available():
        return "cpu"
    try:
        torch.zeros(1, device="cuda") + 1
        return "cuda"
    except RuntimeError as exc:
        logger.warning("CUDA reports available but probe failed ({}); falling back to CPU.", exc)
        return "cpu"


def load_agent(settings: Settings, device: str) -> PPOAgent:
    """Load the production policy checkpoint, or random weights when allowed."""
    policy = PolicyNetwork(state_dim=AGENT_STATE_DIM, action_dim=ACTION_DIM)
    if AGENT_CHECKPOINT.is_file():
        ckpt = torch.load(AGENT_CHECKPOINT, map_location=device, weights_only=False)
        state_dict = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
        policy.load_state_dict(state_dict)
        logger.info("Loaded agent weights from {} (device={})", AGENT_CHECKPOINT, device)
    elif settings.ALLOW_RANDOM_MODELS:
        logger.warning("Agent checkpoint missing at {}; using random weights.", AGENT_CHECKPOINT)
    else:
        raise FileNotFoundError(
            f"Agent checkpoint missing at {AGENT_CHECKPOINT}. "
            "Train the agent or set ALLOW_RANDOM_MODELS=true for synthetic smoke tests."
        )
    return PPOAgent(policy=policy, uncertainty_gate=UncertaintyGate(), device=device)
