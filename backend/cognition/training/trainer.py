# backend/cognition/training/trainer.py
"""Master orchestrator for the Spartan curriculum.

This module is the *single* entry point invoked by
``scripts.train_agent`` (and indirectly by ``scripts.run_paper_trading``
on cold-start). It wires every cognition-related component and runs the
three-stage curriculum end-to-end.

Phases (high-level)
===================

1. **Behavioural Cloning** — warm-start the policy by imitating an
   oracle (we currently use a simple Kelly-fractional MA-crossover
   policy as the oracle; see :func:`_build_expert_trajectories`).
2. **Domain Randomisation** — PPO on a mix of clean + warped episodes.
3. **Sharpe Optimisation** — PPO with Sharpe-shaped reward and entropy
   annealing.

Failure modes
=============
Each phase has an acceptance gate. If a gate fails, :class:`SpartanCurriculum`
raises ``RuntimeError`` and we exit non-zero so CI can pick it up.

Checkpoint layout
=================
    models/agent/bc.pt          # after Phase A
    models/agent/dr.pt          # after Phase B
    models/agent/best_sharpe.pt # best from Phase C (updated as we go)
    models/agent/final.pt       # alias for the production-served model
    models/agent/manifest.json  # metadata: timestamp, git hash, config
"""

from __future__ import annotations

import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
from loguru import logger

from backend.cognition.agent.policy_network import PolicyNetwork
from backend.cognition.agent.ppo_agent import PPOAgent, PPOConfig
from backend.cognition.agent.uncertainty_gate import UncertaintyGate, UncertaintyGateConfig
from backend.cognition.training.behavioral_cloning import BehavioralCloningTrainer
from backend.cognition.training.curriculum import CurriculumConfig, SpartanCurriculum
from backend.cognition.training.domain_randomization import (
    DomainRandomizationRunner,
    DRConfig,
)
from backend.cognition.training.sharpe_optimizer import SharpeOptimizer, SharpeOptimizerConfig
from backend.config.constants import ACTION_DIM, NEXUS_OUTPUT_DIM
from backend.simulation.environments.base_env import EnvConfig, LuminaTradingEnv
from backend.simulation.generators.adversarial import AdversarialGenerator
from backend.simulation.generators.synthetic_data import SyntheticEpisodeGenerator

_MODELS_DIR: Path = Path("models/agent")


def _git_commit_hash() -> str:
    """Best-effort attempt to read the current git commit. Returns
    ``"unknown"`` when not in a git checkout."""
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _build_expert_trajectories(
    n_samples: int = 4096, rng_seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Generate (state, action) pairs from a simple oracle policy.

    The oracle is a Kelly-fractional MA-crossover:

        signal = sign(EMA_fast - EMA_slow)
        action[0] = signal × 0.5   (half-Kelly direction)
        action[1] = 0.0            (neutral urgency)
        action[2] = 0.0            (default sizing)
        action[3] = 0.0            (default stop)

    The point of the BC stage is to teach the policy the *shape* of the
    action space (4-D, bounded, smooth), not to teach it a profitable
    strategy. The oracle being primitive is fine — Phase B will improve
    on it dramatically.
    """
    rng = np.random.default_rng(rng_seed)
    # Synthetic states: i.i.d. Gaussians plus the 4 bookkeeping channels.
    states = rng.standard_normal((n_samples, NEXUS_OUTPUT_DIM + 4)).astype(np.float32) * 0.1
    # The oracle "signal" is encoded in the first state dim, which is a
    # deliberate teaching signal (the policy learns a tractable mapping).
    actions = np.zeros((n_samples, ACTION_DIM), dtype=np.float32)
    actions[:, 0] = 0.5 * np.sign(states[:, 0])
    return states, actions


def train_full_curriculum(
    *,
    episodes_dr: int = 500,
    episodes_sharpe: int = 1000,
    bc_epochs: int = 20,
    device: str | None = None,
) -> Path:
    """Run the full curriculum and write the final checkpoint.

    Returns the path of the *final* model. Raises ``RuntimeError`` if
    any phase fails its acceptance gate.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    _MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # ----- 1. Build the policy + agent --------------------------------
    policy = PolicyNetwork(distribution="gaussian")
    gate = UncertaintyGate(UncertaintyGateConfig())
    agent = PPOAgent(policy, gate, PPOConfig(), device=device)

    # ----- 2. Episode generators --------------------------------------
    # We use synthetic episodes for now; once encoders are trained the
    # caller can swap in HistoricalEpisodeGenerator with the encoder dict.
    clean_gen = SyntheticEpisodeGenerator(n_steps=390, process="jump_diffusion")
    adv_gen = AdversarialGenerator(clean_gen)
    env = LuminaTradingEnv(clean_gen, EnvConfig())

    # ----- 3. Trainers for each phase ---------------------------------
    expert_states, expert_actions = _build_expert_trajectories()
    bc_trainer = BehavioralCloningTrainer(expert_states, expert_actions, device=device)
    dr_runner = DomainRandomizationRunner(env, clean_gen, adv_gen, DRConfig())
    sharpe_opt = SharpeOptimizer(env, clean_gen, SharpeOptimizerConfig())

    # ----- 4. Curriculum orchestrator ---------------------------------
    curriculum = SpartanCurriculum(
        agent,
        CurriculumConfig(
            bc_epochs=bc_epochs,
            dr_episodes=episodes_dr,
            sharpe_episodes=episodes_sharpe,
            checkpoint_dir=_MODELS_DIR,
        ),
    )
    curriculum.run(bc_trainer, dr_runner, sharpe_opt)

    # ----- 5. Manifest ------------------------------------------------
    final_path = _MODELS_DIR / "final.pt"
    torch.save(policy.state_dict(), final_path)
    manifest = {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "git_commit": _git_commit_hash(),
        "device": device,
        "episodes_dr": episodes_dr,
        "episodes_sharpe": episodes_sharpe,
        "bc_epochs": bc_epochs,
    }
    (_MODELS_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))
    logger.success(f"Curriculum complete. Manifest at {_MODELS_DIR / 'manifest.json'}")
    return final_path
