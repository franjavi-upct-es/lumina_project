# tests/cognition/test_ppo_agent.py
"""Tests for the PPO agent's buffer + act() shape contract.

Full PPO training is too heavy to exercise in a unit test (we'd need a
real env). What we pin instead is the shape and reproducibility of the
public surface — exactly what production code consumes.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from backend.cognition.agent.policy_network import PolicyNetwork
from backend.cognition.agent.ppo_agent import PPOAgent, PPOConfig, RolloutBuffer
from backend.cognition.agent.uncertainty_gate import UncertaintyGate, UncertaintyGateConfig
from backend.config.constants import ACTION_DIM, NEXUS_OUTPUT_DIM

_STATE_DIM = NEXUS_OUTPUT_DIM + 4


@pytest.fixture
def agent() -> PPOAgent:
    torch.manual_seed(0)
    policy = PolicyNetwork(state_dim=_STATE_DIM)
    gate = UncertaintyGate(UncertaintyGateConfig(warmup_steps=0, rolling_window=3))
    return PPOAgent(policy, gate, PPOConfig(), device="cpu")


def test_act_returns_five_tuple_with_correct_shapes(agent: PPOAgent) -> None:
    """act(state) → (action, log_prob, value, uncertainty, vetoed)."""
    state = np.random.randn(_STATE_DIM).astype(np.float32)
    out = agent.act(state)
    assert len(out) == 5
    action, log_prob, value, uncertainty, vetoed = out
    assert isinstance(action, np.ndarray)
    assert action.shape == (ACTION_DIM,)
    assert isinstance(log_prob, float)
    assert isinstance(value, float)
    assert 0.0 <= float(uncertainty) <= 1.5
    assert isinstance(vetoed, bool)


def test_act_deterministic_is_reproducible(agent: PPOAgent) -> None:
    """Two consecutive deterministic acts on the same state must agree.

    ``act`` runs MC-Dropout *regardless* of deterministic mode (the flag
    only controls whether we sample or use the mean of the resulting
    samples). For true reproducibility we explicitly put the policy in
    eval() so dropout is disabled.
    """
    state = np.zeros(_STATE_DIM, dtype=np.float32)
    agent.policy.eval()
    a1, _, _, _, _ = agent.act(state, deterministic=True)
    a2, _, _, _, _ = agent.act(state, deterministic=True)
    agent.policy.train()
    np.testing.assert_allclose(a1, a2)


def test_rollout_buffer_accumulates_and_clears() -> None:
    buf = RolloutBuffer()
    assert len(buf) == 0
    buf.add(
        state=np.zeros(_STATE_DIM, dtype=np.float32),
        action=np.zeros(ACTION_DIM, dtype=np.float32),
        log_prob=-1.0,
        value=0.5,
        reward=0.1,
        done=False,
        uncertainty=0.2,
    )
    assert len(buf) == 1
    buf.clear()
    assert len(buf) == 0


def test_compute_gae_matches_simple_case(agent: PPOAgent) -> None:
    """Single-step terminal rollout: advantage = reward - value."""
    agent.buffer.clear()
    agent.buffer.add(
        state=np.zeros(_STATE_DIM, dtype=np.float32),
        action=np.zeros(ACTION_DIM, dtype=np.float32),
        log_prob=-1.0,
        value=0.5,
        reward=1.0,
        done=True,
        uncertainty=0.1,
    )
    adv, _returns = agent.compute_gae(last_value=0.0)
    assert adv.shape == (1,)
    assert pytest.approx(adv[0], rel=1e-5) == 0.5
