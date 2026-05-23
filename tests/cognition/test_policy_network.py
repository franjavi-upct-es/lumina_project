# tests/cognition/test_policy_network.py
"""Tests for the PolicyNetwork shape contract.

The PolicyNetwork is the only object that connects the squashed-Gaussian
maths to the rest of the agent code, so its public surface — ``sample``,
``evaluate_actions``, ``value`` — is what every downstream component
(BC, PPO) depends on. Breaking it silently is the kind of bug that
crashes the whole training stack on first use.

These tests do NOT verify learning behaviour, just the API shape and
gradient flow.
"""

from __future__ import annotations

import torch

from backend.cognition.agent.policy_network import PolicyNetwork
from backend.cognition.policy.distributions import SampledAction
from backend.config.constants import ACTION_DIM, NEXUS_OUTPUT_DIM

_STATE_DIM = NEXUS_OUTPUT_DIM + 4
_BATCH = 8


def test_sample_returns_two_tuple_with_correct_shapes() -> None:
    """policy.sample(s) must return (SampledAction, value_tensor).

    This guards against the bug fixed in Phase 3: a previous version of
    sample() returned a 3-tuple and BC unpacked it accordingly.
    """
    torch.manual_seed(0)
    policy = PolicyNetwork(state_dim=_STATE_DIM)
    state = torch.randn(_BATCH, _STATE_DIM)

    out = policy.sample(state)
    assert isinstance(out, tuple) and len(out) == 2

    sampled, value = out
    assert isinstance(sampled, SampledAction)
    assert sampled.action.shape == (_BATCH, ACTION_DIM)
    assert value.shape == (_BATCH,)


def test_sample_action_in_unit_box() -> None:
    """Every action component must lie in (-1, 1)."""
    torch.manual_seed(1)
    policy = PolicyNetwork(state_dim=_STATE_DIM)
    state = torch.randn(_BATCH, _STATE_DIM)
    sampled, _value = policy.sample(state)
    assert torch.all(sampled.action > -1.0)
    assert torch.all(sampled.action < 1.0)


def test_deterministic_is_reproducible() -> None:
    """Deterministic mode must give identical actions across two calls."""
    torch.manual_seed(2)
    policy = PolicyNetwork(state_dim=_STATE_DIM)
    # Must also turn off dropout for true reproducibility.
    policy.eval()
    state = torch.randn(1, _STATE_DIM)
    a1, _ = policy.sample(state, deterministic=True)
    a2, _ = policy.sample(state, deterministic=True)
    assert torch.allclose(a1.action, a2.action)


def test_evaluate_actions_returns_three_tensors_with_correct_shapes() -> None:
    """``evaluate_actions`` is used by PPO + BC; the contract is
    (log_prob, entropy, value), each shape (B,)."""
    torch.manual_seed(3)
    policy = PolicyNetwork(state_dim=_STATE_DIM)
    state = torch.randn(_BATCH, _STATE_DIM)
    action = torch.zeros(_BATCH, ACTION_DIM)  # in-range action
    log_prob, entropy, value = policy.evaluate_actions(state, action)
    assert log_prob.shape == (_BATCH,)
    assert entropy.shape == (_BATCH,)
    assert value.shape == (_BATCH,)
    assert torch.all(torch.isfinite(log_prob))


def test_value_gradient_flows() -> None:
    """A loss on value() must produce a gradient on the policy weights."""
    torch.manual_seed(4)
    policy = PolicyNetwork(state_dim=_STATE_DIM)
    state = torch.randn(_BATCH, _STATE_DIM, requires_grad=False)
    v = policy.value(state).sum()
    v.backward()
    # At least one parameter must have a non-zero gradient.
    has_grad = any((p.grad is not None and torch.any(p.grad != 0)) for p in policy.parameters())
    assert has_grad, "No gradient flowed from value() to policy parameters"
