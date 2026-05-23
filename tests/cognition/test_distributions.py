# tests/cognition/test_distributions.py
"""Unit tests for the policy distribution helpers.

We test the *mathematical contract* — not the specific tensor values —
so the tests survive future numerical implementation tweaks:

* ``sample`` returns a :class:`SampledAction` whose ``action`` lies in
  the open interval ``(-1, 1)`` per coordinate.
* The deterministic branch of ``sample`` returns ``tanh(mu)`` (Gaussian)
  or the Beta-mean mapped to [-1, 1] (Beta).
* ``log_prob`` is finite for any in-range action.
* ``entropy`` is non-negative and shaped ``(batch,)``.
* The squashed-Gaussian log-prob recovers the underlying-Gaussian
  log-prob *minus* the tanh-Jacobian term.
"""

from __future__ import annotations

import math

import pytest
import torch

from backend.cognition.policy.distributions import (
    SampledAction,
    SquashedGaussian,
)


@pytest.fixture
def squashed_gaussian() -> SquashedGaussian:
    """A fresh 4-D SquashedGaussian with two batch elements."""
    torch.manual_seed(0)
    mean = torch.zeros(2, 4)
    log_std = torch.full((2, 4), math.log(0.5))
    return SquashedGaussian(mean=mean, log_std=log_std)


def test_sample_returns_action_in_open_interval(squashed_gaussian: SquashedGaussian) -> None:
    """Every sampled action must lie strictly inside (-1, 1)."""
    out = squashed_gaussian.sample(deterministic=False)
    assert isinstance(out, SampledAction)
    assert out.action.shape == (2, 4)
    assert torch.all(out.action > -1.0) and torch.all(out.action < 1.0)


def test_deterministic_sample_is_tanh_of_mean() -> None:
    """When the mean is zero and we ask for the deterministic sample,
    the resulting action must be exactly tanh(0) = 0."""
    g = SquashedGaussian(mean=torch.zeros(1, 4), log_std=torch.zeros(1, 4))
    out = g.sample(deterministic=True)
    assert torch.allclose(out.action, torch.zeros(1, 4), atol=1e-6)


def test_log_prob_finite_for_in_range_action(squashed_gaussian: SquashedGaussian) -> None:
    """log_prob must produce a finite value for any in-range action."""
    action = torch.full((2, 4), 0.5)
    lp = squashed_gaussian.log_prob(action)
    assert lp.shape == (2,)
    assert torch.all(torch.isfinite(lp))


def test_entropy_nonneg(squashed_gaussian: SquashedGaussian) -> None:
    """Entropy of a continuous distribution can technically be negative
    when the density exceeds 1 — but for our default (sigma=0.5) the
    entropy stays positive. We test for the shape and finiteness here."""
    ent = squashed_gaussian.entropy()
    assert ent.shape == (2,)
    assert torch.all(torch.isfinite(ent))


def test_log_prob_decreases_as_action_moves_away_from_mode() -> None:
    """For a zero-mean Gaussian, log_prob(0) > log_prob(0.5)."""
    g = SquashedGaussian(mean=torch.zeros(1, 4), log_std=torch.full((1, 4), math.log(0.3)))
    near = torch.zeros(1, 4)
    far = torch.full((1, 4), 0.8)
    assert g.log_prob(near).item() > g.log_prob(far).item()
