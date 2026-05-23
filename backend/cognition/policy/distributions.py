# backend/cognition/policy/distributions.py
"""Action distribution utilities for the continuous-action PPO/SAC agent.

Two distribution families are supported, exactly as described in section 5
of Lumina_V3_Deep_Fusion_Architecture.md:

1. **Squashed Gaussian (PPO-friendly)** — the default. The policy network
   emits a mean μ and a learnable log-std σ; we sample
   ξ ~ Normal(μ, σ), then squash a = tanh(ξ) ∈ (-1, 1)^d. The change-of-
   variables correction for the tanh squash is

       log π(a | s) = log p(ξ | s) − Σ_i log(1 − a_i² + ε)

   This is the canonical SAC formulation (Haarnoja et al., 2018, Appendix C);
   PPO works fine with it as well.

2. **Beta distribution (variance-bounded alternative)** — useful when we
   want strictly bounded actions and well-behaved KL divergences. The
   network emits α, β > 1 (via softplus + 1) and a Beta(α, β) sample is
   linearly mapped from [0, 1] to [-1, 1]. Chou et al. (2017) showed this
   often outperforms tanh-squashed Gaussian on bounded-control tasks.

The choice between the two is a hyperparameter; we default to Gaussian for
parity with most PPO implementations.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.distributions import Beta, Normal

# A small epsilon used to avoid log(0) when computing the tanh-correction.
_TANH_EPS: float = 1e-6


@dataclass
class SampledAction:
    """Container returned by every distribution `.sample(...)` call.

    Attributes
    ----------
    action : torch.Tensor
        Squashed / mapped action, shape (B, action_dim), in [-1, 1].
    log_prob : torch.Tensor
        Joint log-probability of the action under the distribution,
        shape (B,). Already summed across action_dim.
    pre_squash : torch.Tensor | None
        For Gaussian: the raw sample ξ before tanh; needed for retraining
        because PPO `evaluate_actions` must reconstruct it. None for Beta.
    """

    action: torch.Tensor
    log_prob: torch.Tensor
    pre_squash: torch.Tensor | None = None


class SquashedGaussian:
    """Tanh-squashed multivariate-diagonal Normal distribution.

    Math
    ----
    Given μ ∈ R^d, σ ∈ R^d:

        ξ ~ N(μ, σ²)
        a = tanh(ξ)
        log π(a) = Σ_i [log N(ξ_i; μ_i, σ_i) − log(1 − tanh²(ξ_i) + ε)]

    The second term is the log-determinant of the Jacobian of tanh.
    """

    def __init__(self, mean: torch.Tensor, log_std: torch.Tensor):
        """
        Parameters
        ----------
        mean : (B, d)
            Mean of the underlying Gaussian.
        log_std : (B, d) or (d,)
            Log of the standard deviation. Will be `expand_as(mean)`.
        """
        self._mean = mean
        log_std = log_std.expand_as(mean)
        self._dist = Normal(mean, log_std.exp())

    def sample(self, deterministic: bool = False) -> SampledAction:
        """Draw a sample (or take the mode if `deterministic`)."""
        pre = self._mean if deterministic else self._dist.rsample()
        action = torch.tanh(pre)
        # log p(ξ) − log |dtanh/dξ|, summed across the action dims.
        log_prob = self._dist.log_prob(pre) - torch.log(1.0 - action.pow(2) + _TANH_EPS)
        return SampledAction(action=action, log_prob=log_prob.sum(dim=-1), pre_squash=pre)

    def log_prob(self, action: torch.Tensor) -> torch.Tensor:
        """Re-evaluate log π(a) given an action `a` previously sampled.

        We invert the squash by `atanh`, which can blow up at ±1, so we clip.
        """
        a = action.clamp(-1.0 + _TANH_EPS, 1.0 - _TANH_EPS)
        pre = 0.5 * torch.log((1.0 + a) / (1.0 - a))  # = atanh(a)
        log_p = self._dist.log_prob(pre) - torch.log(1.0 - a.pow(2) + _TANH_EPS)
        return log_p.sum(dim=-1)

    def entropy(self) -> torch.Tensor:
        """Entropy of the *underlying* Gaussian (closed form).

        Note: the entropy of the squashed distribution has no closed form;
        the Gaussian entropy is a tight upper bound and is what SAC uses
        for its temperature-tuning rule.
        """
        return self._dist.entropy().sum(dim=-1)


class ScaledBeta:
    """Beta(α, β) distribution mapped from [0, 1] to [-1, 1].

    α, β are constrained to be > 1 by passing them through `softplus + 1`.
    This guarantees a unimodal density and avoids the U-shaped, ill-behaved
    Beta(<1, <1) regime.
    """

    def __init__(self, alpha_raw: torch.Tensor, beta_raw: torch.Tensor):
        alpha = F.softplus(alpha_raw) + 1.0
        beta = F.softplus(beta_raw) + 1.0
        self._dist = Beta(alpha, beta)

    def sample(self, deterministic: bool = False) -> SampledAction:
        u = self._dist.mean if deterministic else self._dist.rsample()
        action = u * 2.0 - 1.0  # [0,1] → [-1,1]
        log_prob = self._dist.log_prob(u) - math.log(2.0)  # − log|dy/du|
        return SampledAction(action=action, log_prob=log_prob.sum(dim=-1), pre_squash=None)

    def log_prob(self, action: torch.Tensor) -> torch.Tensor:
        u = (action + 1.0) * 0.5
        u = u.clamp(_TANH_EPS, 1.0 - _TANH_EPS)
        return (self._dist.log_prob(u) - math.log(2.0)).sum(dim=-1)

    def entropy(self) -> torch.Tensor:
        return self._dist.entropy().sum(dim=-1)
