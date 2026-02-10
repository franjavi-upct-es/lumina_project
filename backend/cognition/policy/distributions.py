# backend/cognition/policy/distributions.py
"""
Probability Distributions for Continuous Action Spaces

Implements specialized probability distributions for trading actions:
- DiagonalGaussian: Standard Gaussian for unbounded actions
- BoundedNormal: Truncated/clipped Gaussian for bounded actions
- SquashedGaussian: Tanh-squashed Gaussian (used in SAC)

These distributions handle the complexities of continuous action spaces
including proper entropy calculation and log-probability computation.

References:
- Haarnoja et al. (2018): "Soft Actor-Critic" (Squashed Gaussian)
- Chou et al. (2017): "Improving Stochastic Policy Gradients" (Bounded actions)
"""

import numpy as np
import torch
from torch.distributions import Independent, Normal


class DiagonalGaussian:
    """
    Diagonal Gaussian distribution for continuous actions.

    Assumes independence between action dimensions (diagonal covariance).
    This is the standard choice for RL in continuous spaces.

    Mathematical Form:
    π(a|s) = N(μ(s), σ²(s)I)

    Where μ and σ are outputs from neural networks.
    """

    def __init__(
        self,
        mean: torch.Tensor,
        log_std: torch.Tensor,
        min_log_std: float = -20,
        max_log_std: float = 2,
    ):
        """
        Initialize Diagonal Gaussian.

        Args:
            mean: Mean of distribution [batch_size, action_dim]
            log_std: Log standard deviation [batch_size, action_dim]
            min_log_std: Minimum log std for numerical stability
            max_log_std: Maximum log std to prevent excessive exploration
        """
        self.mean = mean
        self.log_std = torch.clamp(log_std, min_log_std, max_log_std)
        self.std = torch.exp(self.log_std)

        # Create underlying Normal distribution
        self._distribution = Independent(Normal(self.mean, self.std), reinterpreted_batch_ndims=1)

    def sample(self) -> torch.Tensor:
        """
        Sample action from distribution.

        Returns:
            action: Sampled action
        """
        return self._distribution.sample()

    def rsample(self) -> torch.Tensor:
        """
        Reparameterized sample (allows gradients through sampling).

        Uses the reparameterization trick: a = μ + σ * ε, where ε ~ N(0,1)

        Returns:
            action: Sampled action with gradient support
        """
        return self._distribution.rsample()

    def log_prob(self, action: torch.Tensor) -> torch.Tensor:
        """
        Compute log probability of action.

        Args:
            action: Action to evaluate

        Returns:
            log_prob: Log probability
        """
        return self._distribution.log_prob(action)

    def entropy(self) -> torch.Tensor:
        """
        Compute entropy of distribution.

        For Gaussian: H = 0.5 * log(2πe * σ²)

        Returns:
            entropy: Distribution entropy
        """
        return self._distribution.entropy()

    @property
    def mode(self) -> torch.Tensor:
        """Mean of distribution (mode for Gaussian)."""
        return self.mean


class BoundedNormal:
    """
    Bounded Normal distribution using clipping.

    Useful for actions that must lie in a specific range [low, high].
    Actions are sampled from Gaussian and clipped to bounds.

    Note: This changes the distribution shape (no longer purely Gaussian),
    but is simple and effective in practice.
    """

    def __init__(
        self,
        mean: torch.Tensor,
        log_std: torch.Tensor,
        low: float = -1.0,
        high: float = 1.0,
        min_log_std: float = -20,
        max_log_std: float = 2,
    ):
        """
        Initialize Bounded Normal.

        Args:
            mean: Mean of distribution
            log_std: Log standard deviation
            low: Lower bound for actions
            high: Upper bound for actions
            min_log_std: Minimum log std
            max_log_std: Maximum log std
        """
        self.mean = mean
        self.log_std = torch.clamp(log_std, min_log_std, max_log_std)
        self.std = torch.exp(self.log_std)
        self.low = low
        self.high = high

        # Underlying Gaussian
        self._distribution = Independent(Normal(self.mean, self.std), reinterpreted_batch_ndims=1)

    def sample(self) -> torch.Tensor:
        """
        Sample and clip action.

        Returns:
            action: Clipped action in [low, high]
        """
        action = self._distribution.sample()
        return torch.clamp(action, self.low, self.high)

    def rsample(self) -> torch.Tensor:
        """
        Reparameterized sample with clipping.

        Returns:
            action: Clipped action with gradient support
        """
        action = self._distribution.rsample()
        return torch.clamp(action, self.low, self.high)

    def log_prob(self, action: torch.Tensor) -> torch.Tensor:
        """
        Compute log probability (approximation for clipped actions).

        Note: This is approximate as clipping changes the distribution.
        For exact computation, would need to integrate the truncated Gaussian.

        Args:
            action: Action to evaluate

        Returns:
            log_prob: Approximate log probability
        """
        # Clamp action to valid range
        action = torch.clamp(action, self.low, self.high)
        return self._distribution.log_prob(action)

    def entropy(self) -> torch.Tensor:
        """
        Compute entropy (Gaussian entropy, approximate).

        Returns:
            entropy: Distribution entropy
        """
        return self._distribution.entropy()

    @property
    def mode(self) -> torch.Tensor:
        """Clipped mean."""
        return torch.clamp(self.mean, self.low, self.high)


class SquashedGaussian:
    """
    Squashed Gaussian distribution using tanh transformation.

    Used in SAC for bounded continuous actions. Actions are sampled from
    Gaussian and passed through tanh to bound them in [-1, 1].

    The log-probability is adjusted using the change-of-variables formula:
    log π(a|s) = log π(u|s) - Σ log(1 - tanh²(u))

    where u ~ N(μ, σ²) and a = tanh(u)

    Reference:
    Haarnoja et al. (2018): "Soft Actor-Critic Algorithms and Applications"
    """

    def __init__(
        self,
        mean: torch.Tensor,
        log_std: torch.Tensor,
        min_log_std: float = -20,
        max_log_std: float = 2,
        epsilon: float = 1e-6,
    ):
        """
        Initialize Squashed Gaussian.

        Args:
            mean: Mean of pre-squashed distribution
            log_std: Log std of pre-squashed distribution
            min_log_std: Minimum log std
            max_log_std: Maximum log std
            epsilon: Small constant for numerical stability
        """
        self.mean = mean
        self.log_std = torch.clamp(log_std, min_log_std, max_log_std)
        self.std = torch.exp(self.log_std)
        self.epsilon = epsilon

        # Pre-squashed Gaussian
        self._distribution = Normal(self.mean, self.std)

    def sample(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Sample action and compute log probability.

        Returns:
            action: Squashed action in (-1, 1)
            log_prob: Log probability of action
        """
        # Sample from Gaussian
        u = self._distribution.sample()

        # Apply tanh squashing
        action = torch.tanh(u)

        # Compute log probability with change of variables
        log_prob = self._distribution.log_prob(u)
        log_prob -= torch.log(1 - action.pow(2) + self.epsilon)
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        return action, log_prob

    def rsample(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Reparameterized sample with gradient support.

        Returns:
            action: Squashed action
            log_prob: Log probability
        """
        # Reparameterized sample from Gaussian
        u = self._distribution.rsample()

        # Apply tanh squashing
        action = torch.tanh(u)

        # Compute log probability with change of variables
        log_prob = self._distribution.log_prob(u)
        log_prob -= torch.log(1 - action.pow(2) + self.epsilon)
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        return action, log_prob

    def log_prob(self, action: torch.Tensor) -> torch.Tensor:
        """
        Compute log probability of squashed action.

        Args:
            action: Squashed action in (-1, 1)

        Returns:
            log_prob: Log probability
        """
        # Inverse tanh (atanh) to get pre-squashed action
        # atanh(x) = 0.5 * log((1+x)/(1-x))
        action = torch.clamp(action, -1 + self.epsilon, 1 - self.epsilon)
        u = 0.5 * torch.log((1 + action) / (1 - action + self.epsilon))

        # Compute log probability
        log_prob = self._distribution.log_prob(u)
        log_prob -= torch.log(1 - action.pow(2) + self.epsilon)
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        return log_prob

    def entropy(self) -> torch.Tensor:
        """
        Compute entropy (approximate, using pre-squashed entropy).

        Note: Exact entropy of squashed Gaussian is intractable.
        This returns the entropy of the pre-squashed Gaussian as approximation.

        Returns:
            entropy: Approximate entropy
        """
        # Entropy of Gaussian
        entropy = 0.5 + 0.5 * np.log(2 * np.pi) + self.log_std
        return entropy.sum(dim=-1)

    @property
    def mode(self) -> torch.Tensor:
        """Squashed mean (deterministic action)."""
        return torch.tanh(self.mean)


def create_distribution(
    mean: torch.Tensor,
    log_std: torch.Tensor,
    distribution_type: str = "gaussian",
    action_bounds: tuple[float, float] | None = None,
):
    """
    Factory function to create action distribution.

    Args:
        mean: Mean of distribution
        log_std: Log standard deviation
        distribution_type: Type of distribution
        action_bounds: Optional bounds for actions

    Returns:
        Distribution object
    """
    if distribution_type == "gaussian":
        return DiagonalGaussian(mean, log_std)

    elif distribution_type == "bounded":
        if action_bounds is None:
            action_bounds = (-1.0, 1.0)
        return BoundedNormal(mean, log_std, low=action_bounds[0], high=action_bounds[1])

    elif distribution_type == "squashed":
        return SquashedGaussian(mean, log_std)

    else:
        raise ValueError(f"Unknown distribution type: {distribution_type}")
