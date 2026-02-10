# backend/cognition/policy/networks.py
"""
Policy Network Architectures for RL Agents

Implements Actor-Critic architectures for PPO and SAC:
- ActorCriticNetwork: Combined actor-critic for PPO
- SACActorNetwork: Stochastic policy for SAC
- SACCriticNetwork: Q-function approximator for SAC

All networks use proper initialization and layer normalization
for stable training.

References:
- Schulman et al. (2017): "Proximal Policy Optimization"
- Haarnoja et al. (2018): "Soft Actor-Critic"
- Andrychowicz et al. (2020): "What Matters in On-Policy RL?"
"""

import numpy as np
import torch
import torch.nn as nn

from backend.cognition.policy.distributions import (
    DiagonalGaussian,
    SquashedGaussian,
)


def weight_init(module: nn.Module):
    """
    Initialize network weights using orthogonal initialization.

    Orthogonal initialization helps with gradient flow and is
    recommended for RL (Andrychowicz et al., 2020).

    Args:
        module: Neural network module
    """
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
        nn.init.constant_(module.bias, 0.0)


class ActorCriticNetwork(nn.Module):
    """
    Actor-Critic network for PPO.

    This network has two heads:
    - Actor (policy): Maps states to action distributions
    - Critic (value): Maps states to value estimates

    Sharing the base layers between actor and critic improves
    sample efficiency through representation learning.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: list[int] = [256, 256, 128],
        activation: str = "relu",
        use_layer_norm: bool = False,
        action_bounds: list[tuple[float, float]] | None = None,
    ):
        """
        Initialize Actor-Critic network.

        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            hidden_dims: Hidden layer dimensions
            activation: Activation function ('relu', 'tanh', 'elu')
            use_layer_norm: Whether to use layer normalization
            action_bounds: Bounds for each action dimension
        """
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_bounds = action_bounds

        # Activation function
        if activation == "relu":
            self.activation = nn.ReLU
        elif activation == "tanh":
            self.activation = nn.Tanh
        elif activation == "elu":
            self.activation = nn.ELU
        else:
            raise ValueError(f"Unknown activation: {activation}")

        # Shared feature extraction layers
        layers = []
        input_dim = state_dim

        for hidden_dim in hidden_dims[:-1]:  # All but last layer
            layers.append(nn.Linear(input_dim, hidden_dim))
            if use_layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(self.activation())
            input_dim = hidden_dim

        self.shared_net = nn.Sequential(*layers)

        # Actor head (policy)
        self.actor_hidden = nn.Linear(input_dim, hidden_dims[-1])
        if use_layer_norm:
            self.actor_norm = nn.LayerNorm(hidden_dims[-1])
        self.use_layer_norm = use_layer_norm

        # Mean and log_std for Gaussian policy
        self.actor_mean = nn.Linear(hidden_dims[-1], action_dim)
        self.actor_log_std = nn.Linear(hidden_dims[-1], action_dim)

        # Critic head (value function)
        self.critic_hidden = nn.Linear(input_dim, hidden_dims[-1])
        if use_layer_norm:
            self.critic_norm = nn.LayerNorm(hidden_dims[-1])
        self.critic_value = nn.Linear(hidden_dims[-1], 1)

        # Initialize weights
        self.apply(weight_init)

        # Final layer initialization (smaller scale for stability)
        nn.init.orthogonal_(self.actor_mean.weight, gain=0.01)
        nn.init.orthogonal_(self.critic_value.weight, gain=1.0)

    def forward(self, state: torch.Tensor) -> tuple[DiagonalGaussian, torch.Tensor]:
        """
        Forward pass through network.

        Args:
            state: Input state [batch_size, state_dim]

        Returns:
            action_dist: Action distribution
            value: Value estimate [batch_size, 1]
        """
        # Shared features
        features = self.shared_net(state)

        # Actor forward
        actor_h = self.actor_hidden(features)
        if self.use_layer_norm:
            actor_h = self.actor_norm(actor_h)
        actor_h = self.activation()(actor_h)

        mean = self.actor_mean(actor_h)
        log_std = self.actor_log_std(actor_h)

        # Apply bounds to mean if specified
        if self.action_bounds is not None:
            for i, (low, high) in enumerate(self.action_bounds):
                # Squash mean to bounds using tanh
                mean[:, i] = torch.tanh(mean[:, i]) * (high - low) / 2 + (high + low) / 2

        # Create action distribution
        action_dist = DiagonalGaussian(mean, log_std)

        # Critic forward
        critic_h = self.critic_hidden(features)
        if self.use_layer_norm:
            critic_h = self.critic_norm(critic_h)
        critic_h = self.activation()(critic_h)

        value = self.critic_value(critic_h)

        return action_dist, value

    def get_value(self, state: torch.Tensor) -> torch.Tensor:
        """
        Get value estimate only (faster than full forward).

        Args:
            state: Input state

        Returns:
            value: Value estimate
        """
        features = self.shared_net(state)
        critic_h = self.critic_hidden(features)
        if self.use_layer_norm:
            critic_h = self.critic_norm(critic_h)
        critic_h = self.activation()(critic_h)
        value = self.critic_value(critic_h)
        return value


class SACActorNetwork(nn.Module):
    """
    Actor network for SAC (Squashed Gaussian policy).

    Outputs mean and log_std for a Gaussian distribution,
    which is then squashed through tanh to bound actions.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: list[int] = [256, 256],
        activation: str = "relu",
    ):
        """
        Initialize SAC actor network.

        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            hidden_dims: Hidden layer dimensions
            activation: Activation function
        """
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim

        # Activation
        if activation == "relu":
            self.activation = nn.ReLU
        elif activation == "elu":
            self.activation = nn.ELU
        else:
            raise ValueError(f"Unknown activation: {activation}")

        # Network layers
        layers = []
        input_dim = state_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(self.activation())
            input_dim = hidden_dim

        self.trunk = nn.Sequential(*layers)

        # Output layers
        self.mean_layer = nn.Linear(hidden_dims[-1], action_dim)
        self.log_std_layer = nn.Linear(hidden_dims[-1], action_dim)

        # Initialize
        self.apply(weight_init)

    def forward(self, state: torch.Tensor) -> SquashedGaussian:
        """
        Forward pass.

        Args:
            state: Input state

        Returns:
            action_dist: Squashed Gaussian distribution
        """
        features = self.trunk(state)

        mean = self.mean_layer(features)
        log_std = self.log_std_layer(features)

        # Clamp log_std for stability
        log_std = torch.clamp(log_std, -20, 2)

        return SquashedGaussian(mean, log_std)

    def sample(
        self, state: torch.Tensor, deterministic: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Sample action from policy.

        Args:
            state: Input state
            deterministic: If True, return mean action

        Returns:
            action: Sampled action
            log_prob: Log probability of action
        """
        dist = self.forward(state)

        if deterministic:
            # Use mean (no exploration)
            action = dist.mode
            log_prob = torch.zeros(action.shape[0], 1, device=action.device)
        else:
            # Sample with exploration
            action, log_prob = dist.rsample()

        return action, log_prob


class SACCriticNetwork(nn.Module):
    """
    Critic network for SAC (Q-function).

    Estimates Q(s,a): expected return from taking action a in state s.
    SAC uses two critics to reduce overestimation bias.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: list[int] = [256, 256],
        activation: str = "relu",
    ):
        """
        Initialize SAC critic network.

        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            hidden_dims: Hidden layer dimensions
            activation: Activation function
        """
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim

        # Activation
        if activation == "relu":
            self.activation = nn.ReLU
        elif activation == "elu":
            self.activation = nn.ELU
        else:
            raise ValueError(f"Unknown activation: {activation}")

        # Network layers (state and action concatenated)
        layers = []
        input_dim = state_dim + action_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(self.activation())
            input_dim = hidden_dim

        layers.append(nn.Linear(hidden_dims[-1], 1))

        self.trunk = nn.Sequential(*layers)

        # Initialize
        self.apply(weight_init)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            state: Input state [batch_size, state_dim]
            action: Input action [batch_size, action_dim]

        Returns:
            q_value: Q-value estimate [batch_size, 1]
        """
        # Concatenate state and action
        x = torch.cat([state, action], dim=-1)
        q_value = self.trunk(x)
        return q_value


class MLPNetwork(nn.Module):
    """
    Generic MLP network for flexibility.

    Can be used for custom architectures or ablation studies.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: list[int] = [256, 256],
        activation: str = "relu",
        output_activation: str | None = None,
    ):
        """
        Initialize MLP network.

        Args:
            input_dim: Input dimension
            output_dim: Output dimension
            hidden_dims: Hidden layer dimensions
            activation: Activation function
            output_activation: Output activation (optional)
        """
        super().__init__()

        # Activation functions
        activations = {
            "relu": nn.ReLU,
            "tanh": nn.Tanh,
            "elu": nn.ELU,
            "sigmoid": nn.Sigmoid,
        }

        if activation not in activations:
            raise ValueError(f"Unknown activation: {activation}")

        self.activation = activations[activation]

        # Build layers
        layers = []
        in_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(self.activation())
            in_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(in_dim, output_dim))

        if output_activation is not None:
            if output_activation not in activations:
                raise ValueError(f"Unknown output activation: {output_activation}")
            layers.append(activations[output_activation]())

        self.network = nn.Sequential(*layers)

        # Initialize
        self.apply(weight_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.network(x)
