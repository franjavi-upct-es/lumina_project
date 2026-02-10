# backend/cognition/agent/sac_agent.py
"""
Soft Actor-Critic (SAC) Agent for Continuous Action Space

Implements SAC (Haarnoja et al., 2018) for trading with continuous actions.
SAC is chosen for its sample efficiency and automatic exploration via entropy maximization.

Key Features:
- Off-policy learning (uses replay buffer)
- Maximum entropy RL (encourages exploration)
- Twin Q-networks (reduces overestimation bias)
- Automatic temperature tuning
- Squashed Gaussian policy

References:
- Haarnoja et al. (2018): "Soft Actor-Critic: Off-Policy Maximum Entropy Deep RL"
- Haarnoja et al. (2018): "Soft Actor-Critic Algorithms and Applications"

Mathematical Formulation:
SAC maximizes the expected return plus entropy:
J(π) = E[Σ γ^t (r_t + α H(π(·|s_t)))]

Where:
- H(π(·|s)) = -log π(a|s) (policy entropy)
- α = temperature parameter (automatically tuned)
- Twin Q-networks: Q_θ1, Q_θ2 with min for stability
"""

import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from loguru import logger

from backend.cognition.policy.networks import SACActorNetwork, SACCriticNetwork


class ReplayBuffer:
    """
    Experience replay buffer for off-policy learning.

    Stores transitions and provides random sampling for decorrelation.
    """

    def __init__(self, capacity: int = 100000):
        """
        Intialize replay buffer.

        Args:
            capacity: Maximum number of transitions to store
        """
        self.buffer = deque(maxlen=capacity)

    def push(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ):
        """Add transition to buffer."""
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> tuple:
        """
        Sample batch of transitions.

        Args:
            batch_size: Number of transitions to sample

        Returns:
            Batch of transitions
        """
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones),
        )

    def __len__(self) -> int:
        return len(self.buffer)


class SACAgent:
    """
    SAC agent for continuous action trading.

    This agent uses maximum entropy RL to learn robust policies that
    balance reward maximization with exploration.
    """

    def __init__(
        self,
        state_dim: int = 224,  # Fused super-state dimension
        action_dim: int = 4,  # [direction, urgency, sizing, stop_distance]
        hidden_dims: list[int] = [256, 256],
        # SAC hyperparameters
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        alpha_lr: float = 3e-4,
        gamma: float = 0.99,  # Discount factor
        tau: float = 0.005,  # Soft update coefficient
        initial_alpha: float = 0.2,  # Initial temperature
        auto_alpha: bool = True,  # Automatic temperature
        target_entropy: float | None = None,
        # Training parameters
        batch_size: int = 256,
        buffer_size: int = 100000,
        warmup_steps: int = 1000,  # Random actions before learning
        # Action space bounds
        action_bounds: tuple[float, float] = (-1.0, 1.0),
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        Initialize SAC agent.

        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            hidden_dims: Hidden layer dimensions
            actor_lr: Actor learning rate
            critic_lr: Critic learning rate
            alpha_lr: Temperature learning rate
            gamma: Discount factor
            tau: Soft update coefficient for target networks
            initial_alpha: Initial temperature parameter
            auto_alpha: Enable automatic temperature tuning
            target_entropy: Target entropy (default: -action_dim)
            batch_size: Batch size for updates
            buffer_size: Replay buffer capacity
            warmup_steps: Steps before training starts
            action_bounds: Action space bounds
            device: Device for computation
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = torch.device(device)
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.warmup_steps = warmup_steps
        self.action_bounds = action_bounds

        # Target entropy for automatic temperature tuning
        if target_entropy is None:
            self.target_entropy = -action_dim  # Heuristic value
        else:
            self.target_entropy = target_entropy

        # Networks
        self.actor = SACActorNetwork(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=hidden_dims,
        ).to(self.device)

        # Twin Q-networks (reduces overestimation bias)
        self.critic1 = SACCriticNetwork(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=hidden_dims,
        ).to(self.device)

        self.critic2 = SACCriticNetwork(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=hidden_dims,
        ).to(self.device)

        # Target networks (for stable learning)
        self.critic1_target = SACCriticNetwork(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=hidden_dims,
        ).to(self.device)
        self.critic1_target.load_state_dict(self.critic1.state_dict())

        self.critic2_target = SACCriticNetwork(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=hidden_dims,
        ).to(self.device)
        self.critic2_target.load_state_dict(self.critic2.state_dict())

        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=critic_lr)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=critic_lr)

        # Temperature parameter (entropy regularization)
        self.auto_alpha = auto_alpha
        if self.auto_alpha:
            self.log_alpha = torch.tensor(
                np.log(initial_alpha), requires_grad=True, device=self.device
            )
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=alpha_lr)
            self.alpha = self.log_alpha.exp()
        else:
            self.alpha = torch.tensor(initial_alpha, device=self.device)

        # Replay buffer
        self.replay_buffer = ReplayBuffer(capacity=buffer_size)

        # Metrics
        self.training_step = 0
        self.total_steps = 0

        logger.info(f"SAC Agent initialized with state_dim={state_dim}, action_dim={action_dim}")
        logger.info(f"Actor parameters: {sum(p.numel() for p in self.actor.parameters()):,}")
        logger.info(f"Critic parameters: {sum(p.numel() for p in self.critic1.parameters()):,}")

    def select_action(self, state: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """
        Select action given state.

        Args:
            state: Current state
            deterministic: If True, use mean action

        Returns:
            action: Selected action
        """
        # Warmup: random actions
        if self.total_steps < self.warmup_steps:
            return np.random.uniform(
                self.action_bounds[0], self.action_bounds[1], size=self.action_dim
            )

        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            if deterministic:
                action, _ = self.actor.sample(state_tensor, deterministic=True)
            else:
                action, _ = self.actor.sample(state_tensor, deterministic=False)

        return action.cpu().numpy()[0]

    def store_transition(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ):
        """
        Store transition in replay buffer.

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Episode done flag
        """
        self.replay_buffer.push(state, action, reward, next_state, done)
        self.total_steps += 1

    def update(self) -> dict[str, float]:
        """
        Update networks using batch from replay buffer.

        Returns:
            metrics: Training metrics
        """
        if len(self.replay_buffer) < self.batch_size:
            return {}

        # Sample batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        # ============================================================================
        # Update Critics
        # ============================================================================

        with torch.no_grad():
            # Sample next actions from current policy
            next_actions, next_log_probs = self.actor.sample(next_states)

            # Compute target Q-values (min of two critics for stability)
            target_q1 = self.critic1_target(next_states, next_actions)
            target_q2 = self.critic2_target(next_states, next_actions)
            target_q = torch.min(target_q1, target_q2)

            # Add entropy term
            target_q = target_q - self.alpha * next_log_probs

            # Compute target: r + γ(1-d)Q_target(s',a')
            q_target = rewards + (1 - dones) * self.gamma * target_q

        # Current Q estimates
        q1 = self.critic1(states, actions)
        q2 = self.critic2(states, actions)

        # Critic losses (MSE)
        critic1_loss = F.mse_loss(q1, q_target)
        critic2_loss = F.mse_loss(states, actions)

        # Update critic 1
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()

        # Update critic 2
        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()

        # ============================================================================
        # Update Actor
        # ============================================================================

        # Sample actions from current policy
        actions_pred, log_probs = self.actor.samples(states)

        # Compute Q-values for sampled actions
        q1_pred = self.critic1(states, actions_pred)
        q2_pred = self.critic2(states, actions_pred)
        q_pred = torch.min(q1_pred, q2_pred)

        # Actor loss: maximize Q(s,a) - α*log(π(a|s))
        actor_loss = (self.alpha * log_probs - q_pred).mean()

        # Update actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ============================================================================
        # Update Temperature (if automatic)
        # ============================================================================

        if self.auto_alpha:
            # Temperature loss: α * (log π + target_entropy)
            alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()

            # Update temperature
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

            self.alpha = self.log_alpha.exp()

        # ============================================================================
        # Soft Update Target Networks
        # ============================================================================

        self._soft_update(self.critic1, self.critic1_target)
        self._soft_update(self.critic2, self.critic2_target)

        self.training_step += 1

        # Return metrics
        metrics = {
            "critic1_loss": critic1_loss.item(),
            "critic2_loss": critic2_loss.item(),
            "actor_loss": actor_loss.item(),
            "alpha": self.alpha.item(),
            "q_value": q_pred.mean().item(),
            "log_prob": log_probs.mean().item(),
        }

        if self.auto_alpha:
            metrics["alpha_loss"] = alpha_loss.item()

        return metrics

    def _soft_update(
        self,
        source: nn.Module,
        target: nn.Module,
    ):
        """
        Soft update target network.

        θ_target = τ * θ_source + (1 - τ) * θ_target

        Args:
            source: Source network
            target: Target network
        """
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                self.tau * source_param.data + (1 - self.tau) * target_param.data
            )

    def save(self, path: str):
        """
        Save agent to file.

        Args:
            path: Save path
        """
        torch.save(
            {
                "actor_state_dict": self.actor.state_dict(),
                "critic1_state_dict": self.critic1.state_dict(),
                "critic2_state_dict": self.critic2.state_dict(),
                "critic1_target_state_dict": self.critic1_target.state_dict(),
                "critic2_target_state_dict": self.critic2_target.state_dict(),
                "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
                "critic1_optimizer_state_dict": self.critic1_optimizer.state_dict(),
                "critic2_optimizer_state_dict": self.critic2_optimizer.state_dict(),
                "log_alpha": self.log_alpha if self.auto_alpha else None,
                "alpha_optimizer_state_dict": self.alpha_optimizer.state_dict()
                if self.auto_alpha
                else None,
                "training_step": self.training_step,
                "total_steps": self.total_steps,
            },
            path,
        )
        logger.success(f"Agent saved to {path}")

    def load(self, path: str):
        """
        Load agent from file.

        Args:
            path: Load path
        """
        checkpoint = torch.load(path, map_location=self.device)

        self.actor.load_state_dict(checkpoint["actor_state_dict"])
        self.critic1.load_state_dict(checkpoint["critic1_state_dict"])
        self.critic2.load_state_dict(checkpoint["critic2_state_dict"])
        self.critic1_target.load_state_dict(checkpoint["critic1_target_state_dict"])
        self.critic2_target.load_state_dict(checkpoint["critic2_target_state_dict"])

        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer_state_dict"])
        self.critic1_optimizer.load_state_dict(checkpoint["critic1_optimizer_state_dict"])
        self.critic2_optimizer.load_state_dict(checkpoint["critic2_optimizer_state_dict"])

        if self.auto_alpha and checkpoint["log_alpha"] is not None:
            self.log_alpha = checkpoint["log_alpha"]
            self.alpha_optimizer.load_state_dict(checkpoint["alpha_optimizer_state_dict"])
            self.alpha = self.log_alpha.exp()

        self.training_step = checkpoint.get("training_step", 0)
        self.total_steps = checkpoint.get("total_steps", 0)

        logger.success(f"Agent loaded from {path}")

    def get_action_interpretation(self, action: np.ndarray) -> dict[str, any]:
        """
        Interpret raw action vector.

        Args:
            action: Raw action vector

        Returns:
            Interpreted action
        """
        direction, urgency, sizing, stop_distance = action

        # Interpret direction
        if direction > 0.1:
            position = "LONG"
        elif direction < -0.1:
            position = "SHORT"
        else:
            position = "NEUTRAL"

        # Interpret urgency
        order_type = "MARKET" if urgency > 0.5 else "LIMIT"

        return {
            "position": position,
            "direction_strength": abs(direction),
            "order_type": order_type,
            "urgency": urgency,
            "position_size_pct": sizing * 100,
            "stop_loss_atr_multiple": stop_distance,
            "raw_action": action.tolist(),
        }
