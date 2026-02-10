# backend/cognition/agent/ppo_continuous.py
"""
Proximal Policy Optimization (PPO) Agent for Continuous Action Space

Implements PPO (Schulman et al., 2017) for trading with continuous actions.
PPO is chosen for its stability, sample efficiency, and reliable convergence.

Key Features:
- Continuous action space (4D: direction, urgency, sizing, stop_distance)
- Generalized Advantage Estimation (GAE)
- Clipped surrogate objective
- Value function baseline
- Entropy regularization for exploration

References:
- Schulman et al. (2017): "Proximal Policy Optimization Algorithms"
- Andrychowicz et al. (2020): "What Matters In On-Policy Reinforcement Learning? A Large-Scale Empirical Study"

Mathematical Formulation:
The PPO objective maximizes:
L^CLIP(θ) = E_t[min(r_t(θ)Â_t, clip(r_t(θ), 1-ε, 1+ε)Â_t)]

Where:
- r_t(θ) = π_θ(a_t|s_t) / π_θ_old(a_t|s_t) (probability ratio)
- Â_t = Generalized Advantage Estimate
- ε = clip parameter (typically 0.2)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from loguru import logger

from backend.cognition.policy.networks import ActorCriticNetwork


class PPOContinuousAgent:
    """
    PPO agent for continuous action trading.

    This agent learns a stochastic policy π(a|s) that maps states to
    continuous actions using neural networks with proper regularization.
    """

    def __init__(
        self,
        state_dim: int = 224,  # Fused super-state dimension
        action_dim: int = 4,  # [direction, urgency, sizing, stop_distance]
        hidden_dims: list[int] = [256, 256, 128],
        # PPO hyperparameters
        learning_rate: float = 5e-4,
        gamma: float = 0.99,  # Discount factor
        gae_lambda: float = 0.95,  # GAE lambda
        clip_epsilon: float = 0.2,  # PPO clip parameter
        entropy_coef: float = 0.01,  # Entropy regularization
        value_loss_coef: float = 0.5,  # Value loss weight
        max_grad_norm: float = 0.5,  # Gradient clipping
        # Trainin parameters
        batch_size: int = 64,
        n_epochs: int = 10,  # Epochs per update
        target_kl: float = 0.01,  # Early stopping KL threshold
        # Action space bounds
        direction_bounds: tuple[float, float] = (-1.0, 1.0),
        urgency_bounds: tuple[float, float] = (0.0, 1.0),
        sizing_bounds: tuple[float, float] = (0.0, 1.0),
        stop_bounds: tuple[float, float] = (0.0, 1.0),
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        Initialize PPO agent.

        Args:
            state_dim: Dimension of state space (fused embeddings)
            action_dim: Dimension of action space (always 4)
            hidden_dims: Hidden layer dimensions
            learning_rate: Learning rate for optimizer
            gamma: Discount factor for rewards
            gae_lambda: Lambda parameter for GAE
            clip_epsilon: PPO clipping parameter
            entropy_coef: Entropy regularization coefficient
            value_loss_coef: Value function loss coefficient
            max_grad_norm: Max gradient norm for clipping
            batch_size: Batch size for updates
            n_epochs: Number of optimization epochs per update
            target_kl: KL divergence threshold for early stopping
            direction_bounds: Bounds for direction action
            urgency_bounds: Bounds for urgency action
            sizing_bounds: Bounds for sizing action
            stop_bounds: Bounds for stop-distance action
            device: Device for computation
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = torch.device(device)

        # PPO hyperparameters
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef
        self.max_grad_norm = max_grad_norm
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.target_kl = target_kl

        # Action bounds
        self.action_bounds = {
            "direction": direction_bounds,
            "urgency": urgency_bounds,
            "sizing": sizing_bounds,
            "stop_distance": stop_bounds,
        }

        # Create actor-critic network
        self.network = ActorCriticNetwork(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=hidden_dims,
            action_bounds=[direction_bounds, urgency_bounds, sizing_bounds, stop_bounds],
        ).to(self.device)

        # Optimizer
        self.optimizer = optim.Adam(
            self.network.parameters(),
            lr=learning_rate,
            eps=1e-5,
        )

        # Storage for trajectories
        self.states: list[torch.Tensor] = []
        self.actions: list[torch.Tensor] = []
        self.log_probs: list[torch.Tensor] = []
        self.rewards: list[float] = []
        self.values: list[torch.Tensor] = []
        self.dones: list[bool] = []

        # Metrics
        self.training_step = 0
        self.episode_count = 0

        logger.info(f"PPO Agent initialized with state_dim={state_dim}, action_dim={action_dim}")
        logger.info(f"Network parameters: {sum(p.numel() for p in self.network.parameters()):,}")

    def select_action(
        self,
        state: np.ndarray,
        deterministic: bool = False,
    ) -> tuple[np.ndarray, torch.Tensor | None, torch.Tensor | None]:
        """
        Select action given state.

        Args:
            state: Current state (fused embedding)
            deterministic: If True, use mean action (no exploration)

        Returns:
            action: Selected action (numpy array)
            log_prob: Log probability of action (for training)
            value: Value estimate (for training)
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            action_dist, value = self.network(state_tensor)

            if deterministic:
                # Use mean action (no exploration)
                action = action_dist.mean
            else:
                # Sample from distribution
                action = action_dist.sample()

            log_prob = action_dist.log_prob(action).sum(dim=-1)

        return action.cpu().numpy()[0], log_prob, value

    def store_transition(
        self,
        state: np.ndarray,
        action: np.ndarray,
        log_prob: torch.Tensor,
        reward: float,
        value: torch.Tensor,
        done: bool,
    ):
        """
        Store transition in memory.

        Args:
            state: State
            action: Action taken
            log_prob: Log probability of action
            reward: Reward recieved
            value: Value estimate
            done: Episode done flag
        """
        self.states.append(torch.FloatTensor(state).to(self.device))
        self.actions.append(torch.FloatTensor(action).to(self.device))
        self.log_probs.append(log_prob.to(self.device))
        self.rewards.append(reward)
        self.values.append(value.to(self.device))
        self.dones.append(done)

    def compute_gae(
        self,
        next_value: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Generalized Advantage Estimation (GAE).

        GAE is a variance-reduction technique that balances bias and variance
        in advantage estimates using an exponentially-weighted average.

        Formula:
        Â_t = Σ(γλ)^l δ_{t+l}
        where δ_t = r_t + γV(s_{t+1}) - V(s_t)

        Args:
            next_value: Value estimate for next state

        Returns:
            advantages: GAE advantages
            returns: Discounted returns
        """
        advantages = []
        gae = 0

        values = self.values + [next_value]

        # Compute advantages backwards
        for t in reversed(range(len(self.rewards))):
            if self.dones[t]:
                next_value = 0
            else:
                next_value = values[t + 1]

            # TD error: δ_t = r_t + γV(s_{t+1}) - V(s_t)
            delta = self.rewards[t] + self.gamma * next_value - values[t]

            # GAE: Â_t = δ_t + γλÂ_{t+1}
            gae = delta + self.gamma * self.gae_lambda * gae * (1 - self.dones[t])
            advantages.insert(0, gae)

        advantages = torch.stack(advantages)
        returns = advantages + torch.stack(self.values)

        return advantages, returns

    def update(self, next_state: np.ndarray | None = None) -> dict[str, float]:
        """
        Update policy using collected trajectories.

        This is the core PPO update using the clipped surrogate objective.

        Args:
            next_state: Next state for bootstrapping (if episode not done)

        Returns:
            metrics: Training metrics
        """
        if len(self.states) == 0:
            logger.warning("No data to update, skipping")
            return {}

        # Get next value for GAE
        if next_state is not None:
            next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                _, next_value = self.network(next_state_tensor)
        else:
            next_value = torch.tensor(0.0).to(self.device)

        # Compute advantages and returns
        advantages, returns = self.compute_gae(next_value)

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Convert to tensors
        states = torch.stack(self.states)
        actions = torch.stack(self.actions)
        old_log_probs = torch.stack(self.log_probs)

        # Training metrics
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        total_kl = 0
        update_count = 0

        # Multiple epochs of optimization
        for epoch in range(self.n_epochs):
            # Create mini-batches
            indices = np.arange(len(states))
            np.random.shuffle(indices)

            for start in range(0, len(states), self.batch_size):
                end = start + self.batch_size
                batch_indices = indices[start:end]

                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]

                # Forward pass
                action_dist, values = self.network(batch_states)
                new_log_probs = action_dist.log_prob(batch_actions).sum(dim=-1)
                entropy = action_dist.entropy().sum(dim=-1).mean()

                # Importance ratio: r(θ) = π_θ(a|s) / π_θ_old(a|s)
                ratio = torch.exp(new_log_probs - batch_old_log_probs)

                # Clipped surrogate objective
                surr1 = ratio * batch_advantages
                surr2 = (
                    torch.clamp(
                        ratio,
                        1 - self.clip_epsilon,
                        1 + self.clip_epsilon,
                    )
                    * batch_advantages
                )
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss (MSE)
                value_loss = nn.functional.mse_loss(values.squeeze(), batch_returns)

                # Total loss
                loss = policy_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy

                # Optimization step
                self.optimizer.zero_grad()
                loss.backwards()
                nn.utils.clip_grad_norm_(
                    self.network.parameters(),
                    self.max_grad_norm,
                )
                self.optimizer.step()

                # Metrics
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item()
                update_count += 1

                # Approximate KL divergence
                with torch.no_grad():
                    kl = (batch_old_log_probs - new_log_probs).mean()
                    total_kl += kl.item()

            # Early stopping based on KL divergence
            avg_kl = total_kl / update_count
            if avg_kl > self.target_kl:
                logger.debug(f"Early stopping at epoch {epoch + 1} due to KL={avg_kl:.6f}")

        self.training_step += 1

        # Clear buffers
        self.clear_memory()

        # Return metrics
        metrics = {
            "policy_loss": total_policy_loss / update_count,
            "value_loss": total_value_loss / update_count,
            "entropy": total_entropy / update_count,
            "approx_kl": total_kl / update_count,
            "epochs_completed": epoch + 1,
        }

        return metrics

    def clear_memory(self):
        """Clear trajectory buffers."""
        self.states.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.values.clear()
        self.dones.clear()

    def save(self, path: str):
        """
        Save agent to file.

        Args:
            path: Save path
        """
        torch.save(
            {
                "network_state_dict": self.network.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "training_step": self.training_step,
                "episode_count": self.episode_count,
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
        self.network.load_state_dict(checkpoint["network_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.training_step = checkpoint.get("training_step", 0)
        self.episode_count = checkpoint.get("episode_count", 0)
        logger.success(f"Agent loaded from {path}")

    def get_action_interpretation(self, action: np.ndarray) -> dict[str, any]:
        """
        Interpret raw actions vector into human-readable format.

        Args:
            action: Raw action vector [direction, urgency, sizing, stop_distance]

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
