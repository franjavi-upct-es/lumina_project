# backend/cognition/agent/ppo_agent.py
"""Proximal Policy Optimization agent for Lumina V3.

PPO recap
---------
PPO (Schulman et al., 2017) optimises the policy by maximising the
clipped surrogate objective

    L^{CLIP}(θ) = E_t [ min( r_t(θ) Â_t,  clip(r_t(θ), 1-ε, 1+ε) Â_t ) ]

where r_t(θ) = π_θ(a_t | s_t) / π_{θ_old}(a_t | s_t) is the importance ratio
and Â_t is the Generalised Advantage Estimate (GAE; Schulman et al., 2016)

    δ_t   = r_t + γ V(s_{t+1}) (1 - d_t) - V(s_t)
    Â_t   = δ_t + (γ λ) δ_{t+1} (1 - d_{t+1}) + (γ λ)² δ_{t+2} ...

Total loss combines policy, value, and entropy bonus:

    L = L^{CLIP} − c_v · L^{VF} + c_H · H[π]

where L^{VF} is the *clipped* value loss (we clip the change in V_φ to
within ε to mirror the policy clip; this matches the SB3 implementation).

Notes specific to Lumina V3
---------------------------
* Action space is 4-dimensional and squashed-Gaussian.
* The Uncertainty Gate sits *outside* this class; the agent always queries
  it via the helper ``act()`` and replaces the policy action with the gate's
  defensive action when needed. The gate's veto is **NOT** logged into the
  rollout buffer — we want the policy to keep learning what *it* would have
  done, not what the gate forced.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from loguru import logger

from backend.cognition.agent.policy_network import PolicyNetwork
from backend.cognition.agent.uncertainty_gate import UncertaintyGate
from backend.config.constants import MC_DROPOUT_SAMPLES


# ----------------------------------------------------------------------
# Hyper-parameter container
# ----------------------------------------------------------------------
@dataclass
class PPOConfig:
    """All knobs of the PPO update step.

    Defaults are the standard values from "PPO Implementation Details"
    (Engstrom et al., 2020) and proved stable in our internal experiments.
    """

    gamma: float = 0.99
    """Discount factor. 0.99 → effective horizon ≈ 100 steps. Stay below
    1.0 to avoid divergence on very long episodes."""
    gae_lambda: float = 0.95
    """GAE λ — bias/variance trade-off. λ→1 = high-variance MC return,
    λ→0 = high-bias TD(0). 0.95 is the field-standard sweet spot."""
    clip_range: float = 0.2
    """ε in the policy clip. Smaller ε = smaller per-update step = more
    stable but slower learning."""
    clip_range_vf: float = 0.2
    """ε for the value-function clip; usually equal to clip_range."""
    value_coef: float = 0.5
    """c_v — weight of the value loss in the total objective."""
    entropy_coef: float = 0.01
    """c_H — exploration bonus. Decay this towards 0 in late training."""
    max_grad_norm: float = 0.5
    """Global gradient clip (L2 norm). PPO is sensitive to this."""
    epochs_per_update: int = 10
    """Number of times to iterate over the rollout buffer."""
    batch_size: int = 64
    lr: float = 3e-4
    """Adam learning rate. PPO is robust over a wide range, but 3e-4
    is the standard."""
    target_kl: float = 0.2
    """If approx. KL divergence between π_θ and π_{θ_old} exceeds 1.5x
    this value, we early-stop the update. Prevents catastrophic drift."""


# ----------------------------------------------------------------------
# Rollout buffer
# ----------------------------------------------------------------------
@dataclass
class RolloutBuffer:
    """Append-only storage for one rollout of trajectories.

    Memory layout: list-of-lists, converted to NumPy arrays only at
    update time. This is fine for our buffer sizes (≤ 4096) and is
    much simpler than circular ring buffers.
    """

    states: list = field(default_factory=list)
    actions: list = field(default_factory=list)
    log_probs: list = field(default_factory=list)
    values: list = field(default_factory=list)
    rewards: list = field(default_factory=list)
    dones: list = field(default_factory=list)
    uncertainties: list = field(default_factory=list)

    def add(self, state, action, log_prob, value, reward, done, uncertainty) -> None:
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.rewards.append(reward)
        self.dones.append(done)
        self.uncertainties.append(uncertainty)

    def clear(self) -> None:
        for name in (
            "states",
            "actions",
            "log_probs",
            "values",
            "rewards",
            "dones",
            "uncertainties",
        ):
            getattr(self, name).clear()

    def __len__(self) -> int:
        return len(self.states)


# ----------------------------------------------------------------------
# Agent
# ----------------------------------------------------------------------
class PPOAgent:
    """High-level PPO orchestrator.

    Public API
    ----------
    act(state)        → (action, log_prob, value, uncertainty, vetoed)
    record(...)       → append a transition to the rollout buffer
    update(last_value)→ run K epochs of PPO updates, return metrics dict
    """

    def __init__(
        self,
        policy: PolicyNetwork,
        uncertainty_gate: UncertaintyGate,
        config: PPOConfig | None = None,
        device: str = "cuda",
        mc_samples: int = MC_DROPOUT_SAMPLES,
    ):
        self.policy = policy.to(device)
        self.gate = uncertainty_gate
        self.config = config or PPOConfig()
        self.device = device
        self.mc_samples = mc_samples
        self.optimizer = torch.optim.AdamW(
            policy.parameters(),
            lr=self.config.lr,
            eps=1e-5,
        )
        self.buffer = RolloutBuffer()

    # ------------------------------------------------------------------
    @torch.no_grad()
    def act(
        self,
        state: np.ndarray,
        deterministic: bool = False,
    ) -> tuple[np.ndarray, float, float, float, bool]:
        """Decide on an action for the given state, with epistemic safeguards.

        Steps
        -----
        1. Convert state to tensor.
        2. Run N=10 Monte-Carlo Dropout forward passes (policy in train mode).
        3. Aggregate the N samples → uncertainty scalar.
        4. Ask the Uncertainty Gate whether to veto.
        5. If vetoed, return the defensive action; otherwise the *mean*
           of the N samples (or a deterministic action if requested).
        """
        s = torch.from_numpy(state).float().unsqueeze(0).to(self.device)

        # --- MC-Dropout sampling ----------------------------------------
        was_training = self.policy.training
        self.policy.train()  # enables dropout
        action_samples = np.zeros((self.mc_samples, self.policy.actor.action_dim), dtype=np.float32)
        log_probs = np.zeros(self.mc_samples, dtype=np.float32)
        values = np.zeros(self.mc_samples, dtype=np.float32)
        for i in range(self.mc_samples):
            sampled, value = self.policy.sample(s, deterministic=False)
            action_samples[i] = sampled.action.cpu().numpy().squeeze(0)
            log_probs[i] = sampled.log_prob.item()
            values[i] = value.item()
        self.policy.train(was_training)

        # --- Aggregate --------------------------------------------------
        uncertainty = UncertaintyGate.aggregate_action_samples(action_samples)
        mean_action = action_samples.mean(axis=0)
        mean_log_prob = float(log_probs.mean())
        mean_value = float(values.mean())

        # --- Gate decision ---------------------------------------------
        if self.gate.should_veto(uncertainty):
            return (
                self.gate.defensive_action(self.policy.actor.action_dim),
                mean_log_prob,
                mean_value,
                uncertainty,
                True,
            )

        if deterministic:
            # Replace the noisy mean with the policy's mode.
            self.policy.eval()
            with torch.no_grad():
                sampled, value = self.policy.sample(s, deterministic=True)
            self.policy.train(was_training)
            return (
                sampled.action.cpu().numpy().squeeze(0),
                sampled.log_prob.item(),
                value.item(),
                uncertainty,
                False,
            )

        return mean_action, mean_log_prob, mean_value, uncertainty, False

    # ------------------------------------------------------------------
    @torch.no_grad()
    def act_batch(
        self,
        states: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Batched ``act`` for N independent states.

        Folds the per-state ``mc_samples`` loop into a single forward pass
        of shape (N * M, state_dim) — useful when the environment exposes
        several sub-agents (e.g. one per asset) that must be queried at the
        same timestep. The gate is still consulted *per state* because it
        is a stateful object (see [[uncertainty_gate]]).
        """
        s = torch.from_numpy(states).float().to(self.device)  # (N, D)
        n_states, state_dim = s.shape
        m = self.mc_samples
        action_dim = self.policy.actor.action_dim

        was_training = self.policy.training
        self.policy.train()  # enables dropout

        # Replicate each state M times: (M, N, D) → (M*N, D)
        s_rep = s.unsqueeze(0).expand(m, n_states, state_dim).reshape(m * n_states, state_dim)
        sampled, value = self.policy.sample(s_rep, deterministic=False)

        self.policy.train(was_training)

        # (M, N, A) / (M, N) / (M, N)
        actions_mc = sampled.action.view(m, n_states, action_dim)
        log_probs_mc = sampled.log_prob.view(m, n_states)
        values_mc = value.view(m, n_states)

        # Per-state epistemic uncertainty = mean over action dims of std across MC samples
        if m >= 2:
            uncertainties_t = actions_mc.std(dim=0, unbiased=True).mean(dim=-1)
        else:
            uncertainties_t = torch.zeros(n_states, device=self.device)

        mean_actions = actions_mc.mean(dim=0).cpu().numpy()
        mean_log_probs = log_probs_mc.mean(dim=0).cpu().numpy()
        mean_values = values_mc.mean(dim=0).cpu().numpy()
        uncertainties = uncertainties_t.cpu().numpy()

        vetoed = np.zeros(n_states, dtype=bool)
        for i in range(n_states):
            if self.gate.should_veto(float(uncertainties[i])):
                mean_actions[i] = self.gate.defensive_action(action_dim)
                vetoed[i] = True

        return mean_actions, mean_log_probs, mean_values, uncertainties, vetoed

    # ------------------------------------------------------------------
    def record(
        self,
        state: np.ndarray,
        action: np.ndarray,
        log_prob: float,
        value: float,
        reward: float,
        done: bool,
        uncertainty: float,
    ) -> None:
        """Append one transition to the rollout buffer."""
        self.buffer.add(state, action, log_prob, value, reward, done, uncertainty)

    # ------------------------------------------------------------------
    def compute_gae(self, last_value: float) -> tuple[np.ndarray, np.ndarray]:
        """Vectorised Generalised Advantage Estimate.

        ``last_value`` is V(s_T+1) — used to bootstrap the final advantage
        when the episode did not terminate.
        """
        rewards = np.asarray(self.buffer.rewards, dtype=np.float32)
        values = np.asarray([*self.buffer.values, last_value], dtype=np.float32)
        dones = np.asarray(self.buffer.dones, dtype=np.float32)

        advantages = np.zeros_like(rewards)
        gae = 0.0
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.config.gamma * values[t + 1] * (1.0 - dones[t]) - values[t]
            gae = delta + self.config.gamma * self.config.gae_lambda * (1.0 - dones[t]) * gae
            advantages[t] = gae
        returns = advantages + values[:-1]
        return advantages, returns

    # ------------------------------------------------------------------
    def update(self, last_value: float = 0.0) -> dict[str, Any]:
        """Run ``epochs_per_update`` epochs of PPO over the buffer.

        Returns a dict of mean metrics for logging.
        """
        if len(self.buffer) < self.config.batch_size:
            return {"skipped": True, "reason": "buffer < batch_size"}

        advantages, returns = self.compute_gae(last_value)
        # Per-batch normalisation: improves stability (Engstrom et al.)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Move everything to the GPU once.
        states = torch.from_numpy(np.stack(self.buffer.states)).float().to(self.device)
        actions = torch.from_numpy(np.stack(self.buffer.actions)).float().to(self.device)
        old_log_probs = torch.tensor(self.buffer.log_probs, dtype=torch.float32, device=self.device)
        old_values = torch.tensor(self.buffer.values, dtype=torch.float32, device=self.device)
        adv_t = torch.from_numpy(advantages).to(self.device)
        ret_t = torch.from_numpy(returns).to(self.device)

        n = len(self.buffer)
        metrics = {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
            "kl": 0.0,
            "n_updates": 0,
            "early_stopped_at_epoch": -1,
        }

        self.policy.train()
        for epoch in range(self.config.epochs_per_update):
            idx = torch.randperm(n, device=self.device)
            for start in range(0, n, self.config.batch_size):
                b = idx[start : start + self.config.batch_size]

                new_lp, ent, new_val = self.policy.evaluate_actions(states[b], actions[b])
                ratio = (new_lp - old_log_probs[b]).exp()

                # --- Clipped policy objective ---
                surr1 = ratio * adv_t[b]
                surr2 = (
                    ratio.clamp(1.0 - self.config.clip_range, 1.0 + self.config.clip_range)
                    * adv_t[b]
                )
                policy_loss = -torch.min(surr1, surr2).mean()

                # --- Clipped value loss ---
                v_clip = old_values[b] + (new_val - old_values[b]).clamp(
                    -self.config.clip_range_vf,
                    self.config.clip_range_vf,
                )
                v_loss_unclipped = (new_val - ret_t[b]).pow(2)
                v_loss_clipped = (v_clip - ret_t[b]).pow(2)
                value_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()

                # --- Entropy bonus ---
                entropy_loss = -ent.mean()

                loss = (
                    policy_loss
                    + self.config.value_coef * value_loss
                    + self.config.entropy_coef * entropy_loss
                )

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.max_grad_norm)
                self.optimizer.step()

                metrics["policy_loss"] += policy_loss.item()
                metrics["value_loss"] += value_loss.item()
                metrics["entropy"] += ent.mean().item()
                metrics["n_updates"] += 1

            # --- Early stop on KL divergence ---
            with torch.no_grad():
                new_lp_full, _, _ = self.policy.evaluate_actions(states, actions)
                approx_kl = (old_log_probs - new_lp_full).mean().item()
            metrics["kl"] = approx_kl
            if approx_kl > 1.5 * self.config.target_kl:
                logger.warning(
                    f"PPO update early-stopped at epoch {epoch}: "
                    f"approx KL = {approx_kl:.4f} > 1.5 * target ({self.config.target_kl})"
                )
                metrics["early_stopped_at_epoch"] = epoch
                break

        # Mean over batches
        n_upd = max(metrics["n_updates"], 1)
        for k in ("policy_loss", "value_loss", "entropy"):
            metrics[k] /= n_upd

        self.buffer.clear()
        return metrics
