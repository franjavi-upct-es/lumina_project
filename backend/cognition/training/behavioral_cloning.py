# backend/cognition/training/behavioral_cloning.py
"""Behavioral Cloning (BC) — Spartan Curriculum Phase A.

BC warm-starts the policy by maximum-likelihood imitation of an oracle.
Once the policy can reliably reproduce simple, sensible actions, the
PPO loop in Phase B has a much easier job: it begins inside a region of
parameter space that already produces *coherent* trajectories, instead
of starting from random noise.

Loss function
=============
We minimise the **negative log-likelihood** of the expert action under
the current policy:

    L(theta) = - E_{(s, a*) ~ D}  log pi_theta(a* | s)

This is the standard imitation-learning objective and has two technical
advantages over the obvious alternative of MSE on the squashed action:

1. **No tanh saturation**. For a tanh-squashed Gaussian, the action
   density goes through the change-of-variables factor
       log(1 - a^2 + eps),
   which is well-behaved everywhere in (-1, 1). An MSE loss on the
   squashed action would have near-zero gradient near +-1, which is
   precisely where many expert actions live (full long / full short).
2. **Calibrates the variance**. NLL pushes the policy to match the
   *distribution* of expert actions, not just their mode. This is what
   we want for a stochastic policy — Phase B's exploration depends on
   the learned log-std being sensible from the start.

Held-out validation
===================
We split the expert trajectories 90/10 *chronologically* (the first 90 %
go into training, the last 10 % into validation). Random shuffling
would be wrong here for the same reason it is wrong in time-series
forecasting: any temporal structure in the oracle leaks across the
boundary and inflates the held-out metric.

The reported ``accuracy`` is the fraction of validation samples whose
*deterministic* policy action has the same sign-pattern as the expert
across all four action dimensions. This is the metric the Spartan
Curriculum's gate (``bc_min_accuracy``, default 0.55) compares against.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from loguru import logger
from torch.utils.data import DataLoader, TensorDataset

from backend.cognition.agent.policy_network import PolicyNetwork


@dataclass(slots=True)
class BCMetrics:
    """Container returned by :meth:`BehavioralCloningTrainer.fit`.

    Exposed as a plain dict (via :py:meth:`__call__`-style ``asdict``)
    for the Spartan Curriculum, which logs the values to MLflow and
    consumes the ``accuracy`` key for its gate.
    """

    accuracy: float
    train_loss: float
    val_loss: float
    epochs: int
    n_train: int
    n_val: int

    def to_dict(self) -> dict[str, float]:
        return {
            "accuracy": self.accuracy,
            "train_loss": self.train_loss,
            "val_loss": self.val_loss,
            "epochs": float(self.epochs),
            "n_train": float(self.n_train),
            "n_val": float(self.n_val),
        }


class BehavioralCloningTrainer:
    """Train a :class:`PolicyNetwork` by NLL imitation of an oracle policy.

    Parameters
    ----------
    expert_states
        Array of shape ``(N, state_dim)``. Each row is a state observation
        sampled from the oracle's trajectories. Order matters — the last
        ``val_fraction`` of rows is held out for evaluation.
    expert_actions
        Array of shape ``(N, action_dim)``. Each row is the oracle's
        chosen action for the corresponding state. Values must lie in
        the open interval ``(-1, 1)``; we clip to ``(-1 + eps, 1 - eps)``
        before computing the NLL so :class:`SquashedGaussian.log_prob`'s
        ``atanh`` inversion stays finite.
    lr
        Adam learning rate. ``3e-4`` is the field-standard default for
        BC; the value here matches Phase B's PPO learning rate so the
        optimizer's internal state is meaningfully transferable.
    batch_size
        Mini-batch size. The default of 64 is a compromise between
        gradient noise (smaller = more noise = more regularisation) and
        wall-clock speed (bigger = better hardware utilisation).
    device
        ``"cuda"`` or ``"cpu"``. Auto-falls back to CPU if CUDA is not
        available so the trainer is robust in unit tests.
    val_fraction
        Fraction of the (chronologically ordered) data held out for
        validation. Defaults to ``0.10``.
    weight_decay
        L2 regularisation on the policy parameters. Important here
        because the BC dataset is typically small (a few thousand pairs)
        and over-fitting is easy.
    """

    # Clip applied to expert actions before computing NLL. Keeps the
    # atanh()-based inverse-squash finite — see SquashedGaussian.
    _ACTION_CLIP_EPS: float = 1e-4

    def __init__(
        self,
        expert_states: np.ndarray,
        expert_actions: np.ndarray,
        expert_weights: np.ndarray | None = None,
        lr: float = 3e-4,
        batch_size: int = 64,
        device: str = "cuda",
        val_fraction: float = 0.10,
        weight_decay: float = 1e-4,
    ) -> None:
        if expert_states.shape[0] != expert_actions.shape[0]:
            raise ValueError(
                f"expert_states and expert_actions have mismatched length: "
                f"{expert_states.shape[0]} vs {expert_actions.shape[0]}"
            )
        if expert_weights is not None and expert_weights.shape[0] != expert_states.shape[0]:
            raise ValueError(
                f"expert_weights and expert_states have mismatched length: "
                f"{expert_weights.shape[0]} vs {expert_states.shape[0]}"
            )
        if not 0.0 < val_fraction < 1.0:
            raise ValueError(f"val_fraction must be in (0, 1); got {val_fraction}")

        # Resolve device
        if device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA requested but unavailable; falling back to CPU")
            device = "cpu"

        self.device: str = device
        self.lr: float = lr
        self.batch_size: int = batch_size
        self.weight_decay: float = weight_decay

        n_total = expert_states.shape[0]
        n_val = max(1, int(n_total * val_fraction))
        n_train = n_total - n_val
        self.n_train = n_train
        self.n_val = n_val

        states_t = torch.from_numpy(expert_states).float()
        actions_t = (
            torch.from_numpy(expert_actions)
            .float()
            .clamp(
                -1.0 + self._ACTION_CLIP_EPS,
                1.0 - self._ACTION_CLIP_EPS,
            )
        )
        if expert_weights is not None:
            weights_t = torch.from_numpy(expert_weights).float()
        else:
            weights_t = torch.ones(n_total).float()

        train_ds = TensorDataset(states_t[:n_train], actions_t[:n_train], weights_t[:n_train])
        val_ds = TensorDataset(states_t[n_train:], actions_t[n_train:], weights_t[n_train:])

        self.train_loader: DataLoader[Any] = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
        )
        self.val_loader: DataLoader[Any] = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
        )

    # ----------------------------------------------------------- public API
    def fit(self, policy: PolicyNetwork, epochs: int = 20) -> dict[str, float]:
        """Train ``policy`` for ``epochs`` epochs and return summary metrics."""
        policy = policy.to(self.device)
        # We use eval() mode for BC. Why? Because the expert dataset is
        # usually tiny (a few hundred samples). Dropout would add too much
        # noise for the model to ever converge on these specific "pivotal"
        # corrections. We rely on weight_decay for regularisation instead.
        was_training = policy.training
        policy.eval()

        optim = torch.optim.AdamW(
            policy.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        train_losses: list[float] = []
        val_losses: list[float] = []

        best_val_loss = float("inf")
        best_state = None
        patience = 7
        trigger_times = 0

        for epoch in range(epochs):
            train_loss = self._train_one_epoch(policy, optim)
            val_loss, val_acc = self._validate(policy)
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            logger.info(
                f"BC epoch {epoch + 1}/{epochs}: "
                f"train_nll={train_loss:.4f}  val_nll={val_loss:.4f}  "
                f"val_acc={val_acc:.3f}"
            )

            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                trigger_times = 0
                best_state = {k: v.cpu() for k, v in policy.state_dict().items()}
            else:
                trigger_times += 1
                if trigger_times >= patience:
                    logger.warning(f"Early stopping at epoch {epoch + 1}")
                    policy.load_state_dict(best_state)
                    break

        policy.train(was_training)

        final_val_loss, final_accuracy = self._validate(policy)
        metrics = BCMetrics(
            accuracy=final_accuracy,
            train_loss=train_losses[-1] if train_losses else float("nan"),
            val_loss=final_val_loss,
            epochs=epochs,
            n_train=self.n_train,
            n_val=self.n_val,
        )
        return metrics.to_dict()

    # ---------------------------------------------------------------- inner
    def _train_one_epoch(
        self,
        policy: PolicyNetwork,
        optim: torch.optim.Optimizer,
    ) -> float:
        """Run one pass over the training loader and return the mean NLL."""
        policy.eval()  # Ensure dropout is off
        total_loss = 0.0
        total_samples = 0
        for states, actions, weights in self.train_loader:
            states = states.to(self.device)
            actions = actions.to(self.device)
            weights = weights.to(self.device)

            optim.zero_grad()
            log_prob, _entropy, _value = policy.evaluate_actions(states, actions)
            # Weighted NLL
            loss = -(log_prob * weights).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
            optim.step()
            total_loss += loss.item() * states.size(0)
            total_samples += states.size(0)
        return total_loss / max(total_samples, 1)

    @torch.no_grad()
    def _validate(
        self,
        policy: PolicyNetwork,
    ) -> tuple[float, float]:
        """Return ``(mean_val_nll, sign_match_accuracy)`` on the held-out split."""
        policy.eval()
        total_loss = 0.0
        total_samples = 0
        total_correct_dims = 0
        total_possible_dims = 0

        for states, actions, weights in self.val_loader:
            states = states.to(self.device)
            actions = actions.to(self.device)
            weights = weights.to(self.device)

            log_prob, _entropy, _value = policy.evaluate_actions(states, actions)
            total_loss += (-(log_prob * weights).mean()).item() * states.size(0)
            total_samples += states.size(0)

            sampled, _value = policy.sample(states, deterministic=True)
            action_pred = sampled.action

            # Accuracy is now average sign-match across all dimensions
            match_mask = self._sign_match_mask(action_pred, actions)
            total_correct_dims += int(match_mask.sum().item())
            total_possible_dims += match_mask.numel()

        mean_loss = total_loss / max(total_samples, 1)
        accuracy = total_correct_dims / max(total_possible_dims, 1)
        return mean_loss, accuracy

    @staticmethod
    def _sign_match_mask(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Per-element boolean mask: does this action dimension have matching sign?"""
        zero_tol = 0.1
        pred_class = torch.where(
            pred.abs() < zero_tol,
            torch.zeros_like(pred),
            pred.sign(),
        )
        target_class = torch.where(
            target.abs() < zero_tol,
            torch.zeros_like(target),
            target.sign(),
        )
        return pred_class == target_class
