# backend/cognition/training/trainer.py
"""
RL Training Loop with Curriculum Learning

Implements the main training loop for V3 agents with:
- Episode management and rollout collection
- MLflow experiment tracking and logging
- Early stopping based on performance
- Checkpointing and model persistence
- Evaluation on validation environments
- Integration with curriculum scheduler

This trainer supports both on-policy (PPO) and off-policy (SAC) algorithms
and handles the three-phase curriculum automatically.

References:
- Schulman et al. (2017): "Proximal Policy Optimization"
- Haarnoja et al. (2018): "Soft Actor-Critic"
- Stooke & Abbeel (2019): "RL Implementation Matters"
"""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from loguru import logger

try:
    import mlflow
    import mlflow.pytorch

    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    logger.warning("MLflow not available - experiment tracking disabled")

from backend.cognition.agent.ppo_continuous import PPOContinuousAgent
from backend.cognition.agent.sac_agent import SACAgent
from backend.cognition.training.curriculum import CurriculumScheduler


@dataclass
class TrainingConfig:
    """
    Training configuration.

    Attributes:
        # General
        experiment_name: Name for MLflow experiment
        run_name: Name for this training run
        seed: Random seed for reproducibility

        # Training
        num_episodes: Total number of training episodes
        max_steps_per_episode: Maximum steps per episode
        eval_interval: Episodes between evaluations
        save_interval: Episodes between checkpoints

        # Early stopping
        use_early_stopping: Whether to use early stopping
        patience: Episodes without improvement before stopping
        min_delta: Minimum improvement to reset patience

        # Logging
        log_interval: Episodes between logging
        verbose: Verbosity level (0=quiet, 1=normal, 2=debug)

        # Paths
        checkpoint_dir: Directory for checkpoints
        log_dir: Directory for logs
    """

    # General
    experiment_name: str = "lumina_v3_training"
    run_name: str = field(default_factory=lambda: f"run_{datetime.now():%Y%m%d_%H%M%S}")
    seed: int = 42

    # Training
    num_episodes: int = 10000
    max_steps_per_episode: int = 1000
    eval_interval: int = 100
    save_interval: int = 500

    # Early stopping
    use_early_stopping: bool = True
    patience: int = 50
    min_delta: float = 0.01

    # Logging
    log_interval: int = 10
    verbose: int = 1

    # Paths
    checkpoint_dir: str = "./checkpoints"
    log_dir: str = "./logs"


@dataclass
class TrainingMetrics:
    """Training metrics container."""

    episode: int = 0
    total_steps: int = 0

    # Episode metrics
    episode_return: float = 0.0
    episode_length: int = 0

    # Performance metrics
    avg_return: float = 0.0
    avg_sharpe: float = 0.0
    avg_drawdown: float = 0.0
    success_rate: float = 0.0

    # Training metrics
    policy_loss: float = 0.0
    value_loss: float = 0.0
    entropy: float = 0.0

    # Time
    episode_time: float = 0.0
    total_time: float = 0.0


class RLTrainer:
    """
    Main RL training loop with curriculum learning.

    Handles the complete training pipeline from initialization to
    final model deployment, with automatic phase transitions.

    Example:
        >>> from backend.simulation.environments import TradingEnv
        >>>
        >>> # Create environment
        >>> env = TradingEnv(tickers=["AAPL", "MSFT"])
        >>>
        >>> # Create agent
        >>> agent = PPOContinuousAgent(state_dim=224, action_dim=4)
        >>>
        >>> # Create trainer
        >>> trainer = RLTrainer(
        >>>     agent=agent,
        >>>     env=env,
        >>>     config=TrainingConfig(num_episodes=5000)
        >>> )
        >>>
        >>> # Train
        >>> trainer.train()
    """

    def __init__(
        self,
        agent: PPOContinuousAgent | SACAgent,
        env: Any,  # Gymnasium-compatible environment
        config: TrainingConfig | None = None,
        curriculum: CurriculumScheduler | None = None,
        eval_env: Any | None = None,
    ):
        """
        Initialize RL trainer.

        Args:
            agent: RL agent (PPO or SAC)
            env: Training environment
            config: Training configuration
            curriculum: Curriculum scheduler (creates default if None)
            eval_env: Evaluation environment (uses env if None)
        """
        self.agent = agent
        self.env = env
        self.config = config or TrainingConfig()
        self.curriculum = curriculum or CurriculumScheduler()
        self.eval_env = eval_env or env

        # Agent type detection
        self.agent_type = "PPO" if isinstance(agent, PPOContinuousAgent) else "SAC"

        # Metrics tracking
        self.metrics = TrainingMetrics()
        self.episode_history: list[dict] = []
        self.best_avg_return = -np.inf
        self.patience_counter = 0

        # Paths
        self.checkpoint_dir = Path(self.config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.log_dir = Path(self.config.log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Set seeds
        self._set_seeds(self.config.seed)

        # MLflow setup
        if MLFLOW_AVAILABLE:
            self._setup_mlflow()

        logger.info(f"RLTrainer initialized for {self.agent_type} agent")
        logger.info(f"Training config: {self.config}")
        logger.info(f"Starting curriculum phase: {self.curriculum.current_phase.description}")

    def _set_seeds(self, seed: int):
        """Set random seeds for reproducibility."""
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

    def _setup_mlflow(self):
        """Setup MLflow experiment tracking."""
        try:
            mlflow.set_experiment(self.config.experiment_name)
            mlflow.start_run(run_name=self.config.run_name)

            # Log configuration
            mlflow.log_params(
                {
                    "agent_type": self.agent_type,
                    "seed": self.config.seed,
                    "num_episodes": self.config.num_episodes,
                    "max_steps_per_episode": self.config.max_steps_per_episode,
                }
            )

            # Log curriculum phases
            for i, phase in enumerate(self.curriculum.phase_configs):
                mlflow.log_params(
                    {
                        f"phase_{i}_name": phase.phase.value,
                        f"phase_{i}_episodes": phase.num_episodes,
                    }
                )

            logger.success("MLflow tracking initialized")
        except Exception as e:
            logger.error(f"Failed to setup MLflow: {e}")

    def train(self) -> dict[str, any]:
        """
        Main training loop.

        Returns:
            Training results dictionary
        """
        logger.info("=" * 70)
        logger.info("🚀 Starting RL Training")
        logger.info("=" * 70)

        start_time = datetime.now()

        for episode in range(1, self.config.num_episodes + 1):
            self.metrics.episode = episode

            # Run episode
            episode_metrics = self._run_episode()

            # Update curriculum
            self.curriculum.update(episode_metrics)

            # Store metrics
            self.episode_history.append(episode_metrics)

            # Update moving averages
            self._update_averages()

            # Logging
            if episode % self.config.log_interval == 0:
                self._log_progress()

            # Evaluation
            if episode % self.config.eval_interval == 0:
                self._evaluate()

            # Checkpointing
            if episode % self.config.save_interval == 0:
                self._save_checkpoint()

            # Early stopping check
            if self.config.use_early_stopping:
                if self._check_early_stopping():
                    logger.warning(f"Early stopping triggered at episode {episode}")
                    break

        # Training complete
        total_time = (datetime.now() - start_time).total_seconds()

        logger.info("=" * 70)
        logger.success("✅ Training Complete!")
        logger.info(f"Total episodes: {episode}")
        logger.info(f"Total time: {total_time / 3600:.2f} hours")
        logger.info(f"Best avg return: {self.best_avg_return:.4f}")
        logger.info("=" * 70)

        # Save final model
        self._save_checkpoint(final=True)

        # End MLflow run
        if MLFLOW_AVAILABLE:
            mlflow.end_run()

        return {
            "episodes_trained": episode,
            "total_time": total_time,
            "best_avg_return": self.best_avg_return,
            "final_phase": self.curriculum.current_phase.phase.value,
        }

    def _run_episode(self) -> dict[str, float]:
        """
        Run single training episode.

        Returns:
            Episode metrics
        """
        episode_start = datetime.now()

        # Get current phase config
        phase_config = self.curriculum.get_current_phase_config()

        # Reset environment
        state, info = self.env.reset()

        episode_return = 0.0
        episode_steps = 0
        done = False

        # Episode loop
        while not done and episode_steps < self.config.max_steps_per_episode:
            # Select action
            if self.agent_type == "PPO":
                action, log_prob, value = self.agent.select_action(state)
            else:  # SAC
                action = self.agent.select_action(state)

            # Environment step
            next_state, reward, terminated, truncated, info = self.env.ste(action)
            done = terminated or truncated

            # Store transition
            if self.agent_type == "PPO":
                self.agent.store_transition(
                    state=state,
                    action=action,
                    log_prob=log_prob,
                    reward=reward,
                    value=value,
                    next_state=next_state,
                    done=done,
                )
            else:  # SAC
                self.agent.store_transition(
                    state=state, action=action, reward=reward, next_state=next_state, done=done
                )

            # Update state
            state = next_state
            episode_return += reward
            episode_steps += 1
            self.metrics.total_steps += 1

        # Update agent
        if self.agent_type == "PPO":
            # PPO updates at episode end
            update_metrics = self.agent.update(next_state if not done else None)
        else:
            # SAC updates every step (already done in loop via replay buffer)
            update_metrics = self.agent.update()

        # Episode time
        episode_time = (datetime.now() - episode_start).total_seconds()

        # Gather episode metrics
        episode_metrics = {
            "episode": self.metrics.episode,
            "total_return": episode_return,
            "episode_length": episode_steps,
            "episode_time": episode_time,
            "phase": phase_config.phase.value,
        }

        # Add training metrics
        if update_metrics:
            episode_metrics.update({f"train/{k}": v for k, v in update_metrics.items()})

        # Add environment info
        if info:
            episode_metrics.update({f"env/{k}": v for k, v in info.items()})

        return episode_metrics

    def _update_averages(self, window: int = 100):
        """Update moving average metrics."""
        recent = self.episode_history[-window:]

        if recent:
            self.metrics.avg_return = np.mean([e["total_return"] for e in recent])
            self.metrics.avg_sharpe = np.mean([e.get("env/sharpe_ratio", 0) for e in recent])
            self.metrics.avg_drawdown = np.mean([e.get("env/max_drawdown", 0) for e in recent])

            successes = sum(1 for e in recent if e["total_return"] > 0)
            self.metrics.success_rate = successes / len(recent)

    def _log_progress(self):
        """Log training progress."""
        phase_stats = self.curriculum.get_phase_statistics()

        if self.config.verbose >= 1:
            logger.info(
                f"Episode {self.metrics.episode} | "
                f"Phase: {phase_stats['phase']} | "
                f"Avg Return: {self.metrics.avg_return:.4f} | "
                f"Success Rate: {self.metrics.success_rate:.2%} | "
                f"Avg Sharpe: {self.metrics.avg_sharpe:.4f}"
            )

        # MLflow logging
        if MLFLOW_AVAILABLE:
            try:
                mlflow.log_metrics(
                    {
                        "avg_return": self.metrics.avg_return,
                        "success_rate": self.metrics.success_rate,
                        "avg_sharpe": self.metrics.avg_sharpe,
                        "avg_drawdown": self.metrics.avg_drawdown,
                    },
                    step=self.metrics.episode,
                )
            except Exception as e:
                logger.debug(f"MLflow logging failed: {e}")

    def _evaluate(self, num_episodes: int = 10):
        """
        Evaluate agent on eval environment.

        Args:
            num_episodes: Number of evaluation episodes
        """
        logger.info(f"Running evaluation ({num_episodes} episodes)...")

        eval_returns = []

        for _ in range(num_episodes):
            state, _ = self.eval_env.reset()
            episode_return = 0.0
            done = False
            steps = 0

            while not done and steps < self.config.max_steps_per_episode:
                # Use deterministic policy for evaluation
                if self.agent_type == "PPO":
                    action, _, _ = self.agent.select_action(state, deterministic=True)
                else:
                    action = self.agent.select_action(state, deterministic=True)

                state, reward, terminated, truncated, _ = self.eval_env.step(action)
                done = terminated or truncated
                episode_return += reward
                steps += 1

            eval_returns.append(episode_return)

        eval_mean = np.mean(eval_returns)
        eval_std = np.std(eval_returns)

        logger.info(f"Evaluation: Return = {eval_mean:.4f} ± {eval_std:.4f}")

        # MLflow logging
        if MLFLOW_AVAILABLE:
            mlflow.log_metrics(
                {
                    "eval_mean_return": eval_mean,
                    "eval_std_return": eval_std,
                },
                step=self.metrics.episode,
            )

        # Update best return
        if eval_mean > self.best_avg_return:
            improvement = eval_mean - self.best_avg_return
            self.best_avg_return = eval_mean
            self.patience_counter = 0
            logger.success(f"New best average return: {eval_mean:.4f} (+{improvement:.4f})")
        else:
            self.patience_counter += 1

    def _check_early_stopping(self) -> bool:
        """
        Check early stopping criteria.

        Returns:
            True if should stop
        """
        if self.patience_counter >= self.config.patience:
            logger.warning(
                f"No improvement for {self.patience_counter} evaluations "
                f"(patience={self.config.patience})"
            )
            return True
        return False

    def _save_checkpoint(self, final: bool = False):
        """
        Save training checkpoint.

        Args:
            final: Whether this is the final checkpoint
        """
        if final:
            filename = "final_model.pt"
        else:
            filename = f"checkpoint_episode_{self.metrics.episode}.pt"

        checkpoint_path = self.checkpoint_dir / filename

        # Save agent
        self.agent.save(str(checkpoint_path))

        # Save curriculum state
        curriculum_path = self.checkpoint_dir / f"curriculum_{self.metrics.episode}.json"
        self.curriculum.save_curriculum_state(str(curriculum_path))

        logger.info(f"Checkpoint saved: {checkpoint_path}")

        # MLflow artifact logging
        if MLFLOW_AVAILABLE:
            try:
                mlflow.log_artifact(str(checkpoint_path))
                mlflow.log_artifact(str(curriculum_path))
            except Exception as e:
                logger.debug(f"MLflow artifact logging failed: {e}")

    def load_checkpoint(self, checkpoint_path: str):
        """
        Load training checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file
        """
        self.agent.load(checkpoint_path)

        # Try to load curriculum state
        curriculum_path = Path(checkpoint_path).parent / "curriculum_state.json"
        if curriculum_path.exists():
            self.curriculum.load_curriculum_state(str(curriculum_path))

        logger.success(f"Checkpoint loaded from {checkpoint_path}")
