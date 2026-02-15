# backend/cognition/training/curriculum.py
"""
Curriculum Learning for RL Agent Training

Implements the three-phase "Spartan" training curriculum:
- Phase A: Behavioral Cloning (Apprentice)
- Phase B: Domain Randomization (Matrix)
- Phase C: Self-Play / Pure RL (Master)

This progressive difficulty approach prevents the agent from flailing
randomly and builds robust trading strategies.

References:
- Bengio et al. (2009): "Curriculum Learning"
- Akkaya et al. (2019): "Solving Rubik's Cube with a Robot Hand"
- OpenAI et al. (2019): "Emergent Tool Use from Multi-Agent Interaction"

Mathematical Framework:
Let π_t be the policy at training step t, and D_t be the data distribution.

Phase A: π_t = argmin_π E_{(s,a)~D_expert}[||π(s) - a||²]
  (Minimize imitation error with expert demonstrations)

Phase B: π_t = argmax_π E_{(s,a,r)~D_random}[R(τ)]
  (Maximize return over randomized environments)

Phase C: π_t = argmax_π E_{(s,a,r)~D_real}[Sharpe(τ)]
  (Maximize risk-adjusted returns on real data)
"""

from dataclasses import dataclass
from enum import Enum

import numpy as np
from loguru import logger


class TrainingPhase(Enum):
    """Training phase enumeration."""

    PHASE_A_BEHAVIORAL_CLONING = "phase_a_behavioral_cloning"
    PHASE_B_DOMAIN_RANDOMIZATION = "phase_b_domain_randomization"
    PHASE_C_PURE_RL = "phase_c_pure_rl"


@dataclass
class PhaseConfig:
    """
    Configuration for a single training phase.

    Attributes:
        phase: Training phase identifier
        num_episodes: Number of episodes for this phase
        description: Human-readable description

        # Reward shaping
        reward_weights: Weights for different reward components
        use_shaped_reward: Whether to use reward shaping

        # Environment settings
        use_domain_randomization: Whether to apply domain randomization
        randomization_params: Parameters for randomization

        # Imitation learning (Phase A)
        use_behavioral_cloning: Whether to use BC loss
        bc_weight: Weight for BC loss (0 to 1)

        # Curriculum progression
        success_threshold: Required success rate to advance
        min_episodes: Minimum episodes before phase transition

        # Safety constraints
        max_drawdown_limit: Maximum allowed drawdown
        daily_loss_limit: Maximum daily loss
    """

    phase: TrainingPhase
    num_episodes: int
    description: str

    # Reward configuration
    reward_weights: dict[str, float] = {}
    use_shaped_reward: bool = True

    # Environment configuration
    use_domain_randomization: bool = False
    randomization_params: dict[str, any] = {}

    # Imitation learning
    use_behavioral_cloning: bool = False
    bc_weight: float = 0.0
    expert_data_path: str | None = None

    # Progression criteria
    success_threshold: float = 0.6
    min_episodes: int = 100

    # Safety
    max_drawdown_limit: float = 0.10
    daily_loss_limit: float = 0.03

    def __post_init__(self):
        if self.reward_weights is None:
            self.reward_weights = {"return": 1.0}
        if self.randomization_params is None:
            self.randomization_params = {}


class CurriculumScheduler:
    """
    Curriculum learning scheduler for progressive training.

    Manages transitions between training phases based on performance
    and automatically adjusts difficulty.

    Example:
        >>> scheduler = CurriculumScheduler()
        >>> phase_config = scheduler.get_current_phase_config()
        >>> # Train agent...
        >>> scheduler.update(episode_metrics)
        >>> if scheduler.should_advance():
        >>>     scheduler.advance_phase()
    """

    def __init__(
        self,
        phase_configs: list[PhaseConfig | None] = None,
        auto_advance: bool = True,
    ):
        """
        Initialize curriculum scheduler.

        Args:
            phase_configs: List of phase configurations (uses defaults if None)
            auto_advance: Automatically advance phases when criteria met
        """
        if phase_configs is None:
            phase_configs = self._create_default_phases()

        self.phase_configs = phase_configs
        self.auto_advance = auto_advance

        self.current_phase_idx = 0
        self.episodes_in_phase = 0
        self.phase_metrics_history: list[dict] = []

        logger.info(f"Curriculum scheduler initialized with {len(phase_configs)} phases")
        logger.info(f"Starting with: {self.current_phase.description}")

    @property
    def current_phase(self) -> PhaseConfig:
        """Get current phase configuration."""
        return self.phase_configs[self.current_phase_idx]

    @property
    def is_final_phase(self) -> bool:
        """Check if currently in final phase."""
        return self.current_phase_idx == len(self.phase_configs) - 1

    def _create_default_phases(self) -> list[PhaseConfig]:
        """
        Create default three-phase curriculum.

        Returns:
            List of phase configurations
        """
        phase_a = PhaseConfig(
            phase=TrainingPhase.PHASE_A_BEHAVIORAL_CLONING,
            num_episodes=1000,
            description="Phase A: Behavioral Cloning - Learn from expert (V2) demonstrations",
            # Imitation learning settings
            use_behavioral_cloning=True,
            bc_weight=0.5,  # Balance BC loss with RL loss
            # Simple reward (focus on matching expert)
            reward_weights={
                "return": 0.3,
                "imitation": 0.7,  # High weight on imitating expert
            },
            use_shaped_reward=True,
            # No randomization in Phase A (clean learning)
            use_domain_randomization=False,
            # Progression
            success_threshold=0.7,  # 70% success rate
            min_episodes=500,
            # Moderate safety constraints
            max_drawdown_limit=0.15,
            daily_loss_limit=0.05,
        )

        phase_b = PhaseConfig(
            phase=TrainingPhase.PHASE_B_DOMAIN_RANDOMIZATION,
            num_episodes=5000,
            description="Phase B: Domain Randomization - Train on adversarial scenarios",
            # No more BC (agent must adapt)
            use_behavioral_cloning=False,
            bc_weight=0.0,
            # Survival-focused reward
            reward_weights={
                "return": 0.3,
                "sharpe": 0.3,
                "drawdown_penalty": 0.4,  # High penalty for drawdowns
            },
            use_shaped_reward=True,
            # Heavy randomization
            use_domain_randomization=True,
            randomization_params={
                "volatility_multiplier": (1.0, 5.0),  # 1x to 5x volatility
                "spread_multiplier": (1.0, 3.0),  # Widen spreads
                "data_dropout_prob": 0.1,  # 10% missing candles
                "noise_std": 0.02,  # Add price noise
            },
            # Progression
            success_threshold=0.6,
            min_episodes=2000,
            # Strict safety (must survive chaos)
            max_drawdown_limit=0.10,
            daily_loss_limit=0.03,
        )

        phase_c = PhaseConfig(
            phase=TrainingPhase.PHASE_C_PURE_RL,
            num_episodes=10000,
            description="Phase C: Pure RL - Maximize Sharpe ratio on real data",
            # Pure RL (no imitation)
            use_behavioral_cloning=False,
            bc_weight=0.0,
            # Sharpe ratio maximization
            reward_weights={
                "sharpe": 0.6,
                "sortino": 0.2,
                "calmar": 0.2,
            },
            use_shaped_reward=False,  # Use raw risk-adjusted metrics
            # Mild randomization (still robust)
            use_domain_randomization=True,
            randomization_params={
                "volatility_multiplier": (0.8, 1.5),  # Moderate volatility range
                "spread_multiplier": (1.0, 1.2),  # Slight spread variation
            },
            # Progression (final phase - no advancement)
            success_threshold=0.65,
            min_episodes=5000,
            # Production-level safety
            max_drawdown_limit=0.10,
            daily_loss_limit=0.03,
        )

        return [phase_a, phase_b, phase_c]

    def update(self, episode_metrics: dict[str, float]):
        """
        Update curriculum with episode metrics.

        Args:
            episode_metrics: Metrics from completed episode
        """
        self.episodes_in_phase += 1
        self.phase_metrics_history.append(
            {
                "phase": self.current_phase.phase.value,
                "episode_in_phase": self.episodes_in_phase,
                **episode_metrics,
            }
        )

        # Check for auto-advance
        if self.auto_advance and self.should_advance():
            logger.info(
                f"Auto-advancing from {self.current_phase.phase.value} "
                f"after {self.episodes_in_phase} episodes"
            )
            self.advance_phase()

    def should_advance(self) -> bool:
        """
        Check if criteria are met to advance to next phase.

        Returns:
            bool: True if should advance
        """
        if self.is_final_phase:
            return False

        if self.episodes_in_phase < self.current_phase.min_episodes:
            return False

        # Calculate recent success rate (last 100 episodes or 20% of phase)
        window_size = min(100, max(20, int(self.current_phase.min_episodes * 0.2)))

        recent_metrics = self.phase_metrics_history[-window_size:]

        if len(recent_metrics) < window_size // 2:
            return False

        # Calculate success rate (episodes with positive return)
        successes = sum(1 for m in recent_metrics if m.get("total_return", 0) > 0)
        success_rate = successes / len(recent_metrics)

        # Check if threshold met
        meets_threshold = success_rate >= self.current_phase.success_threshold

        if meets_threshold:
            logger.info(
                f"Phase advancement criteria met: "
                f"success_rate={success_rate:.2%} >= "
                f"threshold={self.current_phase.success_threshold:.2%}"
            )

        return meets_threshold

    def advance_phase(self):
        """Advance to next training phase."""
        if self.is_final_phase:
            logger.warning("Already in final phase, cannot advance")
            return

        old_phase = self.current_phase.phase.value
        self.current_phase_idx += 1
        self.episodes_in_phase = 0

        logger.success(f"Advanced from {old_phase} to {self.current_phase.phase.value}")
        logger.info(f"New phase: {self.current_phase.description}")

    def get_current_phase_config(self) -> PhaseConfig:
        """Get current phase configuration."""
        return self.current_phase

    def get_phase_statistics(self) -> dict[str, any]:
        """
        Get statistics for current phase.

        Returns:
            dictionary of phase statistics
        """
        if not self.phase_metrics_history:
            return {
                "phase": self.current_phase.phase.value,
                "episodes": 0,
                "avg_return": 0.0,
                "success_rate": 0.0,
            }

        # Filter metrics for current phase
        phase_metrics = [
            m for m in self.phase_metrics_history if m["phase"] == self.current_phase.phase.value
        ]

        if not phase_metrics:
            return {
                "phase": self.current_phase.phase.value,
                "episodes": 0,
                "avg_return": 0.0,
                "success_rate": 0.0,
            }

        # Calculate statistics
        returns = [m.get("total_return", 0) for m in phase_metrics]
        successes = sum(1 for r in returns if r > 0)

        stats = {
            "phase": self.current_phase.phase.value,
            "episodes": len(phase_metrics),
            "avg_return": np.mean(returns),
            "success_rate": successes / len(phase_metrics),
            "avg_sharpe": np.mean([m.get("sharpe_ratio", 0) for m in phase_metrics]),
            "avg_drawdown": np.mean([m.get("max_drawdown", 0) for m in phase_metrics]),
        }

        return stats

    def reset_phase(self):
        """Reset current phase (useful for re-training)."""
        logger.warning(f"Resetting phase {self.current_phase.phase.value}")
        self.episodes_in_phase = 0

        # Clear metrics for current phase
        self.phase_metrics_history = [
            m for m in self.phase_metrics_history if m["phase"] != self.current_phase.phase.value
        ]

    def save_curriculum_state(self, path: str):
        """
        Save curriculum state to file.

        Args:
            path: Save path
        """
        import json

        state = {
            "current_phase_idx": self.current_phase_idx,
            "episodes_in_phase": self.episodes_in_phase,
            "metrics_history": self.phase_metrics_history,
        }

        with open(path, "w") as f:
            json.dump(state, f, indent=2)

        logger.info(f"Curriculum state saved to {path}")

    def load_curriculum_state(self, path: str):
        """
        Load curriculum state from file.

        Args:
            path: Load path
        """
        import json

        with open(path, "r") as f:
            state = json.load(f)

        self.current_phase_idx = state["current_phase_idx"]
        self.episodes_in_phase = state["episodes_in_phase"]
        self.phase_metrics_history = state["metrics_history"]

        logger.info(f"Curriculum state loaded from {path}")
        logger.info(f"Resumed at phase {self.current_phase.phase.value}")
