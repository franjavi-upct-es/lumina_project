# backend/cognition/agent/uncertainty.py
"""
Uncertainty Estimation for RL Agent

Implements uncertainty quantification techniques to detect when the agent
is operating outside its training distribution (epistemic uncertainty).

This is a critical safety component that prevents the agent from making
confident predictions on unfamiliar market conditions.

Techniques:
1. Monte Carlo Dropout: Run forward pass multiple times with dropout enabled
2. Deep Ensembles: Train multiple models and measure prediction variance
3. Entropy-based uncertainty: Measure policy entropy

References:
- Gal & Ghahramani (2016): "Dropout as a Bayesian Approximation"
- Lakshminarayanan et al. (2017): "Simple and Scalable Predictive Uncertainty Estimation"
- Osband et al. (2016): "Deep Exploration via Bootstrapped DQN"

Safety Integration:
When uncertainty > CRITICAL_THRESHOLD (typically 0.8):
- Agent action is discarded
- Safety arbitrator takes control
- System moves to defensive mode or cash
"""

import torch
import torch.nn as nn
from loguru import logger


class MonteCarloDropout:
    """
    Monte Carlo Dropout for uncertainty estimation.

    Enables dropout during inference and runs multiple forward passes
    to estimate model uncertainty through prediction variance.

    Key Insight:
    Dropout can be interpreted as approximate Bayesian inference.
    Running multiple passes with different dropout masks samples from
    the posterior distribution over network weights.
    """

    def __init__(
        self,
        model: nn.Module,
        n_samples: int = 10,
        dropout_rate: float = 0.1,
    ):
        """
        Initialize Monte Carlo Dropout.

        Args:
            model: Neural network model
            n_samples: Number of forward passes
            dropout_rate: Dropout probability
        """
        self.model = model
        self.n_samples = n_samples
        self.dropout_rate = dropout_rate

        # Enable dropout modules during inference
        self._enable_dropout()

    def _enable_dropout(self):
        """Enable dropout layers during inference."""
        for module in self.model.modules():
            if isinstance(module, nn.Dropout):
                module.train()  # Keep in training mode

    def predict_with_uncertainty(
        self, state: torch.Tensor, return_all_samples: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """
        Predict action with uncertainty estimate.

        Args:
            state: Input state
            return_all_samples: If True, return all MC samples

        Returns:
            mean_action: Mean predicted action
            uncertainty: Standard deviation across samples (epistemic uncertainty)
            all_samples: All MC samples (if requested)
        """
        samples = []

        # Run multiple forward passes
        for _ in range(self.n_samples):
            with torch.no_grad():
                # Get action distribution
                action_dist, _ = self.model(state)

                # Sample action
                action = action_dist.sample()
                samples.append(action)

        # Stack samples
        samples = torch.stack(samples)  # [n_samples, batch_size, action_dim]

        # Compute statistics
        mean_action = samples.mean(dim=0)
        std_action = samples.std(dim=0)

        # Uncertainty score (average std across)
        uncertainty = std_action.mean(dim=-1)

        if return_all_samples:
            return mean_action, uncertainty, samples
        else:
            return mean_action, uncertainty, None

    def estimate_epistemic_uncertainty(
        self,
        state: torch.Tensor,
    ) -> float:
        """
        Estimate epistemic uncertainty from ensemble disagreement.

        Args:
            state: Input shape

        Returns:
            uncertainty_score: Normalized uncertainty score [0, 1]
        """
        _, uncertainty, _ = self.predict_with_uncertainty(state)

        # Normalized to [0, 1]
        uncertainty_score = torch.clamp(uncertainty / 0.5, 0, 1)

        return uncertainty_score

    def get_ensemble_statistics(
        self,
        state: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Get comprehensive ensemble statistics.

        Args:
            state: Input state

        Returns:
            statistics: Dictionary of ensemble statistics
        """
        mean_action, uncertainty, predictions = self.predict_with_uncertainty(
            state, return_all_samples=True
        )

        # Compute additional statistics
        min_action = predictions.min(dim=0)[0]
        max_action = predictions.max(dim=0)[0]
        median_action = predictions.median(dim=0)[0]

        # Coefficient of variation (relative uncertainty)
        cv = uncertainty / (torch.abs(mean_action) + 1e-8)

        return {
            "mean": mean_action,
            "std": uncertainty,
            "min": min_action,
            "max": max_action,
            "median": median_action,
            "coefficient_of_variation": cv,
            "range": max_action - min_action,
        }


class UncertaintyEstimator:
    """
    Unified uncertainty estimator combining multiple techniques.

    This is the main interface for uncertainty estimation in the V3 system.
    It combines MC Dropout and ensemble methods with additional heuristics.
    """

    def __init__(
        self,
        model: nn.Module,
        method: str = "mc_dropout",
        n_samples: int = 10,
        ensemble_models: list[nn.Module] | None = None,
        critical_threshold: float = 0.8,
    ):
        """
        Initialize uncertainty estimator.

        Args:
            model: Primary model
            method: Uncertainty of MC samples
            n_samples: Number of MC samples
            ensemble_models: List of ensemble models (for ensemble method)
            critical_threshold: Threshold for safety intervention
        """
        self.method = method
        self.critical_threshold = critical_threshold

        if method == "mc_dropout":
            if ensemble_models is None:
                raise ValueError("Ensemble models required for ensemble method")
            self.estimator = DeepEnsemble(models=ensemble_models)
        else:
            raise ValueError(f"Unknown uncertainty method: {method}")

        logger.info(f"Uncertainty estimator intialized with method={method}")

    def estimate_uncertainty(
        self,
        state: torch.Tensor,
        return_action: bool = True,
    ) -> tuple[float, torch.Tensor | None]:
        """
        Estimate uncertainty for state.

        Args:
            state: Input shape
            return_action: If True, also return predicted action

        Returns:
            uncertainty_score: Uncertainty score [0, 1]
            action: Predicted action (if requested)
        """
        if self.method == "mc_dropout":
            mean_action, uncertainty, _ = self.estimator.predict_with_uncertainty(state)
        else:  # ensemble
            mean_action, uncertainty, _ = self.estimator.predict_with_uncertainty(state)

        # Normalize uncertainty to [0, 1]
        uncertainty_score = torch.clamp(uncertainty / 0.5, 0, 1).item()

        if return_action:
            return uncertainty_score, mean_action
        else:
            return uncertainty_score, None

    def is_safe_to_act(self, state: torch.Tensor) -> tuple[bool, float, str]:
        """
        Determine if it's safe to use model's action.

        This is the core safety check. If uncertainty is too high,
        the system should fall back to defensive strategies.

        Args:
            state: Input shape

        Returns:
            is_safe: True if uncertainty is below threshold
            uncertainty_score: Uncertainty score
            recommendation: Safety recommendation
        """
        uncertainty_score, _ = self.estimate_uncertainty(state, return_action=False)

        if uncertainty_score > self.critical_threshold:
            is_safe = False
            recommendation = "REJECT - High epistemic uncertainty. State likely OOD"
        elif uncertainty_score > self.critical_threshold * 0.7:
            is_safe = True
            recommendation = "CAUTION - Moderate uncertainty. Reduce position size."
        else:
            is_safe = True
            recommendation = "CONFIDENT - Low uncertainty. Normal operation."

        return is_safe, uncertainty_score, recommendation

    def get_detailed_analysis(self, state: torch.Tensor) -> dict[str, any]:
        """
        Get detailed uncertainty analysis.

        Args:
            state: Input state

        Returns:
            analysis: Comprehensive uncertainty analysis
        """
        uncertainty_score, action = self.estimate_uncertainty(state, return_action=True)
        is_safe, _, recommendation = self.is_safe_to_act(state)

        # Get prediction interval (95% confidence)
        if self.method == "mc_dropout":
            lower, upper = self.estimator.get_prediction_interval(state, confidence=0.95)
        else:
            # For ensemble, use min/max
            stats = self.estimator.get_ensemble_statistics(state)
            lower = stats["min"]
            upper = stats["max"]

        # Compute action range
        action_range = (upper - lower).abs().mean().item()

        return {
            "uncertainty_score": uncertainty_score,
            "is_safe": is_safe,
            "recommendation": recommendation,
            "predicted_action": action.cpu().numpy() if action is not None else None,
            "prediction_interval": {
                "lower": lower.cpu().numpy(),
                "upper": upper.cpu().numpy(),
                "range": action_range,
            },
            "method": self.method,
            "critical_threshold": self.critical_threshold,
        }

    def log_uncertainty_event(
        self,
        state: torch.Tensor,
        ticker: str,
        context: dict | None = None,
    ):
        """
        Log high uncertainty event for monitoring.

        Args:
            state: Input state
            ticker: Stock ticker
            context: Additional context
        """
        analysis = self.get_detailed_analysis(state)

        if analysis["uncertainty_score"] > self.critical_threshold:
            logger.warning(
                f"HIGH UNCERTAINTY DETECTED for {ticker}: "
                f"score={analysis['uncertainty_score']:.3f} "
                f"(threshold={self.critical_threshold})"
            )
            logger.warning(f"Recommendation: {analysis['recommendation']}")

            if context:
                logger.debug(f"Context: {context}")
