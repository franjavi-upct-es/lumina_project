# backend/ml_engine/evaluation/shap_explainer.py
"""
SHAP (SHapley Additive exPlanations) for model interpretability
Provides feature importance and individual prediction explanations
"""

import enum
from typing import Any
import numpy as np
import pandas as pd
import shap
from loguru import logger


class SHAPExplainer:
    """
    Wrapper for SHAP explanations with financial ML focus

    Supports:
    - Tree-based models (XGBoost, LightGBM, Random Forest)
    - Neural networks (PyTorch, TensorFlow)
    - Linear models
    """

    def __init__(
        self,
        model: Any,
        model_type: str = "auto",
        background_data: np.ndarray | None = None,
        feature_names: list[str] | None = None,
    ):
        """
        Initialize SHAP explainer

        Args:
            model: Trained model to explain
            model_type: Type of model ('tree', 'deep', 'linear', 'kernel', 'auto')
            background_data: Background dataset for explanations
            feature_names: Names of features
        """
        self.model = model
        self.model_type = model_type
        self.background_data = background_data
        self.feature_names = feature_names
        self.explainer: Any | None = None

        # Initialize explainer
        self._initialize_explainer()

    def _initialize_explainer(self):
        """Initialize appropiate SHAP explainer based on model type"""
        try:
            if self.model_type == "auto":
                self.model_type = self._detect_model_type()

            logger.info(f"Initializing {self.model_type} SHAP explainer")

            if self.model_type == "tree":
                # For tree-based models (XGBoost, LightGBM, RandomForest)
                self.explainer = shap.TreeExplainer(self.model)

            elif self.model_type == "deep":
                # For neural networks
                if self.background_data is None:
                    raise ValueError("Background data required for deep explainer")

                import torch

                if isinstance(self.model, torch.nn.Module):
                    self.explainer = shap.DeepExplainer(
                        self.model, torch.FloatTensor(self.background_data)
                    )
                else:
                    self.explainer = shap.DeepExplainer(self.model, self.background_data)

            elif self.model_type == "linear":
                # For linear models
                self.explainer = shap.LinearExplainer(self.model, self.background_data)

            elif self.model_type == "kernel":
                # Model-agnostic (slower but works for any model)
                if self.background_data is None:
                    raise ValueError("Background data required for kernel explaienr")

                def model_predict(x):
                    if hasattr(self.model, "predict"):
                        return self.model.predict(x)
                    elif hasattr(self.model, "forward"):
                        import torch

                        with torch.no_grad():
                            return self.model.forward(torch.FloatTensor(x)).cpu().numpy()
                    else:
                        return self.model(x)

                self.explainer = shap.KernelExplainer(
                    model_predict,
                    self.background_data[:100],  # Use subset for spped
                )
            else:
                raise ValueError(f"Unknown model type: {self.model_type}")

            logger.success("SHAP explainer initialize")

        except Exception as e:
            logger.error(f"Failed to initialize SHAP explainer: {e}")
            raise

    def _detect_model_type(self) -> str:
        """Auto-detect model type"""
        model_class = self.model.__class__.lower()

        if any(x in model_class for x in ["xgb", "lgbm", "forest", "tree"]):
            return "tree"
        elif any(x in model_class for x in ["neural", "network", "sequential", "module"]):
            return "deep"
        elif any(x in model_class for x in ["linear", "ridge", "lasso"]):
            return "linear"
        else:
            logger.warning(f"Could not detect model type for {model_class}, using kernel")
            return "kernel"

    def explain(self, X: np.ndarray, check_additivity: bool = False) -> shap.Explanation:
        """
        Generate SHAP explanations for data

        Args:
            X: Data to explain
            check_additivity: Whether to check SHAP additivity property

        Returns:
            SHAP Explanation object
        """
        logger.info(f"Generating SHAP explanations for {len(X)} samples")

        try:
            if self.model_type == "tree":
                shap_values = self.explainer.shap_values(X, check_additivity=check_additivity)
            else:
                shap_values = self.explainer.shap_values(X)

            # Handle multi-output models
            if isinstance(shap_values, list):
                shap_values = shap_values[0]

            # Create explanation object
            explanation = shap.Explanation(
                values=shap_values,
                base_values=self.explainer.expected_value,
                data=X,
                feature_names=self.feature_names,
            )

            logger.success("SHAP explanations generated")
            return explanation

        except Exception as e:
            logger.error(f"Error generating SHAP explanations: {e}")
            raise

    def get_feature_importance(self, X: np.ndarray, method: str = "mean_abs") -> pd.DataFrame:
        """
        Get global feature importance

        Args:
            X: Data to analyze
            method: Method to aggregate ('mean_abs', 'mean', 'max')

        Returns:
            DataFrame with feature importances
        """
        logger.info("Calculating feature importance")

        shap_values = self.explain(X).values

        # Aggregate SHAP values
        if method == "mean_abs":
            importance = np.abs(shap_values).mean(axis=0)
        elif method == "mean":
            importance = shap_values.mean(axis=0)
        elif method == "max":
            importance = np.abs(shap_values).max(axis=0)
        else:
            raise ValueError(f"Unknown method: {method}")

        # Create DataFrame
        feature_names = self.feature_names or [f"feature_{i}" for i in range(len(importance))]

        importance_df = pd.DataFrame({"feature": feature_names, "importance": importance})

        # Sort by importance
        importance_df = importance_df.sort_values("importance", ascending=False)

        logger.info(f"Top feature: {importance_df.iloc[0]['feature']}")

        return importance_df

    def explain_prediction(self, x: np.ndarray, top_n: int = 10) -> dict[str, Any]:
        """
        Explain a single prediction

        Args:
            x: Single sample to explain
            top_n: Number of top features to return

        Returns:
            Dictionary with explanation details
        """
        if x.ndim == 1:
            x = x.reshape(1, -1)

        explanation = self.explain(x)
        shap_values = explanation.values[0]

        # Get top positive and negative contributors
        sorted_idx = np.argsort(np.abs(shap_values))[::-1]
        top_indices = sorted_idx[:top_n]

        feature_names = self.feature_names or [f"feature_{i}" for i in range(len(shap_values))]

        top_features = []
        for idx in top_indices:
            top_features.append(
                {
                    "feature": feature_names[idx],
                    "shap_values": float(shap_values[idx]),
                    "feature_value": float(x[0, idx]),
                    "contribution": "positive" if shap_values[idx] > 0 else "negative",
                }
            )

        # Base value and prediction
        base_value = float(explanation.base_values)
        prediction = base_value + float(shap_values.sum())

        return {
            "base_value": base_value,
            "prediction": prediction,
            "shap_sum": float(shap_values.sum()),
            "top_features": top_features,
        }

    def get_dependence_data(
        self,
        X: np.ndarray,
        feature_name: str,
        interaction_feature: str | None = None,
    ) -> dict[str, np.ndarray]:
        """
        Get data for dependence plot

        Args:
            X: data
            feature_name: Feature to analyze
            interaction_feature: Feature for interaction coloring

        Returns:
            Dictionary with plot data
        """
        if feature_name not in self.feature_names:
            raise ValueError(f"Feature {feature_name} not found")

        feature_idx = self.feature_names.index(feature_name)
        shap_values = self.explain(X).values

        result = {"feature_values": X[:, feature_idx], "shap_values": shap_values[:, feature_idx]}

        if interaction_feature:
            if interaction_feature not in self.feature_names:
                raise ValueError(f"Feature {interaction_feature} not found")

            interaction_idx = self.feature_names.index(interaction_feature)
            result["interaction_values"] = X[:, interaction_idx]

        return result

    def get_interaction_values(
        self,
        X: np.ndarray,
        feature_names: list[str] | None = None,
    ) -> np.ndarray:
        """
        Calculate SHAP interaction values

        Args:
            X: Data
            feature_names: Specific features to analyze interactions

        Returns:
            Interaction values matrix
        """
        if self.model_type != "tree":
            logger.warning("Interaction values only available for tree models")
            return None

        logger.info("Calculating SHAP interaction values")

        try:
            interaction_values = self.explainer.shap_interaction_values(X)

            if isinstance(interaction_values, list):
                interaction_values = interaction_values[0]

            return interaction_values

        except Exception as e:
            logger.error(f"Error calculating interactions: {e}")
            return None

    def analyze_feature_interactions(
        self,
        X: np.ndarray,
        top_n: int = 10,
    ) -> pd.DataFrame:
        """
        Find top feature interactions

        Args:
            X: Data
            top_n: Number of top interactions to return

        Returns:
            DataFrame with top interactions
        """
        interaction_values = self.get_interaction_values(X)

        if interaction_values is None:
            return pd.DataFrame()

        # Average absolute interaction effects
        mean_abs_interactions = np.abs(interaction_values).mean(axis=0)

        # Find top interactions (off-diagonal)
        n_features = mean_abs_interactions.shape[0]
        interactions = []

        for i in range(n_features):
            for j in range(i + 1, n_features):
                interactions.append(
                    {
                        "feature_1": self.feature_names[i]
                        if self.feature_names
                        else f"feature_{i}",
                        "feature_2": self.feature_names[j]
                        if self.feature_names
                        else f"feature_{j}",
                        "interaction_strength": float(mean_abs_interactions[i, j]),
                    }
                )

        df = pd.DataFrame(interactions)
        df = df.sort_values("interaction_strength", ascending=False)

        return df.head(top_n)

    def get_summary_statistics(self, X: np.ndarray) -> dict[str, Any]:
        """
        Get summary statistics of SHAP values

        Args:
            X: Data to analyze

        Returns:
            Dictionary with statistics
        """
        shap_values = self.explain(X).values

        # Overall statistics
        total_importance = np.abs(shap_values).sum(axis=1).mean()

        # Per-feature statistics
        feature_stats = {}
        feature_names = self.feature_names or [f"feature_{i}" for i in range(shap_values.shape[1])]

        for i, name in enumerate(feature_names):
            values = shap_values[:, i]
            feature_stats[name] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "mean_abs": float(np.abs(values).mean()),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
            }

        return {
            "total_importance": float(total_importance),
            "num_samples": int(shap_values.shape[0]),
            "num_features": int(shap_values.shape[1]),
            "feature_statistics": feature_stats,
        }


class TimeSeriesSHAPExplainer(SHAPExplainer):
    """
    Extended SHAP explainer for time series models
    Handles temporal dependencies and sequence data
    """

    def __init__(self, model: Any, sequence_length: int, **kwargs):
        """
        Args:
            model: Trained model
            sequence_length: Length of input sequences
            **kwargs: Additional arguments for base class
        """
        self.sequence_length = sequence_length
        super().__init__(model, **kwargs)

    def explain_sequence(self, sequence: np.ndarray) -> dict[str, Any]:
        """
        Explain prediction for a sequence

        Args:
            sequence: Input sequence (sequence_length, n_features)

        Returns:
            Explanation with temporal information
        """
        if sequence.ndim == 2:
            sequence = sequence.reshape(1, *sequence.shape)

        # Get SHAP values for entire sequence
        explanation = self.explain(sequence)
        shap_values = explanation.values[0]  # (sequence_length, n_features)

        # Temporal importance (average across features)
        temporal_importance = np.abs(shap_values).mean(axis=1)

        # Most important time steps
        top_timesteps = np.argsort(temporal_importance)[::-1][:5]

        return {
            "temporal_importance": temporal_importance.tolist(),
            "most_important_timesteps": top_timesteps.tolist(),
            "shap_values": shap_values.tolist(),
            "total_contribution": float(shap_values.sum()),
        }


def compare_feature_importance(explainers: dict[str, SHAPExplainer], X: np.ndarray) -> pd.DataFrame:
    """
    Compare feature importance across multiple models

    Args:
        explainers: Dictionary of model_name -> SHAPExplainer
        X: Data to analyze

    Returns:
        DataFrame comparing feature importance
    """
    all_importance = {}

    for model_name, explainer in explainers.items():
        importance_df = explainer.get_feature_importance(X)
        all_importance[model_name] = importance_df.set_index("feature")["importance"]

    # Combine into single DataFrame
    comparison_df = pd.DataFrame(all_importance)

    # Add average importance
    comparison_df["average"] = comparison_df.mean(axis=1)

    # Sort by average importance
    comparison_df = comparison_df.sort_values("average", ascending=False)

    return comparison_df


def export_explanations(
    explainer: SHAPExplainer, X: np.ndarray, output_path: str, format: str = "csv"
):
    """
    Export SHAP explanations to file

    Args:
        explainer: SHAPExplainer instance
        X: Data to explain
        output_path: Path to save file
        format: Output format ('csv', 'json', 'parquet')
    """
    logger.info(f"Exporting explanations to {output_path}")

    explanation = explainer.explain(X)
    shap_values = explanation.values

    # Create DataFrame
    feature_names = explainer.feature_names or [f"feature_{i}" for i in range(shap_values.shape[1])]

    df = pd.DataFrame(shap_values, columns=feature_names)
    df["base_value"] = explanation.base_values
    df["prediction"] = explanation.base_values + shap_values.sum(axis=1)

    # Save
    if format == "csv":
        df.to_csv(output_path, index=False)
    elif format == "json":
        df.to_json(output_path, orient="records")
    elif format == "parquet":
        df.to_parquet(output_path)
    else:
        raise ValueError(f"Unknown format: {format}")

    logger.success(f"Explanations exported to {output_path}")
