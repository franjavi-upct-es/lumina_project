# backend/ml_engine/models/base_model.py
"""
Base model class for all ML models in Lumina
Provides common interface and utilities for model training, evaluation, and deployment
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Tuple, Union
from datetime import datetime
from pathlib import Path
import json
import pickle
import numpy as np
import pandas as pd
from loguru import logger
from pydantic import BaseModel, Field
import mlflow


class ModelMetadata(BaseModel):
    """
    Metadata for a trained model
    """

    model_id: str
    model_name: str
    model_type: str  # 'lstm', 'transformer', 'xgboost', 'ensemble'
    version: str
    ticker: str

    # Training info
    trained_on: datetime = Field(default_factory=datetime.now)
    training_samples: int
    validation_samples: int
    test_samples: Optional[int] = None

    # Hyperparameters
    hyperparameters: Dict[str, Any]

    # Features
    feature_names: List[str]
    num_features: int
    sequence_length: Optional[int] = None
    prediction_horizon: int

    # Performance metrics
    train_metrics: Dict[str, float]
    validation_metrics: Dict[str, float]
    test_metrics: Optional[Dict[str, float]] = None

    # Feature importance (if available)
    feature_importance: Optional[Dict[str, float]] = None

    # MLflow tracking
    mlflow_run_id: Optional[str] = None
    mlflow_experiment_id: Optional[str] = None

    # Status
    is_active: bool = True

    # Additional meta_data
    notes: Optional[str] = None
    tags: Dict[str, str] = Field(default_factory=dict)

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


class BaseModel(ABC):
    """
    Abstract base class for all ML models

    All models (LSTM, Transformer, XGBoost, etc.) should inherit from this class
    and implement the required abstract methods.
    """

    def __init__(
        self,
        model_name: str,
        model_type: str,
        hyperparameters: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize base model

        Args:
            model_name: Unique name for this model
            model_type: Type of model (lstm, transformer, xgboost, etc.)
            hyperparameters: Model hyperparameters
        """
        self.model_name = model_name
        self.model_type = model_type
        self.hyperparameters = hyperparameters or {}

        # Model state
        self.is_trained = False
        self.meta_data: Optional[ModelMetadata] = None

        # Training history
        self.training_history: Dict[str, List[float]] = {
            "train_loss": [],
            "val_loss": [],
            "learning_rate": [],
        }

        # Feature names (set during training)
        self.feature_names: List[str] = []

        logger.info(f"Initialized {model_type} model: {model_name}")

    @abstractmethod
    def build(self, input_shape: Tuple[int, ...], **kwargs) -> Any:
        """
        Build the model architecture

        Args:
            input_shape: Shape of input data
            **kwargs: Additional build parameters

        Returns:
            Built model
        """
        pass

    @abstractmethod
    def fit(
        self,
        X_train: Union[np.ndarray, pd.DataFrame],
        y_train: Union[np.ndarray, pd.Series],
        X_val: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        y_val: Optional[Union[np.ndarray, pd.Series]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Train the model

        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
            **kwargs: Additional training parameters

        Returns:
            Dictionary with training history and metrics
        """
        pass

    @abstractmethod
    def predict(
        self, X: Union[np.ndarray, pd.DataFrame], **kwargs
    ) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        """
        Make predictions

        Args:
            X: Input features
            **kwargs: Additional prediction parameters

        Returns:
            Predictions (can be array or dict with multiple outputs)
        """
        pass

    @abstractmethod
    def evaluate(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        **kwargs,
    ) -> Dict[str, float]:
        """
        Evaluate model performance

        Args:
            X: Features
            y: True targets
            **kwargs: Additional evaluation parameters

        Returns:
            Dictionary with evaluation metrics
        """
        pass

    def predict_with_uncertainty(
        self, X: Union[np.ndarray, pd.DataFrame], **kwargs
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions with uncertainty estimates

        Default implementation returns predictions without uncertainty.
        Override in subclasses for models that support uncertainty quantification.

        Args:
            X: Input features
            **kwargs: Additional parameters

        Returns:
            Tuple of (predictions, uncertainties)
        """
        predictions = self.predict(X, **kwargs)
        uncertainties = np.zeros_like(predictions)
        return predictions, uncertainties

    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """
        Get feature importance scores

        Override in subclasses that support feature importance.

        Returns:
            Dictionary mapping feature names to importance scores
        """
        logger.warning(f"{self.model_type} does not support feature importance")
        return None

    def save(self, path: Union[str, Path]) -> str:
        """
        Save model to disk

        Args:
            path: Directory to save model

        Returns:
            Path where model was saved
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save model-specific components (implemented by subclass)
        model_path = path / f"{self.model_name}_model.pkl"
        self._save_model(model_path)

        # Save meta_data
        if self.meta_data:
            metadata_path = path / f"{self.model_name}_metadata.json"
            with open(metadata_path, "w") as f:
                json.dump(self.meta_data.dict(), f, indent=2, default=str)

        # Save training history
        history_path = path / f"{self.model_name}_history.json"
        with open(history_path, "w") as f:
            json.dump(self.training_history, f, indent=2)

        # Save feature names
        features_path = path / f"{self.model_name}_features.json"
        with open(features_path, "w") as f:
            json.dump(self.feature_names, f, indent=2)

        logger.info(f"Model saved to {path}")
        return str(path)

    @abstractmethod
    def _save_model(self, path: Path):
        """
        Save model-specific components

        Implemented by subclasses
        """
        pass

    def load(self, path: Union[str, Path]) -> "BaseModel":
        """
        Load model from disk

        Args:
            path: Directory containing saved model

        Returns:
            Loaded model instance
        """
        path = Path(path)

        # Load model-specific components
        model_path = path / f"{self.model_name}_model.pkl"
        self._load_model(model_path)

        # Load meta_data
        metadata_path = path / f"{self.model_name}_metadata.json"
        if metadata_path.exists():
            with open(metadata_path, "r") as f:
                metadata_dict = json.load(f)
                self.meta_data = ModelMetadata(**metadata_dict)

        # Load training history
        history_path = path / f"{self.model_name}_history.json"
        if history_path.exists():
            with open(history_path, "r") as f:
                self.training_history = json.load(f)

        # Load feature names
        features_path = path / f"{self.model_name}_features.json"
        if features_path.exists():
            with open(features_path, "r") as f:
                self.feature_names = json.load(f)

        self.is_trained = True
        logger.info(f"Model loaded from {path}")
        return self

    @abstractmethod
    def _load_model(self, path: Path):
        """
        Load model-specific components

        Implemented by subclasses
        """
        pass

    def save_to_mlflow(
        self, experiment_name: str, run_name: Optional[str] = None
    ) -> str:
        """
        Save model to MLflow

        Args:
            experiment_name: Name of MLflow experiment
            run_name: Name for this run (optional)

        Returns:
            MLflow run ID
        """
        # Set experiment
        mlflow.set_experiment(experiment_name)

        # Start run
        with mlflow.start_run(run_name=run_name):
            # Log parameters
            mlflow.log_params(self.hyperparameters)

            # Log metrics
            if self.meta_data:
                mlflow.log_metrics(self.meta_data.train_metrics)
                if self.meta_data.validation_metrics:
                    mlflow.log_metrics(
                        {
                            f"val_{k}": v
                            for k, v in self.meta_data.validation_metrics.items()
                        }
                    )

            # Log model
            self._log_model_to_mlflow()

            # Get run ID
            run_id = mlflow.active_run().info.run_id

            # Update meta_data
            if self.meta_data:
                self.meta_data.mlflow_run_id = run_id
                self.meta_data.mlflow_experiment_id = (
                    mlflow.active_run().info.experiment_id
                )

            logger.info(f"Model saved to MLflow with run_id: {run_id}")
            return run_id

    @abstractmethod
    def _log_model_to_mlflow(self):
        """
        Log model-specific components to MLflow

        Implemented by subclasses
        """
        pass

    def compute_metrics(
        self, y_true: np.ndarray, y_pred: np.ndarray, prefix: str = ""
    ) -> Dict[str, float]:
        """
        Compute standard regression metrics

        Args:
            y_true: True values
            y_pred: Predicted values
            prefix: Prefix for metric names

        Returns:
            Dictionary of metrics
        """
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

        # Handle multi-output predictions
        if len(y_pred.shape) > 1 and y_pred.shape[1] > 1:
            # Take first output if multiple
            y_pred = y_pred[:, 0]

        if len(y_true.shape) > 1 and y_true.shape[1] > 1:
            y_true = y_true[:, 0]

        # Ensure 1D arrays
        y_true = y_true.flatten()
        y_pred = y_pred.flatten()

        # Remove NaN values
        mask = ~(np.isnan(y_true) | np.isnan(y_pred))
        y_true = y_true[mask]
        y_pred = y_pred[mask]

        if len(y_true) == 0:
            logger.warning("No valid samples for metric computation")
            return {}

        metrics = {
            f"{prefix}mae": float(mean_absolute_error(y_true, y_pred)),
            f"{prefix}rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            f"{prefix}r2": float(r2_score(y_true, y_pred)),
            f"{prefix}mape": float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100),
        }

        return metrics

    def get_model_summary(self) -> Dict[str, Any]:
        """
        Get summary of model information

        Returns:
            Dictionary with model summary
        """
        summary = {
            "model_name": self.model_name,
            "model_type": self.model_type,
            "is_trained": self.is_trained,
            "hyperparameters": self.hyperparameters,
            "num_features": len(self.feature_names),
        }

        if self.meta_data:
            summary.update(
                {
                    "trained_on": self.meta_data.trained_on.isoformat(),
                    "training_samples": self.meta_data.training_samples,
                    "validation_samples": self.meta_data.validation_samples,
                    "train_metrics": self.meta_data.train_metrics,
                    "validation_metrics": self.meta_data.validation_metrics,
                }
            )

        return summary

    def __repr__(self) -> str:
        """String representation"""
        status = "trained" if self.is_trained else "untrained"
        return f"{self.__class__.__name__}(name='{self.model_name}', type='{self.model_type}', status='{status}')"


class EnsembleModel(BaseModel):
    """
    Base class for ensemble models that combine multiple models
    """

    def __init__(
        self, model_name: str, base_models: Optional[List[BaseModel]] = None, **kwargs
    ):
        """
        Initialize ensemble model

        Args:
            model_name: Name of ensemble
            base_models: List of base models to ensemble
            **kwargs: Additional parameters
        """
        super().__init__(model_name, "ensemble", kwargs)
        self.base_models = base_models or []
        self.weights: Optional[np.ndarray] = None

    def add_model(self, model: BaseModel):
        """Add a model to the ensemble"""
        self.base_models.append(model)
        logger.info(f"Added {model.model_name} to ensemble")

    def predict(self, X: Union[np.ndarray, pd.DataFrame], **kwargs) -> np.ndarray:
        """
        Make predictions by combining all base models

        Args:
            X: Input features
            **kwargs: Additional parameters

        Returns:
            Ensemble predictions
        """
        if not self.base_models:
            raise ValueError("No models in ensemble")

        # Get predictions from all models
        predictions = []
        for model in self.base_models:
            pred = model.predict(X, **kwargs)
            predictions.append(pred)

        predictions = np.array(predictions)

        # Weighted average
        if self.weights is not None:
            ensemble_pred = np.average(predictions, axis=0, weights=self.weights)
        else:
            ensemble_pred = np.mean(predictions, axis=0)

        return ensemble_pred

    def optimize_weights(
        self,
        X_val: Union[np.ndarray, pd.DataFrame],
        y_val: Union[np.ndarray, pd.Series],
    ):
        """
        Optimize ensemble weights using validation data

        Args:
            X_val: Validation features
            y_val: Validation targets
        """
        from scipy.optimize import minimize

        # Get predictions from all models
        predictions = []
        for model in self.base_models:
            pred = model.predict(X_val)
            predictions.append(pred)

        predictions = np.array(predictions)

        # Objective function (MSE)
        def objective(weights):
            ensemble_pred = np.average(predictions, axis=0, weights=weights)
            mse = np.mean((y_val - ensemble_pred) ** 2)
            return mse

        # Constraints: weights sum to 1, all positive
        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
        bounds = [(0, 1) for _ in range(len(self.base_models))]

        # Initial weights (equal)
        initial_weights = np.ones(len(self.base_models)) / len(self.base_models)

        # Optimize
        result = minimize(
            objective,
            initial_weights,
            bounds=bounds,
            constraints=constraints,
            method="SLSQP",
        )

        self.weights = result.x
        logger.info(f"Optimized ensemble weights: {self.weights}")

    def build(self, input_shape: Tuple[int, ...], **kwargs):
        """Build is not needed for ensemble"""
        pass

    def fit(self, X_train, y_train, X_val=None, y_val=None, **kwargs):
        """Ensemble doesn't train, base models are already trained"""
        if X_val is not None and y_val is not None:
            self.optimize_weights(X_val, y_val)
        return {}

    def evaluate(self, X, y, **kwargs) -> Dict[str, float]:
        """Evaluate ensemble"""
        predictions = self.predict(X)
        return self.compute_metrics(y, predictions)

    def _save_model(self, path: Path):
        """Save ensemble weights"""
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "weights": self.weights,
                    "model_names": [m.model_name for m in self.base_models],
                },
                f,
            )

    def _load_model(self, path: Path):
        """Load ensemble weights"""
        with open(path, "rb") as f:
            data = pickle.load(f)
            self.weights = data["weights"]

    def _log_model_to_mlflow(self):
        """Log ensemble to MLflow"""
        mlflow.log_dict(
            {
                "weights": self.weights.tolist() if self.weights is not None else None,
                "model_names": [m.model_name for m in self.base_models],
            },
            "ensemble_config.json",
        )
