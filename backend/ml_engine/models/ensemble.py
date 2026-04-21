# backend/ml_engine/models/ensemble.py
"""
Ensemble model that combines multiple base models
Supports stacking, voting, and weighted averaging strategies
"""

import pickle
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, Ridge

from backend.ml_engine.models.base_model import BaseModel, ModelMetadata


class EnsembleModel(BaseModel):
    """
    Ensemble model that combines predictions from multiple models

    Strategies:
    - Simple averaging
    - Weighted averaging
    - Stacking (meta-learner)
    - Voting
    - Best model selection
    """

    def __init__(
        self,
        model_name: str,
        base_models: list[BaseModel] | None = None,
        strategy: str = "weighted_average",
        hyperparameters: dict[str, Any] | None = None,
    ):
        """
        Initialize ensemble model

        Args:
            model_name: Name of ensemble
            base_models: List of trained base models
            strategy: Ensemble strategy (weighted_average, stacking, voting, best)
            hyperparameters: Additional parameters
        """
        super().__init__(
            model_name=model_name, model_type="ensemble", hyperparameters=hyperparameters or {}
        )

        self.base_models = base_models or []
        self.strategy = strategy
        self.weights: np.ndarray | None = None
        self.meta_model: Any | None = None

        # Performance tracking
        self.model_performances: dict[str, float] = {}

        logger.info(
            f"Initialized ensemble with {len(self.base_models)} models, strategy: {strategy}"
        )

    def add_model(self, model: BaseModel, weight: float | None = None):
        """
        Add a base model to the ensemble

        Args:
            model: Trained model to add
            weight: Optional weight for this model
        """
        if not model.is_trained:  # type: ignore
            logger.warning(f"Adding untrained model {model.model_name}")  # type: ignore

        self.base_models.append(model)

        if weight is not None:
            if self.weights is None:
                self.weights = np.array([weight])
            else:
                self.weights = np.append(self.weights, weight)

        logger.info(f"Added {model.model_name} to ensemble (total: {len(self.base_models)})")  # type: ignore

    def build(self, input_shape: tuple[int, ...], **kwargs) -> Any:
        """
        Build ensemble (mainly for meta-model if using stacking)

        Args:
            input_shape: Shape of input data
            **kwargs: Additional parameters

        Returns:
            Ensemble model
        """
        if self.strategy == "stacking":
            # Initialize meta-model for stacking
            meta_model_type = kwargs.get("meta_model_type", "ridge")

            if meta_model_type == "ridge":
                self.meta_model = Ridge(alpha=kwargs.get("alpha", 1.0))
            elif meta_model_type == "lasso":
                self.meta_model = Lasso(alpha=kwargs.get("alpha", 1.0))
            elif meta_model_type == "rf":
                self.meta_model = RandomForestRegressor(
                    n_estimators=kwargs.get("n_estimators", 100),
                    max_depth=kwargs.get("max_depth", 5),
                    random_state=42,
                )
            else:
                self.meta_model = Ridge(alpha=1.0)

            logger.info(f"Built stacking ensemble with {meta_model_type} meta-model")

        return self

    @staticmethod
    def _as_array_prediction(prediction: np.ndarray | dict[str, np.ndarray]) -> np.ndarray:
        """Normalize model prediction outputs to a numpy array."""
        if isinstance(prediction, dict):
            if not prediction:
                raise ValueError("Prediction dictionary is empty")
            prediction = next(iter(prediction.values()))

        return np.asarray(prediction)

    def fit(
        self,
        X_train: np.ndarray | pd.DataFrame,
        y_train: np.ndarray | pd.Series,
        X_val: np.ndarray | pd.DataFrame | None = None,
        y_val: np.ndarray | pd.Series | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """
        Fit ensemble model

        For simple strategies: optimize weights on validation set
        For stacking: train meta-model on base model predictions

        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            **kwargs: Additional parameters

        Returns:
            Training history
        """
        logger.info(f"Training ensemble with {len(self.base_models)} models")

        if len(self.base_models) == 0:
            raise ValueError("No base models in ensemble")

        # Ensure all models are trained
        for model in self.base_models:
            if not model.is_trained:  # type: ignore
                raise ValueError(f"Model {model.model_name} is not trained")  # type: ignore

        # Strategy-specific fitting
        if self.strategy == "weighted_average":
            if X_val is not None and y_val is not None:
                self._optimize_weights(X_val, y_val)
            else:
                # Equal weights
                self.weights = np.ones(len(self.base_models)) / len(self.base_models)

        elif self.strategy == "stacking":
            self._train_stacking(X_train, y_train, X_val, y_val, **kwargs)

        elif self.strategy == "best":
            self._select_best_model(X_val, y_val)

        elif self.strategy == "simple_average":
            self.weights = np.ones(len(self.base_models)) / len(self.base_models)

        # Calculate ensemble performance
        train_metrics = {}  # type: ignore
        val_metrics = {}

        if X_val is not None and y_val is not None:
            predictions = self.predict(X_val)
            val_metrics = self.compute_metrics(y_val, predictions, prefix="val_")  # type: ignore

        # Store metadata
        self.meta_data = ModelMetadata(  # type: ignore
            model_id=self.model_name,  # type: ignore
            model_name=self.model_name,  # type: ignore
            model_type=self.model_type,  # type: ignore
            version="1.0",
            ticker=kwargs.get("ticker", "ENSEMBLE"),
            training_samples=len(X_train),
            hyperparameters={
                "strategy": self.strategy,
                "num_models": len(self.base_models),
                "weights": self.weights.tolist() if self.weights is not None else None,
            },
            feature_names=self.base_models[0].feature_names if self.base_models else [],  # type: ignore
            num_features=len(self.base_models[0].feature_names) if self.base_models else 0,  # type: ignore
            prediction_horizon=1,
            train_metrics=train_metrics,
            validation_metrics=val_metrics,
        )

        self.is_trained = True

        logger.success(f"Ensemble training complete with {self.strategy} strategy")

        return {
            "strategy": self.strategy,
            "weights": self.weights.tolist() if self.weights is not None else None,
            "val_metrics": val_metrics,
            "model_performances": self.model_performances,
        }

    def _optimize_weights(self, X_val: np.ndarray | pd.DataFrame, y_val: np.ndarray | pd.Series):
        """
        Optimize ensemble weights using validation data

        Args:
            X_val: Validation features
            y_val: Validation targets
        """
        from scipy.optimize import minimize

        logger.info("Optimizing ensemble weights")

        # Get predictions from all models
        model_predictions: list[np.ndarray] = []
        for model in self.base_models:
            pred = self._as_array_prediction(cast(Any, model).predict(X_val))
            model_predictions.append(pred)

        predictions_array = np.asarray(model_predictions)

        # Ensure y_val is 1D
        if isinstance(y_val, pd.Series):
            y_val = y_val.values
        if isinstance(y_val, pd.DataFrame):
            y_val = y_val.values.ravel()

        # Handle multi-step predictions
        if predictions_array.ndim == 3:  # (n_models, n_samples, n_horizons)
            predictions_2d = predictions_array[:, :, 0]  # Use first horizon
        else:
            predictions_2d = predictions_array

        # Objective function (MSE)
        def objective(weights):
            ensemble_pred = np.average(predictions_2d, axis=0, weights=weights)
            mse = np.mean((y_val - ensemble_pred) ** 2)
            return mse

        # Constraints: weights sum to 1, all non-negative
        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
        bounds = [(0, 1) for _ in range(len(self.base_models))]

        # Initial weights (equal)
        initial_weights = np.ones(len(self.base_models)) / len(self.base_models)

        # Optimize
        result = minimize(
            objective, initial_weights, bounds=bounds, constraints=constraints, method="SLSQP"
        )

        self.weights = result.x

        logger.info(f"Optimized weights: {self.weights}")

        # Store individual model performance
        for i, model in enumerate(self.base_models):
            pred = predictions_2d[i]
            mse = np.mean((y_val - pred) ** 2)
            self.model_performances[model.model_name] = float(mse)  # type: ignore

    def _train_stacking(
        self,
        X_train: np.ndarray | pd.DataFrame,
        y_train: np.ndarray | pd.Series,
        X_val: np.ndarray | pd.DataFrame | None = None,
        y_val: np.ndarray | pd.Series | None = None,
        **kwargs,
    ):
        """
        Train stacking ensemble with meta-model

        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            **kwargs: Additional parameters
        """
        logger.info("Training stacking ensemble")

        # Build meta-model if not already built
        if self.meta_model is None:
            self.build(input_shape=X_train.shape, **kwargs)

        # Get base model predictions on training data
        train_meta_features = []
        for model in self.base_models:
            pred = model.predict(X_train)  # type: ignore
            if pred.ndim > 1:  # type: ignore
                pred = pred[:, 0]  # First horizon if multi-step  # type: ignore
            train_meta_features.append(pred)

        train_meta_features = np.column_stack(train_meta_features)  # type: ignore

        # Train meta-model
        if isinstance(y_train, pd.Series):
            y_train = y_train.values

        self.meta_model.fit(train_meta_features, y_train)  # type: ignore

        logger.success("Stacking meta-model trained")

        # Calculate performances
        if X_val is not None and y_val is not None:
            val_meta_features = []
            for model in self.base_models:
                pred = model.predict(X_val)  # type: ignore
                if pred.ndim > 1:  # type: ignore
                    pred = pred[:, 0]  # type: ignore
                val_meta_features.append(pred)

            val_meta_features = np.column_stack(val_meta_features)  # type: ignore

            if isinstance(y_val, pd.Series):
                y_val = y_val.values

            # Store individual performances
            for i, model in enumerate(self.base_models):
                pred = val_meta_features[:, i]  # type: ignore
                mse = np.mean((y_val - pred) ** 2)
                self.model_performances[model.model_name] = float(mse)  # type: ignore

    def _select_best_model(
        self, X_val: np.ndarray | pd.DataFrame | None, y_val: np.ndarray | pd.Series | None
    ):
        """
        Select best performing model

        Args:
            X_val: Validation features
            y_val: Validation targets
        """
        if X_val is None or y_val is None:
            logger.warning("No validation data, using first model")
            self.weights = np.zeros(len(self.base_models))
            self.weights[0] = 1.0
            return

        logger.info("Selecting best model")

        # Evaluate all models
        performances = []
        for model in self.base_models:
            pred = model.predict(X_val)  # type: ignore
            if pred.ndim > 1:  # type: ignore
                pred = pred[:, 0]  # type: ignore

            if isinstance(y_val, pd.Series):
                y_val = y_val.values

            mse = np.mean((y_val - pred) ** 2)
            performances.append(mse)
            self.model_performances[model.model_name] = float(mse)  # type: ignore

        # Select best
        best_idx = np.argmin(performances)
        self.weights = np.zeros(len(self.base_models))
        self.weights[best_idx] = 1.0

        logger.info(
            f"Best model: {self.base_models[best_idx].model_name} (MSE: {performances[best_idx]:.4f})"  # type: ignore
        )

    def predict(self, X: np.ndarray | pd.DataFrame, **kwargs) -> np.ndarray | dict[str, np.ndarray]:
        """
        Make predictions using ensemble

        Args:
            X: Input features
            **kwargs: Additional parameters

        Returns:
            Ensemble predictions
        """
        if not self.is_trained:
            raise ValueError("Ensemble must be trained before prediction")

        if len(self.base_models) == 0:
            raise ValueError("No models in ensemble")

        # Get predictions from all models
        model_predictions: list[np.ndarray] = []
        for model in self.base_models:
            pred = self._as_array_prediction(cast(Any, model).predict(X, **kwargs))
            model_predictions.append(pred)

        predictions_array = np.asarray(model_predictions)

        # Combine based on strategy
        if self.strategy == "stacking":
            # Use meta-model
            if predictions_array.ndim == 3:  # Multi-step predictions
                predictions_array = predictions_array[:, :, 0]

            meta_features = predictions_array.T  # (n_samples, n_models)
            ensemble_pred = self.meta_model.predict(meta_features)  # type: ignore

        elif self.weights is not None:
            # Weighted average
            if predictions_array.ndim == 3:  # (n_models, n_samples, n_horizons)
                ensemble_pred = np.average(predictions_array, axis=0, weights=self.weights)  # type: ignore
            else:  # (n_models, n_samples)
                ensemble_pred = np.average(predictions_array, axis=0, weights=self.weights)  # type: ignore

        else:
            # Simple average
            ensemble_pred = np.mean(predictions_array, axis=0)  # type: ignore

        return ensemble_pred  # type: ignore

    def predict_with_uncertainty(
        self, X: np.ndarray | pd.DataFrame, **kwargs
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Make predictions with uncertainty estimates

        Uncertainty is measured as the standard deviation of base model predictions

        Args:
            X: Input features
            **kwargs: Additional parameters

        Returns:
            Tuple of (predictions, uncertainties)
        """
        # Get predictions from all models
        model_predictions: list[np.ndarray] = []
        for model in self.base_models:
            pred = self._as_array_prediction(cast(Any, model).predict(X, **kwargs))
            model_predictions.append(pred)

        predictions_array = np.asarray(model_predictions)

        # Ensemble prediction
        ensemble_pred = self._as_array_prediction(self.predict(X, **kwargs))

        # Uncertainty as standard deviation
        if predictions_array.ndim == 3:  # Multi-step
            uncertainty = np.std(predictions_array, axis=0)  # type: ignore
        else:
            uncertainty = np.std(predictions_array, axis=0)  # type: ignore

        return ensemble_pred, uncertainty

    def evaluate(
        self, X: np.ndarray | pd.DataFrame, y: np.ndarray | pd.Series, **kwargs
    ) -> dict[str, float]:
        """
        Evaluate ensemble performance

        Args:
            X: Features
            y: True targets
            **kwargs: Additional parameters

        Returns:
            Evaluation metrics
        """
        predictions = self.predict(X)

        # Compute metrics
        metrics = self.compute_metrics(y, predictions)  # type: ignore

        # Add individual model metrics
        for _i, model in enumerate(self.base_models):
            model_pred = model.predict(X)  # type: ignore
            if model_pred.ndim > 1:  # type: ignore
                model_pred = model_pred[:, 0]  # type: ignore

            model_metrics = self.compute_metrics(y, model_pred)  # type: ignore

            for metric_name, value in model_metrics.items():
                metrics[f"{model.model_name}_{metric_name}"] = value  # type: ignore

        return metrics  # type: ignore

    def get_feature_importance(self) -> dict[str, float] | None:
        """
        Get aggregated feature importance from base models

        Returns:
            Dictionary with feature importance
        """
        if not self.base_models:
            return None

        # Collect importance from all models
        all_importances = []

        for model in self.base_models:
            importance = model.get_feature_importance()  # type: ignore
            if importance is not None:
                all_importances.append(importance)

        if not all_importances:
            logger.warning("No feature importance available from base models")
            return None

        # Average importances
        feature_names = list(all_importances[0].keys())
        aggregate = {}

        for feature in feature_names:
            values = [imp.get(feature, 0.0) for imp in all_importances]

            # Weight by model weights if available
            if self.weights is not None:
                aggregate[feature] = np.average(values, weights=self.weights)
            else:
                aggregate[feature] = np.mean(values)

        # Sort by importance
        aggregate = dict(sorted(aggregate.items(), key=lambda x: x[1], reverse=True))

        return aggregate

    def _save_model(self, path: Path):
        """
        Save ensemble-specific components

        Args:
            path: Path to save
        """
        ensemble_data = {
            "strategy": self.strategy,
            "weights": self.weights,
            "meta_model": self.meta_model,
            "model_performances": self.model_performances,
            "base_model_names": [m.model_name for m in self.base_models],  # type: ignore
        }

        with open(path, "wb") as f:
            pickle.dump(ensemble_data, f)

        # Save best models
        for model in self.base_models:
            model_path = path.parent / f"{model.model_name}_model.pkl"  # type: ignore
            model._save_model(model_path)  # type: ignore

    def _load_model(self, path: Path):
        """
        Load ensemble-specific components

        Args:
            paths: Path to load from
        """
        with open(path, "rb") as f:
            ensemble_data = pickle.load(f)

        self.strategy = ensemble_data["strategy"]
        self.weights = ensemble_data["weights"]
        self.meta_model = ensemble_data["meta_model"]
        self.model_performances = ensemble_data["model_performances"]

        # Note: Base models need to be loaded separately
        logger.warning("Base models must be loaded separately and added to ensemble")

    def _log_model_to_mlflow(self):
        """
        Log ensemble to MLflow
        """
        import mlflow

        # Log ensemble config
        mlflow.log_param("ensemble_strategy", self.strategy)
        mlflow.log_param("num_base_models", len(self.base_models))

        if self.weights is not None:
            mlflow.log_dict(
                {f"weight_{i}": float(w) for i, w in enumerate(self.weights)},
                "ensemble_weights.json",
            )

        # Log model performances
        if self.model_performances:
            mlflow.log_dict(self.model_performances, "model_performances.json")

        # Log base model names
        mlflow.log_dict(
            {"base_models": [m.model_name for m in self.base_models]}, "base_models.json"
        )

    def get_model_contributions(
        self,
        X: np.ndarray | pd.DataFrame,
    ) -> dict[str, np.ndarray]:
        """
        Get contribution of each model to final prediction

        Args:
            X: Input features

        Returns:
            Dictionary mapping model names to their contributions
        """
        contributions = {}

        for i, model in enumerate(self.base_models):
            pred = model.predict(X)  # type: ignore

            if self.weights is not None:
                contribution = pred * self.weights[i]
            else:
                contribution = pred / len(self.base_models)  # type: ignore

            contributions[model.model_name] = contribution  # type: ignore

        return contributions
