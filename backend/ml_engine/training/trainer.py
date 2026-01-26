# backend/ml_engine/training/trainer.py
"""
Generic trainer for all ML models with unified interface
Supports XGBoost, LSTM, Transformers, and ensemble models
"""

import json
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pandas as pd
import polars as pl
from loguru import logger
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from backend.config.settings import get_settings
from backend.ml_engine.evaluation.metrics import calculate_directional_accuracy
from backend.ml_engine.models.base_model import BaseModel
from backend.ml_engine.training.purged_cv import PurgedKFold, calculate_sample_weights

settings = get_settings()


class ModelTrainer:
    """
    Universal model trainer with advanced features

    Features:
    - Purged cross-validation for time series
    - Multiple evaluation metrics
    - Early stopping
    - Learning rate scheduling
    - Model checkpointing
    - MLflow integration
    - Feature importance tracking
    """

    def __init__(
        self, model: BaseModel, experiment_name: str | None = None, use_mlflow: bool = True
    ):
        """
        Initialize trainer

        Args:
            model: Model instance implementing BaseModel interface
            experiment_name: MLflow experiment name
            use_mlflow: Whether to use MLflow for tracking
        """
        self.model = model
        self.experiment_name = experiment_name or f"{model.model_type}_training"
        self.use_mlflow = use_mlflow

        # Training history
        self.history: dict[str, list[float]] = {
            "train_loss": [],
            "val_loss": [],
            "train_mae": [],
            "val_mae": [],
            "learning_rate": [],
        }

        # Best model state
        self.best_score = float("inf")
        self.best_epoch = 0
        self.patience_counter = 0

        if self.use_mlflow:
            mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
            mlflow.set_experiment(self.experiment_name)

        logger.info(f"Initialized ModelTrainer for {model.model_type}")

    def train(
        self,
        X_train: np.ndarray | pd.DataFrame | pl.DataFrame,
        y_train: np.ndarray | pd.Series,
        X_val: np.ndarray | pd.DataFrame | pl.DataFrame | None = None,
        y_val: np.ndarray | pd.Series | None = None,
        # Training parameters
        epochs: int = 100,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        # Early stopping
        early_stopping: bool = True,
        patience: int = 10,
        min_delta: float = 1e-4,
        # Sample weighting
        sample_weights: np.ndarray | None = None,
        sample_times: pd.Series | None = None,
        pred_times: pd.Series | None = None,
        # Callbacks
        callbacks: list[Callable] | None = None,
        # Checkpointing
        save_best: bool = True,
        checkpoint_dir: str | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """
        Train the model

        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate
            early_stopping: Enable early stopping
            patience: Early stopping patience
            min_delta: Minimum improvement for early stopping
            sample_weights: Sample weights for training
            sample_times: Sample start times (for time series)
            pred_times: Prediction times (for time series)
            callbacks: Custom callback functions
            save_best: Save best model checkpoint
            checkpoint_dir: Directory for checkpoints
            **kwargs: Additional model-specific parameters

        Returns:
            Dictionary with training results
        """
        logger.info("=" * 60)
        logger.info("STARTING MODEL TRAINING")
        logger.info("=" * 60)
        logger.info(f"Model: {self.model.model_name} ({self.model.model_type})")
        logger.info(f"Training samples: {len(X_train)}")
        if X_val is not None:
            logger.info(f"Validation samples: {len(X_val)}")
        logger.info(f"Epochs: {epochs}, Batch size: {batch_size}, LR: {learning_rate}")

        # Start MLflow run
        if self.use_mlflow:
            with mlflow.start_run(
                run_name=f"{self.model.model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            ):
                return self._train_loop(
                    X_train,
                    y_train,
                    X_val,
                    y_val,
                    epochs,
                    batch_size,
                    learning_rate,
                    early_stopping,
                    patience,
                    min_delta,
                    sample_weights,
                    sample_times,
                    pred_times,
                    callbacks,
                    save_best,
                    checkpoint_dir,
                    **kwargs,
                )

    def _train_loop(
        self,
        X_train,
        y_train,
        X_val,
        y_val,
        epochs,
        batch_size,
        learning_rate,
        early_stopping,
        patience,
        min_delta,
        sample_weights,
        sample_times,
        pred_times,
        callbacks,
        save_best,
        checkpoint_dir,
        **kwargs,
    ) -> dict[str, Any]:
        """Internal training loop"""

        # Log hyperparameter to MLflow
        if self.use_mlflow:
            mlflow.log_params(
                {
                    "model_type": self.model.model_type,
                    "epochs": epochs,
                    "batch_size": batch_size,
                    "learning_rate": learning_rate,
                    "early_stopping": early_stopping,
                    "patience": patience,
                    **self.model.hyperparameters,
                }
            )

        # Calculate sample weights if time information provided
        if sample_weights is None and sample_times is not None and pred_times is not None:
            logger.info("Calculating sample weights based on label uniqueness")
            sample_weights = calculate_sample_weights(sample_weights, pred_times)

        # Train the model using its fit method
        training_params = {
            "X_train": X_train,
            "y_train": y_train,
            "X_val": X_val,
            "y_val": y_val,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "early_stopping_patience": patience if early_stopping else None,
            "sample_weights": sample_weights,
            **kwargs,
        }

        # Fit model
        try:
            history = self.model.fit(**training_params)

            # Update training history
            if isinstance(history, dict):
                for key, values in history.items():
                    if key in self.history:
                        self.history[key].extend(values if isinstance(values, list) else [values])

        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise

        # Evaluate final model
        train_metrics = self.evaluate(X_train, y_train, prefix="train_")
        val_metrics = {}
        if X_val is not None and y_val is not None:
            val_metrics = self.evaluate(X_val, y_val, prefix="val_")

        # Log metrics to MLflow
        if self.use_mlflow:
            mlflow.log_metrics({**train_metrics, **val_metrics})

        # Save model
        if save_best:
            checkpoint_path = checkpoint_dir or settings.MODEL_STORAGE_PATH
            self.model.save(checkpoint_path)

            if self.use_mlflow:
                mlflow.log_artifact(
                    str(Path(checkpoint_path) / f"{self.model.model_name}_model.pkl")
                )

        # Build results
        results = {
            "model_name": self.model.model_name,
            "model_type": self.model.model_type,
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "history": self.history,
            "best_epoch": self.best_epoch,
            "training_time": datetime.now().isoformat(),
        }

        logger.success("=" * 60)
        logger.success("TRAINING COMPLETED")
        logger.success("=" * 60)
        logger.success(f"Train MAE: {train_metrics.get('train_mae', 0):.4f}")
        if val_metrics:
            logger.success(f"Val MAE: {val_metrics.get('val_mae', 0):.4f}")

        return results

    def evaluate(
        self, X: np.ndarray | pd.DataFrame, y: np.ndarray | pd.Series, prefix: str = ""
    ) -> dict[str, float]:
        """
        Evaluate model on data

        Args:
            X: Features
            y: Targets
            prefix: Prefix for metric names

        Returns:
            Dictionary of metrics
        """
        try:
            # Get predictions
            y_pred = self.model.predict(X)

            # Handle multi-output
            if len(y_pred.shape) > 1 and y_pred.shape[1] > 1:
                y_pred = y_pred[:, 0]

            y_true = y.values if isinstance(y, pd.Series) else y
            if len(y_true.shape) > 1:
                y_true = y_true[:, 0]

            # Flattern
            y_true = y_true.flatten()
            y_pred = y_pred.flatten()

            # Remove NaN
            mask = ~(np.isnan(y_true) | np.isnan(y_pred))
            y_true = y_true[mask]
            y_pred = y_pred[mask]

            if len(y_true) == 0:
                logger.warning("No valid samples for evaluation")
                return {}

            # Calculate metrics
            metrics = {
                f"{prefix}mae": float(mean_absolute_error(y_true, y_pred)),
                f"{prefix}rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
                f"{prefix}r2": float(r2_score(y_true, y_pred)),
                f"{prefix}mape": float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100),
            }

            # Directional accuracy
            if len(y_true) > 1:
                metrics[f"{prefix}directional_accuracy"] = calculate_directional_accuracy(
                    y_true, y_pred
                )

            return metrics

        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            return {}

    def cross_valiate(
        self,
        X: np.ndarray | pd.DataFrame,
        y: np.ndarray | pd.Series,
        cv: int = 5,
        sample_times: pd.Series | None = None,
        pred_times: pd.Series | None = None,
        use_purged_cv: bool = True,
        **train_kwargs,
    ) -> dict[str, Any]:
        """
        Perform cross-validation

        Args:
            X: Features
            y: Targets
            cv: Number of folds
            sample_times: Sample start time
            pred_times: Prediction times
            use_purged_cv: Use purged K-fold for time series
            **train_kwargs: Training parameters

        Returns:
            Cross-validation results
        """
        logger.info(f"Starting {cv}-fold cross-validation")

        # Choose CV strategy
        if use_purged_cv and sample_times is not None and pred_times is not None:
            logger.info("Using Purged K-Fold for time series")
            cv_splitter = PurgedKFold(n_splits=cv, embargo_pct=0.01)
        else:
            from sklearn.model_selection import KFold

            cv_splitter = KFold(n_splits=cv)

        # Store results
        fold_results = []

        for fold, (train_idx, val_idx) in enumerate(
            cv_splitter.split(X, y, sample_times, pred_times)
        ):
            logger.info(f"Training fold {fold + 1}/{cv}")

            # Split data
            if isinstance(X, pd.DataFrame):
                X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
            else:
                X_train_fold, X_val_fold = X[train_idx], X[val_idx]

            if isinstance(y, pd.Series):
                y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
            else:
                y_train_fold, y_val_fold = y[train_idx], y[val_idx]

            # Train on fold
            fold_history = self.train(
                X_train_fold, y_train_fold, X_val_fold, y_val_fold, **train_kwargs
            )

            fold_results.append(fold_history)

        # Aggregate results
        avg_train_mae = np.mean([r["train_metrics"]["train_mae"] for r in fold_results])
        avg_val_mae = np.mean([r["val_metrics"]["val_mae"] for r in fold_results])
        std_val_mae = np.std([r["val_metrics"]["val_mae"] for r in fold_results])

        cv_results = {
            "cv_strategy": "pruged_kfold" if use_purged_cv else "kfold",
            "n_splits": cv,
            "fold_results": fold_results,
            "avg_train_mae": avg_train_mae,
            "avg_val_mae": avg_val_mae,
            "std_val_mae": std_val_mae,
        }

        logger.success(
            f"Cross-validation complete: Val MAE = {avg_val_mae:.4f} ± {std_val_mae:.4f}"
        )

        return cv_results

    def save_results(self, results: dict[str, Any], filepath: str):
        """
        Save training results to file

        Args:
            results: Results dictionary
            filepath: Output file path
        """
        try:
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)

            with open(filepath, "w") as f:
                json.dump(results, f, indent=2, default=str)

            logger.info(f"Results saved to {filepath}")

        except Exception as e:
            logger.error(f"Failed to save results: {e}")

    def load_results(self, filepath: str) -> dict[str, Any]:
        """
        Load training results from file

        Args:
            filepath: Input file path

        Returns:
            Results dictionary
        """
        try:
            with open(filepath, "r") as f:
                results = json.load(f)

            logger.info(f"Results loaded from {filepath}")
            return results

        except Exception as e:
            logger.error(f"Failed to load results: {e}")
            return {}
