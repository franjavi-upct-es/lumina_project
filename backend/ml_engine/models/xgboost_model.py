# backend/ml_engine/models/xgboost_model.py
"""
XGBoost model for financial time series prediction
Optimized for tabular feature data with built-in feature importance
"""

from typing import Dict, Any, Optional, Tuple, Union
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
import pickle
from loguru import logger
import mlflow

from backend.ml_engine.models.base_model import BaseModel, ModelMetadata


class XGBoostFinancialModel(BaseModel):
    """
    XGBoost model optimized for financial prediction

    Features:
    - Time series cross-validation
    - Feature importance analysis
    - Early stopping
    - Multi-step prediction
    - Uncertainty estimation via quantile regression
    """

    def __init__(
        self,
        model_name: str,
        hyperparameters: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize XGBoost model

        Args:
            model_name: Unique name for model
            hyperparameters: XGBoost hyperparameters
        """
        # Default hyperparameters optimized for financial data
        default_params = {
            "objective": "reg:squarederror",
            "max_depth": 6,
            "learning_rate": 0.05,
            "n_estimators": 500,
            "min_child_weight": 3,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "gamma": 0.1,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "random_state": 42,
            "n_jobs": -1,
        }

        if hyperparameters:
            default_params.update(hyperparameters)

        super().__init__(
            model_name=model_name, model_type="xgboost", hyperparameters=default_params
        )

        self.model: Optional[xgb.XGBRegressor] = None
        self.scaler = StandardScaler()
        self.prediction_horizon = self.hyperparameters.get("prediction_horizon", 1)

        # For multi-step prediction
        self.models_per_horizon: Dict[int, xgb.XGBRegressor] = {}

    def build(self, input_shape: Tuple[int, ...], **kwargs) -> xgb.XGBRegressor:
        """
        Build XGBoost model

        Args:
            input_shape: Shape of input features (not used for XGBoost but kept for interface)
            **kwargs: Additional parameters

        Returns:
            XGBoost model
        """
        logger.info(f"Building XGBoost model with params: {self.hyperparameters}")

        self.model = xgb.XGBRegressor(**self.hyperparameters)

        logger.success("XGBoost model built successfully")
        return self.model

    def fit(
        self,
        X_train: Union[np.ndarray, pd.DataFrame],
        y_train: Union[np.ndarray, pd.Series],
        X_val: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        y_val: Optional[Union[np.ndarray, pd.Series]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Train XGBoost model

        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
            **kwargs: Additional training parameters

        Returns:
            Training history and metrics
        """
        logger.info(f"Training XGBoost model on {len(X_train)} samples")

        # Store feature names if DataFrame
        if isinstance(X_train, pd.DataFrame):
            self.feature_names = X_train.columns.tolist()
        else:
            self.feature_names = [f"feature_{i}" for i in range(X_train.shape[1])]

        # Convert to numpy if needed
        X_train_np = X_train.values if isinstance(X_train, pd.DataFrame) else X_train
        y_train_np = y_train.values if isinstance(y_train, pd.Series) else y_train

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train_np)

        # Build model if not already built
        if self.model is None:
            self.build(input_shape=X_train_scaled.shape)

        # Prepare validation data if provided
        eval_set = None
        if X_val is not None and y_val is not None:
            X_val_np = X_val.values if isinstance(X_val, pd.DataFrame) else X_val
            y_val_np = y_val.values if isinstance(y_val, pd.Series) else y_val
            X_val_scaled = self.scaler.transform(X_val_np)
            eval_set = [(X_train_scaled, y_train_np), (X_val_scaled, y_val_np)]
        else:
            eval_set = [(X_train_scaled, y_train_np)]

        # Train
        verbose = kwargs.get("verbose", 50)

        self.model.fit(
            X_train_scaled,
            y_train_np,
            eval_set=eval_set,
            verbose=verbose,
        )

        # Get training results
        evals_result = self.model.evals_result()

        # Calculate metrics
        train_pred = self.model.predict(X_train_scaled)
        train_metrics = self.compute_metrics(y_train_np, train_pred, prefix="train_")

        val_metrics = {}
        if X_val is not None:
            val_pred = self.model.predict(X_val_scaled)
            val_metrics = self.compute_metrics(y_val_np, val_pred, prefix="val_")

        # Store meta_data
        self.meta_data = ModelMetadata(
            model_id=self.model_name,
            model_name=self.model_name,
            model_type=self.model_type,
            version="1.0",
            ticker=kwargs.get("ticker", "UNKNOWN"),
            trained_on=pd.Timestamp.now(),
            training_samples=len(X_train),
            validation_samples=len(X_val) if X_val is not None else 0,
            hyperparameters=self.hyperparameters,
            feature_names=self.feature_names,
            num_features=len(self.feature_names),
            prediction_horizon=self.prediction_horizon,
            train_metrics=train_metrics,
            validation_metrics=val_metrics,
        )

        self.is_trained = True

        # Store training history
        self.training_history = {
            "train_loss": evals_result["validation_0"]["rmse"],
        }
        if "validation_1" in evals_result:
            self.training_history["val_loss"] = evals_result["validation_1"]["rmse"]

        # Calculate final metrics
        train_predictions = self.model.predict(X_train_scaled)
        train_mae = float(np.mean(np.abs(y_train_np - train_predictions)))

        logger.success("Training complete!")
        logger.info(f"Train MAE: {train_mae:.4f}")

        # Prepare result
        history = {
            "train_mae": train_mae,
        }

        # Only add best_iteration if it exists (when early stopping is used)
        try:
            history["best_iteration"] = self.model.best_iteration
        except AttributeError:
            # No early stopping, use n_estimators
            history["best_iteration"] = self.hyperparameters.get("n_estimators", 0)

        # Validation metrics if provided
        if X_val is not None and y_val is not None:
            val_predictions = self.model.predict(X_val_scaled)
            val_mae = float(np.mean(np.abs(y_val_np - val_predictions)))
            logger.info(f"Val MAE: {val_mae:.4f}")
            history["val_mae"] = val_mae

        # Save metrics
        self.metrics = history
        self.status = "trained"

        return history

    def fit_multistep(
        self,
        X_train: Union[np.ndarray, pd.DataFrame],
        y_train: Union[np.ndarray, pd.DataFrame],
        X_val: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        y_val: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Train separate models for each prediction horizon

        Args:
            X_train: Training features
            y_train: Training targets (multi-step: shape [samples, horizons])
            X_val: Validation features
            y_val: Validation targets
            **kwargs: Additional parameters

        Returns:
            Training results for all horizons
        """
        logger.info(f"Training multi-step XGBoost models")

        # Ensure y is 2D
        if isinstance(y_train, pd.Series):
            y_train = y_train.values.reshape(-1, 1)
        elif isinstance(y_train, pd.DataFrame):
            y_train = y_train.values

        if y_train.ndim == 1:
            y_train = y_train.reshape(-1, 1)

        num_horizons = y_train.shape[1]
        logger.info(f"Training {num_horizons} models for {num_horizons}-step prediction")

        results = {}

        for horizon in range(num_horizons):
            logger.info(f"Training model for horizon {horizon + 1}/{num_horizons}")

            # Create separate model for this horizon
            model = xgb.XGBRegressor(**self.hyperparameters)

            # Prepare target for this horizon
            y_train_h = y_train[:, horizon]
            y_val_h = y_val[:, horizon] if y_val is not None else None

            # Train
            result = self.fit(
                X_train=X_train,
                y_train=y_train_h,
                X_val=X_val,
                y_val=y_val_h,
                **kwargs,
            )

            # Store model
            self.models_per_horizon[horizon] = self.model
            results[f"horizon_{horizon + 1}"] = result

        self.is_trained = True
        logger.success(f"Multi-step training complete for {num_horizons} horizons")

        return results

    def predict(
        self, X: Union[np.ndarray, pd.DataFrame], **kwargs
    ) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        """
        Make predictions

        Args:
            X: Input features
            **kwargs: Additional parameters
                - return_uncertainty: Whether to return prediction intervals

        Returns:
            Predictions (with uncertainty if requested)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")

        # Convert to numpy and scale
        X_np = X.values if isinstance(X, pd.DataFrame) else X
        X_scaled = self.scaler.transform(X_np)

        # Multi-step prediction
        if self.models_per_horizon:
            predictions = []
            for horizon in range(len(self.models_per_horizon)):
                model = self.models_per_horizon[horizon]
                pred = model.predict(X_scaled)
                predictions.append(pred)

            predictions = np.column_stack(predictions)
        else:
            # Single-step prediction
            predictions = self.model.predict(X_scaled)

        # Normalize dtype for downstream consistency
        if isinstance(predictions, np.ndarray):
            predictions = predictions.astype(np.float64, copy=False)

        # Return with uncertainty if requested
        if kwargs.get("return_uncertainty", False):
            # Estimate uncertainty using prediction variance
            # (simplified - could use quantile regression for better estimates)
            uncertainty = (
                np.std(predictions, axis=1, keepdims=True)
                if predictions.ndim > 1
                else np.ones_like(predictions) * 0.05
            )
            uncertainty = uncertainty.astype(np.float64, copy=False)

            return {
                "predictions": predictions,
                "uncertainty": uncertainty,
                "confidence_lower": predictions - 1.96 * uncertainty,
                "confidence_upper": predictions + 1.96 * uncertainty,
            }

        return predictions

    def predict_with_uncertainty(
        self, X: Union[np.ndarray, pd.DataFrame], **kwargs
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions with uncertainty estimates

        Args:
            X: Input features
            **kwargs: Additional parameters

        Returns:
            Tuple of (predictions, uncertainties)
        """
        result = self.predict(X, return_uncertainty=True)

        if isinstance(result, dict):
            return result["predictions"], result["uncertainty"]
        else:
            # Estimate uncertainty
            uncertainty = np.ones_like(result) * 0.05  # 5% default uncertainty
            return result, uncertainty

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
            **kwargs: Additional parameters

        Returns:
            Evaluation metrics
        """
        logger.info("Evaluating model...")

        predictions = self.predict(X)

        # Compute metrics
        metrics = self.compute_metrics(y, predictions)

        logger.info(f"Evaluation metrics: {metrics}")

        return metrics

    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """
        Get feature importance scores

        Returns:
            Dictionary mapping feature names to importance scores
        """
        if not self.is_trained or self.model is None:
            logger.warning("Model not trained, cannot get feature importance")
            return None

        # Get importance scores
        importance_scores = self.model.feature_importances_

        # Create dictionary
        feature_importance = {
            name: float(score) for name, score in zip(self.feature_names, importance_scores)
        }

        # Sort by importance
        feature_importance = dict(
            sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        )

        logger.info(f"Top 5 features: {list(feature_importance.keys())[:5]}")

        return feature_importance

    def _save_model(self, path: Path):
        """
        Save model-specific components

        Args:
            path: Path to save model
        """
        model_data = {
            "model": self.model,
            "scaler": self.scaler,
            "models_per_horizon": self.models_per_horizon,
            "prediction_horizon": self.prediction_horizon,
        }

        with open(path, "wb") as f:
            pickle.dump(model_data, f)

        logger.info(f"Model saved to {path}")

    def _load_model(self, path: Path):
        """
        Load model-specific components

        Args:
            path: Path to load model from
        """
        with open(path, "rb") as f:
            model_data = pickle.load(f)

        self.model = model_data["model"]
        self.scaler = model_data["scaler"]
        self.models_per_horizon = model_data.get("models_per_horizon", {})
        self.prediction_horizon = model_data.get("prediction_horizon", 1)

        logger.info(f"Model loaded from {path}")

    def _log_model_to_mlflow(self):
        """
        Log model to MLflow
        """
        # Log model
        mlflow.xgboost.log_model(self.model, "model")

        # Log feature importance
        if self.feature_names:
            importance = self.get_feature_importance()
            if importance:
                # Log as JSON
                mlflow.log_dict(importance, "feature_importance.json")

                # Log top 20 as params
                for i, (feat, score) in enumerate(list(importance.items())[:20]):
                    mlflow.log_param(f"top_feature_{i + 1}", feat)
                    mlflow.log_metric(f"importance_{i + 1}", score)

        logger.info("Model logged to MLflow")

    def cross_validate(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        n_splits: int = 5,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Perform time series cross-validation

        Args:
            X: Features
            y: Targets
            n_splits: Number of splits
            **kwargs: Additional parameters

        Returns:
            Cross-validation results
        """
        logger.info(f"Performing {n_splits}-fold time series cross-validation")

        # Convert to numpy
        X_np = X.values if isinstance(X, pd.DataFrame) else X
        y_np = y.values if isinstance(y, pd.Series) else y

        # Time series split
        tscv = TimeSeriesSplit(n_splits=n_splits)

        cv_results = {
            "train_mae": [],
            "val_mae": [],
            "train_rmse": [],
            "val_rmse": [],
            "train_r2": [],
            "val_r2": [],
        }

        for fold, (train_idx, val_idx) in enumerate(tscv.split(X_np)):
            logger.info(f"Fold {fold + 1}/{n_splits}")

            X_train_fold = X_np[train_idx]
            y_train_fold = y_np[train_idx]
            X_val_fold = X_np[val_idx]
            y_val_fold = y_np[val_idx]

            # Train model
            self.fit(
                X_train=X_train_fold,
                y_train=y_train_fold,
                X_val=X_val_fold,
                y_val=y_val_fold,
                verbose=0,
            )

            # Evaluate
            train_pred = self.predict(X_train_fold)
            val_pred = self.predict(X_val_fold)

            train_metrics = self.compute_metrics(y_train_fold, train_pred)
            val_metrics = self.compute_metrics(y_val_fold, val_pred)

            # Store results
            cv_results["train_mae"].append(train_metrics["mae"])
            cv_results["val_mae"].append(val_metrics["mae"])
            cv_results["train_rmse"].append(train_metrics["rmse"])
            cv_results["val_rmse"].append(val_metrics["rmse"])
            cv_results["train_r2"].append(train_metrics["r2"])
            cv_results["val_r2"].append(val_metrics["r2"])

        # Calculate averages
        avg_results = {
            "avg_train_mae": np.mean(cv_results["train_mae"]),
            "avg_val_mae": np.mean(cv_results["val_mae"]),
            "avg_train_rmse": np.mean(cv_results["train_rmse"]),
            "avg_val_rmse": np.mean(cv_results["val_rmse"]),
            "avg_train_r2": np.mean(cv_results["train_r2"]),
            "avg_val_r2": np.mean(cv_results["val_r2"]),
            "std_val_mae": np.std(cv_results["val_mae"]),
        }

        logger.success(
            f"Cross-validation complete: Val MAE = {avg_results['avg_val_mae']:.4f} "
            f"± {avg_results['std_val_mae']:.4f}"
        )

        return {**cv_results, **avg_results}
