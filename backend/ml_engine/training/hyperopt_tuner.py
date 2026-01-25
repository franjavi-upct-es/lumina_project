# backend/ml_engine/training/hyperopt_tuner.py
"""
Hyperparameter optimization using Hyperopt
Supports Bayesian optimization for model tuning
"""

import time
from typing import Any

import numpy as np
import pandas as pd
from hyperopt import STATUS_OK, Trials, fmin, hp, space_eval, tpe
from loguru import logger

from backend.ml_engine.models.base_model import BaseModel


class HyperparameterTuner:
    """
    Hyperparameter tuner using Bayesian optimization (Hyperopt)

    Features:
    - Tree-structure Parzen Estimator (TPE)
    - Parallel evaluations support
    - Early stopping
    - Result stopping
    - Best parameter selection
    """

    def __init__(
        self,
        model_class: type,
        param_space: dict[str, Any],
        metric: str = "val_loss",
        maximize: bool = False,
    ):
        """
        Initialize tuner

        Args:
            model_class: Model class to optimize
            param_space: Hyperparameter search space (hyperopt format)
            metric: Metric to optimize
            maximize: Whether to maximize metric (False = minimize)
        """
        self.model_class = model_class
        self.param_space = param_space
        self.metric = metric
        self.maximize = maximize

        self.trials = Trials()
        self.best_params: dict[str, Any] | None = None
        self.best_score: float | None = None
        self.results_history: list[dict[str, Any]] = []

        logger.info(f"Initialized HyperparameterTuner for {model_class.__name__}")

    def optimize(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        max_evals: int = 50,
        timeout: int | None = None,
        **model_kwargs,
    ) -> dict[str, Any]:
        """
        Run hyperparameter optimization

        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            max_evals: Maximum number of evaluations
            timeout: Maximum time in seconds
            **model_kwargs: Additional model arguments

        Returns:
            Dictionary with best parameters and results
        """
        logger.info(f"Starting hyperparameter optimization ({max_evals} evals)")

        start_time = time.time()

        # Define objective function
        def objective(params):
            # Check timeout
            if timeout and (time.time() - start_time) > timeout:
                logger.warning("Timeout reached, stopping optimization")
                return {"loss": float("inf"), "status": STATUS_OK}

            try:
                # Train model with these parameters
                result = self._evaluate_params(
                    params, X_train, y_train, X_val, y_val, **model_kwargs
                )

                # Store result
                self.results_history.append(
                    {
                        "params": params.copy(),
                        "score": result["score"],
                        "metrics": result["metrics"],
                    }
                )

                # Return loss (negated if maximizing)
                loss = -result["score"] if self.maximize else result["score"]

                return {"loss": loss, "status": STATUS_OK, "metrics": result["metrics"]}

            except Exception as e:
                logger.error(f"Error evaluating params: {e}")
                return {"loss": float("inf"), "status": STATUS_OK}

        # Run optimization
        best = fmin(
            fn=objective,
            space=self.param_space,
            algo=tpe.suggest,
            max_evals=max_evals,
            trials=self.trials,
            verbose=1,
        )

        # Get best parameters
        self.best_params = space_eval(self.param_space, best)

        # Get best score
        best_trial = min(self.trials.trials, key=lambda x: x["result"]["loss"])
        self.best_score = (
            -best_trial["result"]["loss"] if self.maximize else best_trial["result"]["loss"]
        )

        elapsed_time = time.time() - start_time

        logger.success(
            f"Optimization complete in {elapsed_time:.2f}s - "
            f"Best {self.metric}: {self.best_score:.4f}"
        )

        return {
            "best_params": self.best_params,
            "best_score": self.best_score,
            "num_trials": len(self.trials.trials),
            "elapsed_time": elapsed_time,
            "results_history": self.results_history,
        }

    def _evaluate_params(
        self,
        params: dict[str, Any],
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        **model_kwargs,
    ) -> dict[str, Any]:
        """
        Evaluate a set of parameters

        Args:
            params: Hyperparameters to evaluate
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            **model_kwargs: Additional model arguments

        Returns:
            Dictionary with score and metrics
        """
        # Create model with these parameters
        model.build(input_shape=X_train.shape)

        # Train model
        history = model.fit(X_train, y_train, X_val, y_val, verbose=0, **model_kwargs)

        # Extract metric
        if self.metric in history.get("val_metrics", {}):
            score = history["val_metrics"][self.metric]
        elif self.metric in history:
            score = (
                history[self.metric][-1]
                if isinstance(history[self.metric], list)
                else history[self.metric]
            )
        else:
            # Fallback: evaluate on validation set
            metrics = model.evaluate(X_val, y_val)
            score = metrics.get(self.metric, metrics.get("mae", float("inf")))

        return {"score": score, "metrics": history.get("val_metrics", {})}

    def get_best_model(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> BaseModel:
        """
        Train and return model with best parameters

        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            **model_kwargs: Additional model arguments

        Returns:
            Trained model with best parameters
        """
        if self.best_params is None:
            raise ValueError("Must run optimize() first")

        logger.info("Training final model with best parameters")

        # Create model
        model = self.model_class(model_name="best_model", hyperparameters=self.best_params)

        # Build
        model.build(input_shape=X_train.shape)

        # Train
        model.fit(X_train, y_train, X_val, y_val, **model_kwargs)

        logger.success("Best model trained")

        return model

    def plot_optimization_history(self):
        """
        Plot optimization history

        Returns:
            Plotly figure
        """
        try:
            import plotly.graph_objects as go

            # Extract scores
            scores = [r["score"] for r in self.results_history]

            # Running best
            if self.maximize:
                running_best = np.maximum.accumulate(scores)
            else:
                running_best = np.minimum.accumulate(scores)

            fig = go.Figure()

            # Add scatter of all trials
            fig.add_trace(
                go.Scatter(
                    y=scores, mode="markets", name="Trial Score", marker=dict(size=8, opacity=0.6)
                )
            )

            # Add running best
            fig.add_trace(
                go.Scatter(
                    y=running_best,
                    mode="lines",
                    name="Running Best",
                    line=dict(color="red", width=2),
                )
            )

            fig.update_layout(
                title=f"Hyperparameter Optimization: {self.metric}",
                xaxis_title="Trial",
                yaxis_title=self.metric,
                hovermode="x unified",
            )

            return fig

        except ImportError:
            logger.warning("Plotly not available, cannot plot history")
            return None

    def get_param_importance(self) -> dict[str, float]:
        """
        Analyze parameter importance

        Returns:
            Dictionary with parameter importance scores
        """
        if not self.results_history:
            return {}

        # Convert to DataFrame
        param_list = []
        scores = []

        for result in self.results_history:
            param_list.append(result["params"])
            scores.append(result["score"])

        df_params = pd.DataFrame(param_list)

        # Calculate correlation with score
        importance = {}
        for col in df_params.columns:
            if df_params[col].dtype in [np.float64, np.int64]:
                corr = abs(df_params[col].corr(pd.Series(scores)))
                importance[col] = float(corr)

        # Sort by importance
        importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))

        return importance

    def save_results(self, filepath: str):
        """
        Save optimization results

        Args:
            filepath: Path to save results
        """
        import json

        results = {
            "best_params": self.best_params,
            "best_score": self.best_score,
            "metric": self.metric,
            "maximize": self.maximize,
            "num_trials": len(self.trials.trials),
            "results_history": self.results_history,
            "param_importance": self.get_param_importance(),
        }

        with open(filepath, "w") as f:
            json.dump(results, f, indent=2, default=str)

        logger.info(f"Results saved to {filepath}")

    def load_results(self, filepath: str):
        """
        Load optimization results

        Args:
            filepath: Path to load results from
        """
        import json

        with open(filepath, "r") as f:
            results = json.load(f)

        self.best_params = results["best_params"]
        self.best_score = results["best_score"]
        self.results_history = results["results_history"]


# Predefined search spaces for common models
SEARCH_SPACES = {
    "xgboost": {
        "n_estimators": hp.quniform("n_estimators", 50, 500, 50),
        "max_depth": hp.quniform("max_depth", 3, 10, 1),
        "learning_rate": hp.loguniform("learning_rate", np.log(0.001), np.log(0.3)),
        "min_child_weight": hp.quniform("min_child_weight", 1, 10, 1),
        "subsample": hp.uniform("subsample", 0.6, 1.0),
        "colsample_bytree": hp.uniform("colsample_bytree", 0.6, 1.0),
        "gamma": hp.uniform("gamma", 0, 0.5),
        "reg_alpha": hp.loguniform("reg_alpha", np.log(0.001), np.log(10)),
        "reg_lambda": hp.loguniform("reg_lambda", np.log(0.1), np.log(10)),
    },
    "lstm": {
        "hidden_dim": hp.quniform("hidden_dim", 64, 256, 32),
        "num_layers": hp.quniform("num_layers", 2, 5, 1),
        "dropout": hp.uniform("dropout", 0.1, 0.5),
        "learning_rate": hp.loguniform("learning_rate", np.log(0.0001), np.log(0.01)),
        "batch_size": hp.quniform("batch_size", 16, 64, 16),
    },
    "transformer": {
        "d_model": hp.quniform("d_model", 64, 256, 32),
        "nhead": hp.choice("nhead", [4, 8, 16]),
        "num_encoder_layers": hp.quniform("num_encoder_layers", 2, 6, 1),
        "dim_feedforward": hp.quniform("dim_feedforward", 256, 1024, 128),
        "dropout": hp.uniform("dropout", 0.1, 0.4),
        "learning_rate": hp.loguniform("learning_rate", np.log(0.0001), np.log(0.01)),
    },
}


def get_search_space(model_type: str) -> dict[str, Any]:
    """
    Get predefined search space for a model type

    Args:
        model_type: Type of model (xgboost, lstm, transformer)

    Returns:
        Hyperopt search space
    """
    if model_type not in SEARCH_SPACES:
        raise ValueError(
            f"Unknown model type: {model.type}. Available: {list(SEARCH_SPACES.keys())}"
        )

    return SEARCH_SPACES[model_type]


def quick_tune(
    model_class: type,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    model_type: str = "xgboost",
    max_evals: int = 30,
    metric: str = "val_mae",
    **model_kwargs,
) -> BaseModel:
    """
    Quick hyperparameter tuning with sensible defaults

    Args:
        model_class: Model class
        X_train: Training features
        y_train: Training targets
        X_val: Validation features
        y_val: Validation targets
        model_type: Type of model for search space
        max_evals: Maximum evaluations
        metric: Metric to optimize
        **model_kwargs: Additional model arguments

    Returns:
        Best trained model
    """
    logger.info(f"Quick tuning for {model_type} (max_evals={max_evals})")

    # Get search space
    search_space = get_search_space(model_type)

    # Create tuner
    tuner = HyperparameterTuner(
        model_class=model_class,
        param_space=search_space,
        metric=metric,
        maximize=False,  # Minimize loss metrics
    )

    # Optimize
    tuner.optimize(X_train, y_train, X_val, y_val, max_evals=max_evals, **model_kwargs)

    # Get best model
    best_model = tuner.get_best_model(X_train, y_train, X_val, y_val, **model_kwargs)

    logger.success(f"Best {metric}: {tuner.best_score:.4f}")
    logger.info(f"Best params: {tuner.best_params}")

    return best_model
