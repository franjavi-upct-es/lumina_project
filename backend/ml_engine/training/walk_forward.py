# backend/ml_engine/training/walk_forward.py
"""
Walk-Forward Optimization for time series models
Implements rolling window training and testing with parameter optimization
"""

import itertools
from concurrent.futures import ProcessPoolExecutor
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from backend.ml_engine.training.trainer import ModelTrainer


class WalkForwardOptimizer:
    """
    Walk-Forward Optimization framework

    Process:
    1. Split data into train/test windows
    2. Optimize hyperparameters on training window
    3. Test on out-of-sample window
    4. Roll forward and repeat
    5. Aggregate results

    Prevents overfitting by validating on truly out-of-sample data
    """

    def __init__(
        self,
        model_class: type,
        train_window_size: int = 252,  # 1 year
        test_window_size: int = 63,  # 3 months
        step_size: int = 21,  # 1 month
        min_train_size: int = 126,  # Minimum 6 months
    ):
        """
        Initialize Walk-Forward Optimizer

        Args:
            model_class: Model class to instantiate
            train_window_size: Training window size (in periods)
            test_window_size: Testing window size (in periods)
            step_size: Step size for rolling windows
            min_train_size: Minimum training data required
        """
        self.model_class = model_class
        self.train_window_size = train_window_size
        self.step_size = step_size
        self.min_train_size = min_train_size

        self.results: list[dict[str, Any]] = []

        logger.info(
            f"Initialized WalkForwardOptimizer: "
            f"train={train_window_size}, test={test_window_size}, step={step_size}"
        )

    def optimize(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        param_grid: dict[str, list[Any]],
        metric: str = "mae",
        n_jobs: int = 1,
        verbose: bool = True,
    ) -> dict[str, Any]:
        """
        Run walk-forward optimization

        Args:
            X: Features DataFrame
            y: Target Series
            param_grid: Dictionary of parameter lists to test
            metric: Metric to optimize ('mae', 'rmse', 'sharpe', etc.)
            n_jobs: Number of parallel jobs
            verbose: Print progress

        Returns:
            Dictionary witn optimization results
        """
        logger.info("=" * 60)
        logger.info("STARTING WALK-FORWARD OPTIMIZATION")
        logger.info("=" * 60)

        n_samples = len(X)

        # Calculate number of windows
        num_windows = 0
        start = 0
        while start + self.train_window_size + self.test_window_size <= n_samples:
            num_windows += 1
            start += self.step_size

        logger.info(f"Total samples: {n_samples}")
        logger.info(f"Number of windows: {num_windows}")
        logger.info(f"Parameter combinations: {self._count_combinations(param_grid)}")

        # Run walk-forward
        window_results = []
        start_idx = 0
        window_num = 0

        while start_idx + self.train_window_size + self.test_window_size <= n_samples:
            window_num += 1

            # Define window indices
            train_start = start_idx
            train_end = start_idx + self.train_window_size
            test_start = train_end
            test_end = test_start + self.test_window_size

            if verbose:
                logger.info("=" * 60)
                logger.info(f"Window {window_num}/{num_windows}")
                logger.info(
                    f"Train: [{train_start}:{train_end}] ({train_end - train_start} samples)"
                )
                logger.info(f"Test:  [{test_start}:{test_end}] ({test_end - test_start} samples)")

            # Split data
            X_train = X.iloc[train_start:train_end]
            y_train = y.iloc[train_start:train_end]
            X_test = X.iloc[test_start:test_end]
            y_test = y.iloc[test_start:test_end]

            # Optimize hyperparameters on training window
            best_params, opt_results = self._optimize_window(
                X_train, y_train, param_grid, metric, n_jobs
            )

            if verbose:
                logger.info(f"Best parameters: {best_params}")

            # Train final model with best parameters
            model = self.model_class(model_name=f"wfo_window_{window_num}", **best_params)

            trainer = ModelTrainer(model, use_mlflow=False)
            trainer.train(
                X_train, y_train, epochs=50, batch_size=32, early_stopping=True, patience=10
            )

            # Test on out-of-sample window
            test_metrics = trainer.evaluate(X_test, y_test, prefix="test_")
            if verbose:
                logger.success(f"Test {metric}: {test_metrics.get(f'tes_{metric}', 0):.4f}")

            # Store results
            window_result = {
                "window": window_num,
                "train_range": (train_start, train_end),
                "test_range": (test_start, test_end),
                "best_params": best_params,
                "test_metrics": test_metrics,
                "optimization_results": opt_results,
            }
            window_results.append(window_result)

            # Move window forward
            start_idx += self.step_size

        # Aggregate results
        aggregate_results = self._aggregate_results(window_results, metric)

        logger.success("=" * 60)
        logger.success("WALK-FORWARD OPTIMIZATION COMPLETED")
        logger.success("=" * 60)
        logger.success(f"Average Test {metric}: {aggregate_results[f'avg_test_{metric}']:.4f}")
        logger.success(f"Std Test {metric}: {aggregate_results[f'std_test_{metric}']:.4f}")

        return {
            "window_results": window_results,
            "aggregate_results": aggregate_results,
            "configuration": {
                "train_window_size": self.train_window_size,
                "test_window_size": self.test_window_size,
                "step_size": self.step_size,
                "num_windows": self.num_windows,
            },
        }

    def _optimize_window(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        param_grid: dict[str, list[Any]],
        metric: str,
        n_jobs: int,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        """
        Optimize hyperparameters for a single window

        Returns:
            Tuple of (best_parmas, optimization_results)
        """
        # Generate all parameter combinations
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        param_combinations = list(itertools.product(*param_values))

        logger.info(f"Testing {len(param_combinations)} parameter combinations")

        # Test each combination
        opt_results = []

        if n_jobs > 1:
            # Parallel optimization
            with ProcessPoolExecutor(max_workers=n_jobs) as executor:
                futures = []
                for params_tuple in param_combinations:
                    params = dict(zip(param_names, params_tuple))
                    future = executor.submit(
                        self._evaluate_params, params, X_train, y_train, metric
                    )
                    futures.append((params, future))

                for params, future in futures:
                    try:
                        score = future.result()
                        opt_reuslts.append({"params": params, "score": score})
                    except Exception as e:
                        logger.error(f"Error evaluating {params}: {e}")

        else:
            # Sequential optimization
            for params_tuple in param_combinations:
                params = dict(zip(param_names, params_tuple))
                try:
                    score = self._evaluate_params(params, X_train, y_train, metric)
                    opt_results.append({"params": params, "score": score})
                except Exception as e:
                    logger.error(f"Error evaluating {params}: {e}")

        # Find best params
        if metric in ["mae", "rmse", "mse"]:
            best_result = min(opt_results, key=lambda x: x["score"])
        else:  # Higher is better (sharpe, r2, etc.)
            best_result = max(opt_results, key=lambda x: x["score"])

    def _evaluate_params(
        self,
        params: dict[str, Any],
        X_train: pd.DataFrame,
        y_train: pd.Series,
        metric: str,
    ) -> float:
        """
        Evaluate a single parameter combination

        Returns:
            Metric score
        """
        # Split into train/validation
        split_idx = int(len(X_train) * 0.8)
        X_tr, X_val = X_train.iloc[:split_idx], X_train.iloc[split_idx:]
        y_tr, y_val = y_train.iloc[:split_idx], y_train.iloc[split_idx:]

        # Train model
        model = self.model_class(model_name="temp_model", **params)

        trainer = ModelTrainer(model, use_mlflow=False)
        trainer.train(
            X_tr,
            y_tr,
            X_val,
            y_val,
            epochs=30,
            batch_size=32,
            early_stopping=True,
            patience=5,
        )

        # Evaluate
        val_metrics = trainer.evaluate(X_val, y_val, prefix="val_")

        return val_metrics.get(f"val_{metric}", float("inf"))

    def _aggregate_results(
        self,
        window_results: list[dict[str, Any]],
        metric: str,
    ) -> dict[str, Any]:
        """
        Aggregate results across all windows

        Returns:
            Dictionary with aggregate statistics
        """
        # Extract test scores
        test_scores = [r["test_metrics"].get(f"test_{metric}", np.nan) for r in window_results]

        # Calculate statistics
        aggregate = {
            f"avg_test_{metric}": np.mean(test_scores),
            f"std_test_{metric}": np.std(test_scores),
            f"min_test_{metric}": np.min(test_scores),
            f"max_test_{metric}": np.max(test_scores),
            f"median_test_{metric}": np.median(test_scores),
        }

        # Parameter stability analysis
        all_params = [r["best_params"] for r in window_results]
        param_stability = {}

        for param_name in all_params[0].keys():
            values = [p[param_name] for p in all_params]
            if isinstance(values[0], (int, float)):
                param_stability[param_name] = {
                    "mean": np.mean(values),
                    "std": np.std(values),
                }
            else:
                # Categorical parameter
                from collections import Counter

                param_stability[param_name] = {
                    "mode": Counter(values).most_common(1)[0][0],
                    "unique_values": len(set(values)),
                }

        aggregate["parameter_stability"] = param_stability

        return aggregate

    def _count_combinations(self, param_grid: dict[str, list[Any]]) -> int:
        """Count toal parameter combinations"""
        count = 1
        for values in param_grid.values():
            count *= len(values)
        return count

    def plot_results(self, results: dict[str, Any], metric: str = "mae"):
        """
        Plot walk-forward results

        Args:
            results: Results from optimize()
            metric: Metric to plot
        """
        try:
            import matplotlib.pyplot as plt

            window_results = results["window_results"]
            windows = [r["window"] for r in window_results]
            scores = [r["test_metrics"].get(f"test_{metric}", np.nan) for r in window_results]

            plt.figure(figsize=(12, 6))
            plt.plot(windows, scores, marker="o", linewidth=2, markersize=8)
            plt.axhline(
                y=np.mean(scores),
                color="r",
                linestyle="--",
                label=f"Average: {np.mean(scores):.4f}",
            )
            plt.xlabel("Window")
            plt.ylabel(f"Test {metric.upper()}")
            plt.title("Walk-Forward Optimization Results")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()

            logger.info("Results plotted successfully")

        except ImportError:
            logger.warning("matplotlib not available for plotting")
        except Exception as e:
            logger.error(f"Error plotting results: {e}")
