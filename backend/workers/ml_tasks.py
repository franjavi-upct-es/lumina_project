# backend/workers/ml_tasks.py
"""
Celery tasks for machine learning model training and inference.

Fixed:
- Each async operation uses run_async() with fresh engine (no stale event loop)
- MLflow tracking URI read from env (resolves to http://mlflow:5000 in Docker)
- Models saved to DB Model table after training
- Heavy deps (torch, mlflow, shap) lazy-imported inside tasks
- Proper error handling and retries
"""

import os
from datetime import date, datetime, time, timedelta
from datetime import date, datetime, time, timedelta
from typing import Any
from uuid import uuid4

import numpy as np
import pandas as pd
import pandas as pd
from celery import shared_task
from loguru import logger

from backend.config.settings import get_settings
from backend.data_engine.collectors.yfinance_collector import YFinanceCollector
from backend.data_engine.transformers.feature_engineering import FeatureEngineer
from backend.db.models import run_async, save_model_to_db

settings = get_settings()

# ---------------------------------------------------------------------------
#  CUDA / Device helper
# ---------------------------------------------------------------------------


def _get_device():
    """
    Safely resolve the compute device.
    In forked Celery workers CUDA cannot be re-initialised, so we fall back
    to CPU and log a warning instead of crashing.
    """
    import torch

    if not torch.cuda.is_available():
        logger.info("CUDA not available – using CPU")
        return torch.device("cpu")

    try:
        # Attempt a tiny allocation to verify CUDA works in this process
        torch.tensor([0.0], device="cuda")
        device = torch.device("cuda")
        logger.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
        return device
    except RuntimeError as e:
        if "CUDA" in str(e) or "fork" in str(e):
            logger.warning(
                "CUDA unavailable in forked subprocess – falling back to CPU. "
                "Start the worker with --pool=solo or --pool=threads to use GPU."
            )
            return torch.device("cpu")
        raise


# ---------------------------------------------------------------------------
#  MLflow helper
# ---------------------------------------------------------------------------


def _setup_mlflow(experiment_name: str | None = None):
    """
    Configure MLflow tracking URI and experiment.
    Must be called inside each worker task (not at module-level)
    because the tracking URI differs between local and Docker.
    """
    import mlflow

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", settings.MLFLOW_TRACKING_URI)
    mlflow.set_tracking_uri(tracking_uri)
    logger.info(f"MLflow tracking URI: {tracking_uri}")

    exp_name = experiment_name or os.getenv("MLFLOW_EXPERIMENT_NAME", "lumina_quant")
    experiment = mlflow.get_experiment_by_name(exp_name)  # type: ignore
    if experiment is None:  # type: ignore
        mlflow.create_experiment(exp_name)  # type: ignore
        logger.info(f"Created MLflow experiment: {exp_name}")  # type: ignore
    mlflow.set_experiment(exp_name)

    return mlflow


# ---------------------------------------------------------------------------
#  Data collection helper (sync, uses run_async internally)
# ---------------------------------------------------------------------------


def _collect_data(ticker: str, start_date, end_date):
    """Collect data using run_async (Celery-safe)."""
    collector = YFinanceCollector()
    return run_async(
        collector.collect_with_retry(ticker=ticker, start_date=start_date, end_date=end_date)
    )


def _resolve_market_time_column(data) -> str | None:
    for candidate in ("time", "date", "datetime"):
        if candidate in data.columns:
            return candidate
    return None


def _sort_market_data(data):
    time_column = _resolve_market_time_column(data)
    if time_column is None:
        return data
    return data.sort(time_column)


def _coerce_market_timestamp(value) -> datetime:
    if hasattr(value, "to_pydatetime"):
        value = value.to_pydatetime()
    if isinstance(value, datetime):
        return value
    if isinstance(value, date):
        return datetime.combine(value, time.min)
    raise TypeError(f"Unsupported timestamp type: {type(value)!r}")


def _get_last_market_timestamp(data) -> datetime:
    time_column = _resolve_market_time_column(data)
    if time_column is None:
        raise ValueError("Market data is missing a time/date column")
    return _coerce_market_timestamp(data[time_column].tail(1).item())


def _select_xgboost_feature_columns(enriched_pd: pd.DataFrame) -> list[str]:
    excluded_columns = {
        "time",
        "date",
        "datetime",
        "ticker",
        "source",
        "collected_at",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "adjusted_close",
        "dividends",
        "stock_splits",
    }

    feature_columns = []
    for column in enriched_pd.columns:
        if column in excluded_columns:
            continue
        series = enriched_pd[column]
        if pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
            feature_columns.append(column)
    return feature_columns


def _resolve_market_time_column(data) -> str | None:
    for candidate in ("time", "date", "datetime"):
        if candidate in data.columns:
            return candidate
    return None


def _sort_market_data(data):
    time_column = _resolve_market_time_column(data)
    if time_column is None:
        return data
    return data.sort(time_column)


def _coerce_market_timestamp(value) -> datetime:
    if hasattr(value, "to_pydatetime"):
        value = value.to_pydatetime()
    if isinstance(value, datetime):
        return value
    if isinstance(value, date):
        return datetime.combine(value, time.min)
    raise TypeError(f"Unsupported timestamp type: {type(value)!r}")


def _get_last_market_timestamp(data) -> datetime:
    time_column = _resolve_market_time_column(data)
    if time_column is None:
        raise ValueError("Market data is missing a time/date column")
    return _coerce_market_timestamp(data[time_column].tail(1).item())


def _select_xgboost_feature_columns(enriched_pd: pd.DataFrame) -> list[str]:
    excluded_columns = {
        "time",
        "date",
        "datetime",
        "ticker",
        "source",
        "collected_at",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "adjusted_close",
        "dividends",
        "stock_splits",
    }

    feature_columns = []
    for column in enriched_pd.columns:
        if column in excluded_columns:
            continue
        series = enriched_pd[column]
        if pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
            feature_columns.append(column)
    return feature_columns


# ---------------------------------------------------------------------------
#  LSTM training task
# ---------------------------------------------------------------------------


@shared_task(
    bind=True,
    name="workers.ml_tasks.train_model_task",
    max_retries=2,
    default_retry_delay=120,
)
def train_model_task(self, job_id: str, ticker: str, model_type: str, hyperparams: dict[str, Any]):
    """
    Train a machine learning model asynchronously.
    """
    try:
        import torch
        from torch.utils.data import DataLoader, Subset
        from torch.utils.data import DataLoader, Subset

        from backend.ml_engine.models.lstm_advanced import (
            AdvancedLSTM,
            LSTMTrainer,
            TimeSeriesDataset,  # type: ignore
        )

        logger.info(f"Starting training job {job_id} for {ticker} ({model_type})")

        # ── Resolve device safely ──────────────────────────────────────────
        device = _get_device()

        self.update_state(state="PROGRESS", meta={"step": "data_collection", "progress": 0})

        # 1. Collect data ---------------------------------------------------
        start_date = hyperparams.get("start_date")
        end_date = hyperparams.get("end_date")

        if not start_date:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=365 * 3)

        data = _collect_data(ticker, start_date, end_date)

        if data is None or data.height == 0:
            raise ValueError(f"No data collected for {ticker}")

        data = _sort_market_data(data)
        data = _sort_market_data(data)
        logger.info(f"Collected {data.height} data points for {ticker}")

        self.update_state(state="PROGRESS", meta={"step": "feature_engineering", "progress": 20})

        # 2. Feature engineering --------------------------------------------
        fe = FeatureEngineer()
        enriched_data = fe.create_all_features(data, add_lags=True, add_rolling=True)

        max_features = hyperparams.get("max_features", 50)
        feature_columns = [
            column
            for column in fe.get_all_feature_names()[:max_features]
            if column in enriched_data.columns
        ]
        if not feature_columns:
            raise ValueError("No valid feature columns generated for LSTM training")
        feature_columns = [
            column
            for column in fe.get_all_feature_names()[:max_features]
            if column in enriched_data.columns
        ]
        if not feature_columns:
            raise ValueError("No valid feature columns generated for LSTM training")

        logger.info(f"Engineered {len(feature_columns)} features")

        self.update_state(state="PROGRESS", meta={"step": "dataset_preparation", "progress": 40})

        # 3. Prepare dataset ------------------------------------------------
        sequence_length = hyperparams.get("sequence_length", 60)
        prediction_horizon = hyperparams.get("prediction_horizon", 5)
        target_mode = "relative_returns"
        feature_scaler = TimeSeriesDataset.build_feature_scaler(enriched_data, feature_columns)
        target_mode = "relative_returns"
        feature_scaler = TimeSeriesDataset.build_feature_scaler(enriched_data, feature_columns)

        dataset = TimeSeriesDataset(
            data=enriched_data,
            feature_columns=feature_columns,
            sequence_length=sequence_length,
            prediction_horizon=prediction_horizon,
            stride=1,
            feature_scaler=feature_scaler,
            target_mode=target_mode,
        )

        if len(dataset) < 2:
            raise ValueError(
                f"Not enough sequences for LSTM training. Need at least 2, got {len(dataset)}."
            )
            feature_scaler=feature_scaler,
            target_mode=target_mode,
        )

        if len(dataset) < 2:
            raise ValueError(
                f"Not enough sequences for LSTM training. Need at least 2, got {len(dataset)}."
            )

        train_size = int(0.8 * len(dataset))
        train_size = min(max(train_size, 1), len(dataset) - 1)
        train_size = min(max(train_size, 1), len(dataset) - 1)
        val_size = len(dataset) - train_size
        train_dataset = Subset(dataset, range(0, train_size))
        val_dataset = Subset(dataset, range(train_size, len(dataset)))
        train_dataset = Subset(dataset, range(0, train_size))
        val_dataset = Subset(dataset, range(train_size, len(dataset)))

        batch_size = hyperparams.get("batch_size", 32)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        logger.info(f"Dataset: {train_size} train, {val_size} validation samples")

        self.update_state(state="PROGRESS", meta={"step": "model_initialization", "progress": 50})

        # 4. Initialize model -----------------------------------------------
        hidden_dim = hyperparams.get("hidden_dim", 128)
        num_layers = hyperparams.get("num_layers", 3)
        dropout = hyperparams.get("dropout", 0.3)

        if model_type == "lstm":
            model = AdvancedLSTM(
                input_dim=len(feature_columns),
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                dropout=dropout,
                output_horizon=prediction_horizon,
                bidirectional=True,
            )
        else:
            raise NotImplementedError(f"Model type {model_type} not yet implemented")

        # ── Move model to resolved device ──────────────────────────────────
        model = model.to(device)

        trainer = LSTMTrainer(model, device=device)
        trainer.target_mode = target_mode
        trainer.target_mode = target_mode

        self.update_state(state="PROGRESS", meta={"step": "training", "progress": 60})

        # 5. MLflow run -----------------------------------------------------
        mlflow = _setup_mlflow(f"lumina_{ticker}")

        with mlflow.start_run(run_name=f"{model_type}_{job_id}") as active_run:
            mlflow.log_params(
                {
                    "ticker": ticker,
                    "model_type": model_type,
                    "num_features": len(feature_columns),
                    "train_samples": train_size,
                    "val_samples": val_size,
                    "hidden_dim": hidden_dim,
                    "num_layers": num_layers,
                    "dropout": dropout,
                    "sequence_length": sequence_length,
                    "prediction_horizon": prediction_horizon,
                    "batch_size": batch_size,
                }
            )

            num_epochs = hyperparams.get("num_epochs", 50)
            learning_rate = hyperparams.get("learning_rate", 0.001)
            early_stopping_patience = hyperparams.get("early_stopping_patience", 10)

            history = trainer.train(
                train_loader=train_loader,
                val_loader=val_loader,
                num_epochs=num_epochs,
                learning_rate=learning_rate,
                early_stopping_patience=early_stopping_patience,
            )

            final_metrics = {
                "final_train_loss": history["train_loss"][-1],
                "final_val_loss": history["val_loss"][-1],
                "best_val_loss": min(history["val_loss"]),
                "epochs_trained": len(history["train_loss"]),
            }
            mlflow.log_metrics(final_metrics)

            # Save model checkpoint
            model_path = f"{settings.MODEL_STORAGE_PATH}/{ticker}_{model_type}_{job_id}.pt"
            os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)

            checkpoint_data = {
                "model_state_dict": model.state_dict(),
                "meta_data": {
                    "input_dim": len(feature_columns),
                    "hidden_dim": hidden_dim,
                    "num_layers": num_layers,
                    "dropout": dropout,
                    "sequence_length": sequence_length,
                    "prediction_horizon": prediction_horizon,
                    "feature_columns": feature_columns,
                    "feature_scaler": feature_scaler,
                    "target_mode": target_mode,
                    "feature_scaler": feature_scaler,
                    "target_mode": target_mode,
                    "ticker": ticker,
                    "trained_at": datetime.now().isoformat(),
                },
            }

            torch.save(checkpoint_data, model_path)
            mlflow.log_artifact(model_path)

            run_id = active_run.info.run_id

        logger.success(f"Training completed for job {job_id}, MLflow run_id={run_id}")

        # 6. Save model record to database ----------------------------------
        model_version = datetime.now().strftime("%Y%m%d_%H%M%S")
        try:
            db_model_id = run_async(  # type: ignore
                save_model_to_db(
                    {
                        "model_name": f"{model_type}_{ticker}",
                        "model_type": model_type,
                        "version": model_version,
                        "ticker": ticker,
                        "training_samples": train_size,
                        "validation_samples": val_size,
                        "mae": final_metrics.get("final_val_loss"),
                        "rmse": final_metrics.get("best_val_loss"),
                        "hyperparameters": {
                            "hidden_dim": hidden_dim,
                            "num_layers": num_layers,
                            "dropout": dropout,
                            "sequence_length": sequence_length,
                            "prediction_horizon": prediction_horizon,
                            "batch_size": batch_size,
                            "learning_rate": learning_rate,
                            "max_features": max_features,
                        },
                        "mlflow_run_id": run_id,
                        "is_active": True,
                    }
                )
            )
            logger.success(f"Model saved to DB with id={db_model_id}")
        except Exception as db_err:
            logger.warning(f"Failed to save model to DB (training still succeeded): {db_err}")
            db_model_id = None

        self.update_state(state="PROGRESS", meta={"step": "completed", "progress": 100})

        return {
            "job_id": job_id,
            "ticker": ticker,
            "model_type": model_type,
            "mlflow_run_id": run_id,
            "db_model_id": db_model_id,
            "model_path": model_path,
            "metrics": final_metrics,
            "feature_columns": feature_columns,
            "completed_at": datetime.now().isoformat(),
        }

    except Exception as e:
        logger.error(f"Training failed for job {job_id}: {e}", exc_info=True)
        raise


# ---------------------------------------------------------------------------
#  XGBoost training task
# ---------------------------------------------------------------------------


@shared_task(
    bind=True,
    name="workers.ml_tasks.train_xgboost_task",
    max_retries=2,
    default_retry_delay=120,
)
def train_xgboost_task(
    self,
    ticker: str,
    hyperparams: dict[str, Any] | None = None,
    lookback_days: int = 500,
    prediction_horizon: int = 5,
):
    """Train an XGBoost model and log to MLflow + DB."""
    try:
        import xgboost as xgb
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

        job_id = str(uuid4())[:8]
        logger.info(f"Starting XGBoost training for {ticker} (job={job_id})")

        self.update_state(state="PROGRESS", meta={"step": "data_collection", "progress": 10})

        # 1. Collect data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_days)
        data = _collect_data(ticker, start_date, end_date)

        if data is None or data.height == 0:
            raise ValueError(f"No data collected for {ticker}")

        data = _sort_market_data(data)

        data = _sort_market_data(data)

        self.update_state(state="PROGRESS", meta={"step": "feature_engineering", "progress": 30})

        # 2. Feature engineering  # type: ignore
        fe = FeatureEngineer()
        enriched = fe.create_all_features(data, add_lags=True, add_rolling=True)
        enriched_pd = enriched.to_pandas()

        feature_cols = _select_xgboost_feature_columns(enriched_pd)
        feature_cols = _select_xgboost_feature_columns(enriched_pd)
        if not feature_cols:
            raise ValueError("No feature columns generated")

        # Target: future return
        enriched_pd["target"] = (
            enriched_pd["close"].pct_change(prediction_horizon).shift(-prediction_horizon)
        )
        enriched_pd = enriched_pd.dropna(subset=["target"])

        X = enriched_pd[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
        y = enriched_pd["target"].values

        split_idx = int(len(X) * 0.8)
        if split_idx < 30:
            raise ValueError(f"Not enough data for training: {len(X)} rows")

        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        self.update_state(state="PROGRESS", meta={"step": "training", "progress": 50})

        # 3. Hyperparameters
        default_params = {
            "n_estimators": 500,
            "max_depth": 6,
            "learning_rate": 0.01,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "min_child_weight": 3,
            "random_state": 42,
            "n_jobs": -1,
        }
        if hyperparams:
            default_params.update(hyperparams)
        params = default_params

        # 4. MLflow run
        mlflow = _setup_mlflow(f"lumina_{ticker}")
        model_name = f"xgboost_{ticker}"
        model_version = datetime.now().strftime("%Y%m%d_%H%M%S")

        with mlflow.start_run(run_name=f"{model_name}_{model_version}") as active_run:
            mlflow.set_tags(
                {
                    "ticker": ticker,
                    "model_type": "xgboost",
                    "prediction_horizon": str(prediction_horizon),
                    "training_samples": str(len(X_train)),
                    "test_samples": str(len(X_test)),
                    "num_features": str(len(feature_cols)),
                }
            )
            mlflow.log_params({k: v for k, v in params.items() if k != "n_jobs"})

            model = xgb.XGBRegressor(**params)
            model.fit(
                X_train,
                y_train,
                eval_set=[(X_test, y_test)],
                verbose=False,
            )

            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)

            metrics = {
                "train_rmse": float(np.sqrt(mean_squared_error(y_train, y_pred_train))),
                "test_rmse": float(np.sqrt(mean_squared_error(y_test, y_pred_test))),
                "train_mae": float(mean_absolute_error(y_train, y_pred_train)),
                "test_mae": float(mean_absolute_error(y_test, y_pred_test)),
                "train_r2": float(r2_score(y_train, y_pred_train)),
                "test_r2": float(r2_score(y_test, y_pred_test)),
            }

            importance = model.feature_importances_
            feat_importance = sorted(
                zip(feature_cols, importance), key=lambda x: x[1], reverse=True
            )[:20]

            mlflow.log_metrics(metrics)
            mlflow.xgboost.log_model(
                model,
                artifact_path="model",
                registered_model_name=model_name,
            )

            run_id = active_run.info.run_id
            model_uri = f"runs:/{run_id}/model"

        logger.success(
            f"XGBoost trained for {ticker}: test_rmse={metrics['test_rmse']:.6f}, "
            f"test_r2={metrics['test_r2']:.4f}, run_id={run_id}"
        )

        # 5. Save to DB
        try:
            db_model_id = run_async(  # type: ignore
                save_model_to_db(
                    {
                        "model_name": model_name,
                        "model_type": "xgboost",
                        "version": model_version,
                        "ticker": ticker,
                        "training_samples": len(X_train),
                        "validation_samples": len(X_test),
                        "mae": metrics["test_mae"],
                        "rmse": metrics["test_rmse"],
                        "r2_score": metrics["test_r2"],
                        "hyperparameters": params,
                        "feature_importance": {f: float(v) for f, v in feat_importance},
                        "mlflow_run_id": run_id,
                        "is_active": True,
                    }
                )
            )
            logger.success(f"XGBoost model saved to DB: id={db_model_id}")
        except Exception as db_err:
            logger.warning(f"Failed to save XGBoost model to DB: {db_err}")
            db_model_id = None

        self.update_state(state="PROGRESS", meta={"step": "completed", "progress": 100})

        return {
            "ticker": ticker,
            "status": "success",
            "model_name": model_name,
            "model_version": model_version,
            "db_model_id": db_model_id,
            "mlflow_run_id": run_id,
            "mlflow_model_uri": model_uri,
            "metrics": metrics,
            "top_features": {f: float(v) for f, v in feat_importance[:10]},
            "completed_at": datetime.now().isoformat(),
        }

    except Exception as exc:
        logger.error(f"XGBoost training failed for {ticker}: {exc}", exc_info=True)
        raise self.retry(exc=exc) from exc


# ---------------------------------------------------------------------------
#  Prediction task
# ---------------------------------------------------------------------------


@shared_task(
    bind=True,
    name="workers.ml_tasks.predict_task",
    max_retries=2,
    default_retry_delay=60,
)
def predict_task(self, ticker: str, model_id: str, days_ahead: int):
    """Generate predictions using a trained LSTM model."""
    try:
        import torch

        from backend.ml_engine.models.lstm_advanced import (
            AdvancedLSTM,
            LSTMTrainer,
            TimeSeriesDataset,
        )
        from backend.ml_engine.models.lstm_advanced import (
            AdvancedLSTM,
            LSTMTrainer,
            TimeSeriesDataset,
        )

        logger.info(f"Prediction task for {ticker} using model {model_id}")

        self.update_state(state="PROGRESS", meta={"step": "loading_model", "progress": 10})

        # 1. Load model
        model_path = f"{settings.MODEL_STORAGE_PATH}/{model_id}.pt"

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)

        meta_data = checkpoint.get("meta_data", {})  # type: ignore
        input_dim = meta_data.get("input_dim", 50)
        hidden_dim = meta_data.get("hidden_dim", 128)
        num_layers = meta_data.get("num_layers", 3)
        dropout = meta_data.get("dropout", 0.3)
        sequence_length = meta_data.get("sequence_length", 60)
        feature_columns = meta_data.get("feature_columns", None)
        feature_scaler = meta_data.get("feature_scaler")
        target_mode = meta_data.get("target_mode", "price")
        trained_horizon = meta_data.get("prediction_horizon", days_ahead)
        feature_scaler = meta_data.get("feature_scaler")
        target_mode = meta_data.get("target_mode", "price")
        trained_horizon = meta_data.get("prediction_horizon", days_ahead)

        model = AdvancedLSTM(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            output_horizon=trained_horizon,
            output_horizon=trained_horizon,
            bidirectional=True,
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        trainer = LSTMTrainer(model)
        trainer.target_mode = target_mode
        trainer.target_mode = target_mode

        self.update_state(state="PROGRESS", meta={"step": "collecting_data", "progress": 30})

        # 2. Collect recent data
        lookback_days = max(sequence_length * 2, 90)
        data = _collect_data(
            ticker,
            start_date=datetime.now() - timedelta(days=lookback_days),
            end_date=datetime.now(),
        )

        if data is None or data.height == 0:
            raise ValueError(f"No data available for {ticker}")

        data = _sort_market_data(data)
        data = _sort_market_data(data)
        logger.info(f"Collected {data.height} data points")

        self.update_state(state="PROGRESS", meta={"step": "feature_engineering", "progress": 50})

        # 3. Feature engineering
        fe = FeatureEngineer()
        enriched_data = fe.create_all_features(data, add_lags=True, add_rolling=True)

        if feature_columns is None:
            feature_columns = fe.get_all_feature_names()[:input_dim]
            logger.warning(f"Using default feature columns: {len(feature_columns)} features")

        if not feature_columns:
            raise ValueError("No valid feature columns available for prediction")
        features = TimeSeriesDataset.transform_features(
            enriched_data,
            feature_columns,
            feature_scaler=feature_scaler,
        )
        if not feature_columns:
            raise ValueError("No valid feature columns available for prediction")
        features = TimeSeriesDataset.transform_features(
            enriched_data,
            feature_columns,
            feature_scaler=feature_scaler,
        )

        self.update_state(state="PROGRESS", meta={"step": "generating_predictions", "progress": 80})

        # 4. Prepare input and predict
        if len(features) < sequence_length:
            raise ValueError(f"Not enough data. Need {sequence_length}, got {len(features)}")

        input_sequence = torch.FloatTensor(features[-sequence_length:]).unsqueeze(0)
        current_price = float(data["close"].tail(1).item())
        trainer.prediction_context = {"last_close": current_price, "target_mode": target_mode}
        current_price = float(data["close"].tail(1).item())
        trainer.prediction_context = {"last_close": current_price, "target_mode": target_mode}

        with torch.no_grad():
            predictions = trainer.predict(input_sequence)

        # 5. Format results
        last_date = _get_last_market_timestamp(data)
        forecast_horizon = min(days_ahead, int(predictions["price"].shape[1]))
        last_date = _get_last_market_timestamp(data)
        forecast_horizon = min(days_ahead, int(predictions["price"].shape[1]))

        formatted_predictions = []
        predicted_price_values: list[float] = []
        for i in range(forecast_horizon):
        predicted_price_values: list[float] = []
        for i in range(forecast_horizon):
            pred_price = float(predictions["price"][0][i])
            pred_date = last_date + timedelta(days=i + 1)

            while pred_date.weekday() >= 5:
                pred_date += timedelta(days=1)

            rounded_pred_price = round(pred_price, 2)
            predicted_price_values.append(rounded_pred_price)
            rounded_pred_price = round(pred_price, 2)
            predicted_price_values.append(rounded_pred_price)
            formatted_predictions.append(
                {
                    "day": i + 1,
                    "date": pred_date.isoformat(),
                    "predicted_price": rounded_pred_price,
                    "predicted_price": rounded_pred_price,
                    "change": round(pred_price - current_price, 2),
                    "change_percent": round(
                        ((pred_price - current_price) / current_price) * 100, 2
                    ),
                    "confidence_lower": round(float(predictions["price_lower"][0][i]), 2),
                    "confidence_upper": round(float(predictions["price_upper"][0][i]), 2),
                    "uncertainty": round(float(predictions["uncertainty"][0][i]), 4),
                }
            )

        avg_predicted_price = float(np.mean(np.asarray(predicted_price_values, dtype=float)))
        total_change_percent = (predicted_price_values[-1] - current_price) / current_price * 100

        self.update_state(state="PROGRESS", meta={"step": "completed", "progress": 100})

        return {
            "ticker": ticker,
            "model_id": model_id,
            "current_price": round(current_price, 2),
            "current_date": last_date.isoformat(),
            "predictions": formatted_predictions,
            "regime_probabilities": {
                "bear": round(float(predictions["regime_probs"][0][0]), 4),
                "sideways": round(float(predictions["regime_probs"][0][1]), 4),
                "bull": round(float(predictions["regime_probs"][0][2]), 4),
            },
            "volatility_forecast": round(float(predictions["volatility"][0][0]), 4),
            "summary": {
                "avg_predicted_price": round(avg_predicted_price, 2),
                "total_change_percent": round(total_change_percent, 2),
                "trend": (
                    "bullish"
                    if total_change_percent > 1
                    else "bearish"
                    if total_change_percent < -1
                    else "neutral"
                ),
            },
            "metadata": {
                "model_type": "lstm",
                "sequence_length": sequence_length,
                "num_features": len(feature_columns),
                "prediction_horizon": forecast_horizon,
                "trained_prediction_horizon": trained_horizon,
                "target_mode": target_mode,
                "prediction_horizon": forecast_horizon,
                "trained_prediction_horizon": trained_horizon,
                "target_mode": target_mode,
            },
            "completed_at": datetime.now().isoformat(),
        }

    except FileNotFoundError as e:
        logger.error(f"Model not found: {e}")
        raise
    except ValueError as e:
        logger.error(f"Data error: {e}")
        raise
    except Exception as e:
        logger.error(f"Prediction failed: {e}", exc_info=True)
        raise


# ---------------------------------------------------------------------------
#  Periodic retraining
# ---------------------------------------------------------------------------


@shared_task(name="workers.ml_tasks.retrain_models")
def retrain_models_task():
    """Periodic task to retrain models with new data."""
    try:
        logger.info("Starting periodic model retraining")

        tickers_to_retrain = ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN"]

        results = []
        for ticker in tickers_to_retrain:
            try:
                job_id = f"retrain_{ticker}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

                hyperparams = {
                    "hidden_dim": 128,
                    "num_layers": 3,
                    "dropout": 0.3,
                    "sequence_length": 60,
                    "prediction_horizon": 5,
                    "batch_size": 32,
                    "num_epochs": 30,
                    "learning_rate": 0.001,
                    "early_stopping_patience": 10,
                    "max_features": 50,
                }

                task = train_model_task.delay(
                    job_id=job_id,
                    ticker=ticker,
                    model_type="lstm",  # type: ignore
                    hyperparams=hyperparams,
                )

                results.append(
                    {
                        "ticker": ticker,
                        "job_id": job_id,
                        "task_id": task.id,
                        "status": "queued",
                    }
                )

                logger.info(f"Queued retraining for {ticker}")

            except Exception as e:
                logger.error(f"Failed to queue retraining for {ticker}: {e}")
                results.append({"ticker": ticker, "status": "failed", "error": str(e)})
        # type: ignore
        return {
            "status": "Retraining jobs submitted",
            "total_models": len(tickers_to_retrain),
            "successful": len([r for r in results if r["status"] == "queued"]),
            "failed": len([r for r in results if r["status"] == "failed"]),
            "results": results,
            "timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        logger.error(f"Periodic retraining failed: {e}")
        raise


# ---------------------------------------------------------------------------
#  Model evaluation task
# ---------------------------------------------------------------------------


@shared_task(name="workers.ml_tasks.evaluate_model")
def evaluate_model_task(model_id: str, test_start_date: str, test_end_date: str):
    """Evaluate a trained model on test data."""
    try:
        import torch
        from torch.utils.data import DataLoader

        from backend.ml_engine.models.lstm_advanced import (
            AdvancedLSTM,
            LSTMTrainer,
            TimeSeriesDataset,
        )

        logger.info(f"Evaluating model {model_id}")

        start_date = datetime.fromisoformat(test_start_date)
        end_date = datetime.fromisoformat(test_end_date)

        ticker = model_id.split("_")[0]

        # Load model
        model_path = f"{settings.MODEL_STORAGE_PATH}/{model_id}.pt"
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)

        meta_data = checkpoint.get("meta_data", {})

        model = AdvancedLSTM(
            input_dim=meta_data.get("input_dim", 50),
            hidden_dim=meta_data.get("hidden_dim", 128),
            num_layers=meta_data.get("num_layers", 3),
            dropout=meta_data.get("dropout", 0.3),
            output_horizon=meta_data.get("prediction_horizon", 5),
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        target_mode = meta_data.get("target_mode", "price")
        target_mode = meta_data.get("target_mode", "price")
        trainer = LSTMTrainer(model)
        trainer.target_mode = target_mode
        trainer.target_mode = target_mode

        # Collect test data
        data = _collect_data(ticker, start_date, end_date)

        if data is None or data.height == 0:
            raise ValueError(f"No test data available for {ticker}")

        data = _sort_market_data(data)

        data = _sort_market_data(data)

        # Engineer features
        fe = FeatureEngineer()
        enriched_data = fe.create_all_features(data)

        feature_columns = meta_data.get(
            "feature_columns", fe.get_all_feature_names()[: meta_data.get("input_dim", 50)]
        )

        test_dataset = TimeSeriesDataset(
            data=enriched_data,
            feature_columns=feature_columns,
            sequence_length=meta_data.get("sequence_length", 60),
            prediction_horizon=meta_data.get("prediction_horizon", 5),
            feature_scaler=meta_data.get("feature_scaler"),
            target_mode=target_mode,
            feature_scaler=meta_data.get("feature_scaler"),
            target_mode=target_mode,
        )

        test_loader = DataLoader(test_dataset, batch_size=32)

        test_metrics = trainer.validate(test_loader)

        # Directional accuracy
        all_predictions = []
        all_targets = []
        all_last_closes = []
        all_last_closes = []

        with torch.no_grad():
            for batch_features, batch_targets in test_loader:
                outputs = model(batch_features)
                decoded_prices, _, _, _ = trainer.decode_price_outputs(
                    outputs["price"].cpu().numpy(),
                    outputs["uncertainty"].cpu().numpy(),
                    last_close=batch_targets["last_close"].cpu().numpy().reshape(-1),
                    target_mode=target_mode,
                )
                all_predictions.append(decoded_prices)
                all_targets.append(batch_targets["future_prices"].cpu().numpy())
                all_last_closes.append(batch_targets["last_close"].cpu().numpy())
                decoded_prices, _, _, _ = trainer.decode_price_outputs(
                    outputs["price"].cpu().numpy(),
                    outputs["uncertainty"].cpu().numpy(),
                    last_close=batch_targets["last_close"].cpu().numpy().reshape(-1),
                    target_mode=target_mode,
                )
                all_predictions.append(decoded_prices)
                all_targets.append(batch_targets["future_prices"].cpu().numpy())
                all_last_closes.append(batch_targets["last_close"].cpu().numpy())

        predictions = np.concatenate(all_predictions, axis=0)
        targets = np.concatenate(all_targets, axis=0)
        last_closes = np.concatenate(all_last_closes, axis=0).reshape(-1)
        last_closes = np.concatenate(all_last_closes, axis=0).reshape(-1)

        pred_direction = np.sign(predictions[:, 0] - last_closes)
        actual_direction = np.sign(targets[:, 0] - last_closes)
        pred_direction = np.sign(predictions[:, 0] - last_closes)
        actual_direction = np.sign(targets[:, 0] - last_closes)
        directional_accuracy = float(np.mean(pred_direction == actual_direction))

        result = {
            "model_id": model_id,
            "ticker": ticker,
            "test_period": {
                "start": start_date.isoformat(),
                "end": end_date.isoformat(),
            },
            "metrics": {
                "test_loss": test_metrics["total_loss"],
                "price_loss": test_metrics["price_loss"],
                "volatility_loss": test_metrics.get("volatility_loss", 0),
                "regime_loss": test_metrics.get("regime_loss", 0),
                "directional_accuracy": directional_accuracy,
                "num_samples": len(test_dataset),
            },
            "evaluated_at": datetime.now().isoformat(),
        }

        logger.success(f"Model {model_id} evaluated successfully")
        return result

    except Exception as e:
        logger.error(f"Evaluation failed: {e}", exc_info=True)
        raise


# ---------------------------------------------------------------------------
#  SHAP feature importance
# ---------------------------------------------------------------------------


@shared_task(name="workers.ml_tasks.compute_feature_importance")
def compute_feature_importance_task(model_id: str, num_samples: int = 100):
    """Compute SHAP feature importance for a model."""
    try:
        import shap
        import torch

        from backend.ml_engine.models.lstm_advanced import AdvancedLSTM, TimeSeriesDataset
        from backend.ml_engine.models.lstm_advanced import AdvancedLSTM, TimeSeriesDataset

        logger.info(f"Computing feature importance for model {model_id}")

        ticker = model_id.split("_")[0]

        model_path = f"{settings.MODEL_STORAGE_PATH}/{model_id}.pt"
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)

        meta_data = checkpoint.get("meta_data", {})

        model = AdvancedLSTM(
            input_dim=meta_data.get("input_dim", 50),
            hidden_dim=meta_data.get("hidden_dim", 128),
            num_layers=meta_data.get("num_layers", 3),
            dropout=meta_data.get("dropout", 0.3),
            output_horizon=meta_data.get("prediction_horizon", 5),
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        # Collect data
        data = _collect_data(
            ticker,
            start_date=datetime.now() - timedelta(days=180),
            end_date=datetime.now(),
        )

        fe = FeatureEngineer()
        enriched_data = fe.create_all_features(_sort_market_data(data))
        enriched_data = fe.create_all_features(_sort_market_data(data))

        feature_columns = meta_data.get(
            "feature_columns",
            fe.get_all_feature_names()[: meta_data.get("input_dim", 50)],
        )
        features = TimeSeriesDataset.transform_features(
            enriched_data,
            feature_columns,
            feature_scaler=meta_data.get("feature_scaler"),
        )
        features = TimeSeriesDataset.transform_features(
            enriched_data,
            feature_columns,
            feature_scaler=meta_data.get("feature_scaler"),
        )

        seq_len = meta_data.get("sequence_length", 60)

        if len(features) > num_samples + seq_len:
            indices = np.random.choice(len(features) - seq_len, num_samples, replace=False)
            sample_data = np.array([features[i : i + seq_len] for i in indices])
        else:
            sample_data = np.array(
                [features[i : i + seq_len] for i in range(len(features) - seq_len)]
            )

        sample_data = torch.FloatTensor(sample_data)  # type: ignore

        def model_predict(x):
            with torch.no_grad():
                x_tensor = torch.FloatTensor(x)
                outputs = model(x_tensor)
                return outputs["price"][:, 0].cpu().numpy()

        background = sample_data[: min(10, len(sample_data))]

        explainer = shap.DeepExplainer(lambda x: model_predict(x), background.numpy())  # type: ignore
        shap_values = explainer.shap_values(sample_data.numpy())  # type: ignore

        mean_shap = np.abs(shap_values).mean(axis=(0, 1))

        feature_importance = {
            feature_columns[i]: float(mean_shap[i]) for i in range(len(feature_columns))
        }

        sorted_importance = dict(
            sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        )

        result = {
            "model_id": model_id,
            "ticker": ticker,
            "feature_importance": sorted_importance,
            "top_10_features": dict(list(sorted_importance.items())[:10]),
            "num_samples_analyzed": len(sample_data),
            "computed_at": datetime.now().isoformat(),
        }

        logger.success(f"Feature importance computed for {model_id}")
        return result

    except Exception as e:
        logger.error(f"Feature importance calculation failed: {e}", exc_info=True)
        raise
