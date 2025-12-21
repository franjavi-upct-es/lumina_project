# backend/workers/ml_tasks.py
"""
Celery tasks for machine learning model training and inference
"""

from celery import shared_task
from typing import Dict, Any
from datetime import datetime, timedelta
from torch.utils.data import DataLoader, random_split
from loguru import logger
import mlflow

from ml_engine.models.lstm_advanced import AdvancedLSTM, LSTMTrainer, TimeSeriesDataset
from data_engine.collectors.yfinance_collector import YFinanceCollector
from data_engine.transformers.feature_engineering import FeatureEngineer
from config.settings import get_settings

settings = get_settings()


@shared_task(bind=True, name="workers.ml_tasks.train_model_task")
def train_model_task(
    self, job_id: str, ticker: str, model_type: str, hyperparams: Dict[str, Any]
):
    """
    Train a machine learning model asynchronously

    Args:
        job_id: Unique job identifier
        ticker: Stock ticker symbol
        model_type: Type of model (lstm, transformer, xgboost, ensemble)
        hyperparams: Model and training hyperparameters
    """
    try:
        logger.info(f"Starting training job {
                    job_id} for {ticker} ({model_type})")

        # Update task state
        self.update_state(
            state="PROGRESS", meta={"step": "data_collection", "progress": 0}
        )

        # 1. Collect data
        import asyncio

        collector = YFinanceCollector()

        start_date = hyperparams.get("start_date")
        end_date = hyperparams.get("end_date")

        if not start_date:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=365 * 3)  # 3 years default

        # Run async collection in sync context
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        data = loop.run_until_complete(
            collector.collect_with_retry(ticker, start_date, end_date)
        )
        loop.close()

        if data is None or data.height == 0:
            raise ValueError(f"No data collected for {ticker}")

        logger.info(f"Collected {data.height} data points for {ticker}")

        # Update progress
        self.update_state(
            state="PROGRESS", meta={"step": "feature_engineering", "progress": 20}
        )

        # 2. Engineer features
        fe = FeatureEngineer()
        enriched_data = fe.create_all_features(
            data, add_lags=True, add_rolling=True)

        # Select top features
        max_features = hyperparams.get("max_features", 50)
        feature_columns = fe.get_all_feature_names()[:max_features]

        logger.info(f"Engineered {len(feature_columns)} features")

        # Update progress
        self.update_state(
            state="PROGRESS", meta={"step": "dataset_preparation", "progress": 40}
        )

        # 3. Prepare dataset
        sequence_length = hyperparams.get("sequence_length", 60)
        prediction_horizon = hyperparams.get("prediction_horizon", 5)

        dataset = TimeSeriesDataset(
            data=enriched_data,
            feature_columns=feature_columns,
            sequence_length=sequence_length,
            prediction_horizon=prediction_horizon,
            stride=1,
        )

        # Split train/validation
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(
            dataset, [train_size, val_size])

        batch_size = hyperparams.get("batch_size", 32)
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        logger.info(
            f"Dataset prepared: {train_size} train, {
                val_size} validation samples"
        )

        # Update progress
        self.update_state(
            state="PROGRESS", meta={"step": "model_initialization", "progress": 50}
        )

        # 4. Initialize model
        if model_type == "lstm":
            model = AdvancedLSTM(
                input_dim=len(feature_columns),
                hidden_dim=hyperparams.get("hidden_dim", 128),
                num_layers=hyperparams.get("num_layers", 3),
                dropout=hyperparams.get("dropout", 0.3),
                output_horizon=prediction_horizon,
                bidirectional=True,
            )
        else:
            raise NotImplementedError(
                f"Model type {model_type} not yet implemented")

        trainer = LSTMTrainer(model)

        logger.info(f"Initialized {model_type} model")

        # Update progress
        self.update_state(state="PROGRESS", meta={
                          "step": "training", "progress": 60})

        # 5. Start MLflow run
        mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
        mlflow.set_experiment(f"lumina_{ticker}")

        with mlflow.start_run(run_name=f"{model_type}_{job_id}"):
            # Log parameters
            mlflow.log_params(hyperparams)
            mlflow.log_param("ticker", ticker)
            mlflow.log_param("num_features", len(feature_columns))
            mlflow.log_param("train_samples", train_size)
            mlflow.log_param("val_samples", val_size)

            # Train model
            num_epochs = hyperparams.get("num_epochs", 50)
            learning_rate = hyperparams.get("learning_rate", 0.001)
            early_stopping_patience = hyperparams.get(
                "early_stopping_patience", 10)

            history = trainer.train(
                train_loader=train_loader,
                val_loader=val_loader,
                num_epochs=num_epochs,
                learning_rate=learning_rate,
                early_stopping_patience=early_stopping_patience,
            )

            # Log metrics
            mlflow.log_metrics(
                {
                    "final_train_loss": history["train_loss"][-1],
                    "final_val_loss": history["val_loss"][-1],
                    "best_val_loss": min(history["val_loss"]),
                    "epochs_trained": len(history["train_loss"]),
                }
            )

            # Save model
            model_path = (
                f"{settings.MODEL_STORAGE_PATH}/{ticker}_{model_type}_{job_id}.pt"
            )
            trainer.save_checkpoint(model_path)
            mlflow.log_artifact(model_path)

            run_id = mlflow.active_run().info.run_id

        logger.success(f"Training completed for job {job_id}")

        # Update progress
        self.update_state(state="PROGRESS", meta={
                          "step": "completed", "progress": 100})

        # Return result
        return {
            "job_id": job_id,
            "ticker": ticker,
            "model_type": model_type,
            "mlflow_run_id": run_id,
            "model_path": model_path,
            "metrics": {
                "train_loss": history["train_loss"][-1],
                "val_loss": history["val_loss"][-1],
                "epochs": len(history["train_loss"]),
            },
            "completed_at": datetime.now().isoformat(),
        }

    except Exception as e:
        logger.error(f"Training failed for job {job_id}: {e}")
        self.update_state(state="FAILURE", meta={"error": str(e)})
        raise


@shared_task(bind=True, name="workers.ml_taks.predict_task")
def predict_task(self, ticker: str, model_id: str, days_ahead: int):
    """
    Generate predictions using a trained model
    """
    try:
        logger.info(f"Prediction task for {ticker} using model {model_id}")

        # TODO: Implement prediction logic

        return {
            "ticker": ticker,
            "model_id": model_id,
            "predictions": [],
            "completed_at": datetime.now().isoformat(),
        }

    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise


@shared_task(name="workers.ml_taks.retrained_models")
def retrained_models_task():
    """
    Periodic task to retrain models with new data
    """
    try:
        logger.info("Starting periodic model retraining")

        # TODO: Implement periodic retraining

        return {"status": "Models retrained successfully"}

    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise


@shared_task(name="workers.ml_tasks.compute_feature_importance")
def compute_feature_importance_task(model_id: str):
    """
    Compute SHAP feature importance for a model
    """
    try:
        logger.info(f"Computing feature importance for model {model_id}")

        # TODO: Implement SHAP calculation

        return {"model_id": model_id, "feature_importance": {}}

    except Exception as e:
        logger.error(f"Feature importance calculation failed: {e}")
        raise
