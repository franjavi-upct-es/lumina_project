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
import numpy as np
import torch
import shap

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
        logger.info(f"Starting training job {job_id} for {ticker} ({model_type})")

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
        enriched_data = fe.create_all_features(data, add_lags=True, add_rolling=True)

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
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        batch_size = hyperparams.get("batch_size", 32)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        logger.info(
            f"Dataset prepared: {train_size} train, {val_size} validation samples"
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
            raise NotImplementedError(f"Model type {model_type} not yet implemented")

        trainer = LSTMTrainer(model)

        logger.info(f"Initialized {model_type} model")

        # Update progress
        self.update_state(state="PROGRESS", meta={"step": "training", "progress": 60})

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
            early_stopping_patience = hyperparams.get("early_stopping_patience", 10)

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
        self.update_state(state="PROGRESS", meta={"step": "completed", "progress": 100})

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
            "feature_columns": feature_columns,
            "completed_at": datetime.now().isoformat(),
        }

    except Exception as e:
        logger.error(f"Training failed for job {job_id}: {e}")
        self.update_state(state="FAILURE", meta={"error": str(e)})
        raise


@shared_task(bind=True, name="workers.ml_tasks.predict_task")
def predict_task(self, ticker: str, model_id: str, days_ahead: int):
    """
    Generate predictions using a trained model
    """
    try:
        logger.info(f"Prediction task for {ticker} using model {model_id}")

        # 1. Load model
        model_path = f"{settings.MODEL_STORAGE_PATH}/{model_id}.pt"

        # Load checkpoint
        checkpoint = torch.load(model_path, map_location="cpu")

        # Reconstruct model (need to get hyperparams from DB or MLflow)
        # For now, use default params
        model = AdvancedLSTM(
            input_dim=50,  # Should load from metadata
            hidden_dim=128,
            num_layers=3,
            dropout=0.3,
            output_horizon=days_ahead,
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        trainer = LSTMTrainer(model)

        # 2. Collect recent data
        import asyncio

        collector = YFinanceCollector()
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        data = loop.run_until_complete(
            collector.collect_with_retry(
                ticker=ticker,
                start_date=datetime.now() - timedelta(days=90),
                end_date=datetime.now(),
            )
        )
        loop.close()

        if data is None or data.height == 0:
            raise ValueError(f"No data available for {ticker}")

        # 3. Engineer features
        fe = FeatureEngineer()
        enriched_data = fe.create_all_features(data)

        # Get feature columns (should load from model metadata)
        feature_columns = fe.get_all_feature_names()[:50]
        features = enriched_data.select(feature_columns).to_numpy()

        # 4. Prepare input sequence
        sequence_length = 60
        if len(features) < sequence_length:
            raise ValueError(
                f"Not enough data points. Need {sequence_length}, got {len(features)}"
            )

        input_sequence = torch.FloatTensor(features[-sequence_length:])

        # 5. Generate predictions
        predictions = trainer.predict(input_sequence)

        # 6. Format predictions
        current_price = float(data["close"].tail(1).item())

        formatted_predictions = []
        for i in range(days_ahead):
            pred_price = float(predictions["price"][0][i])
            formatted_predictions.append(
                {
                    "day": i + 1,
                    "date": (datetime.now() + timedelta(days=i + 1)).isoformat(),
                    "predicted_price": pred_price,
                    "change": pred_price - current_price,
                    "change_percent": ((pred_price - current_price) / current_price)
                    * 100,
                    "confidence_lower": float(predictions["price_lower"][0][i]),
                    "confidence_upper": float(predictions["price_upper"][0][i]),
                    "uncertainty": float(predictions["uncertainty"][0][i]),
                }
            )

        result = {
            "ticker": ticker,
            "model_id": model_id,
            "current_price": current_price,
            "predictions": formatted_predictions,
            "regime_probabilities": {
                "bear": float(predictions["regime_probs"][0][0]),
                "sideways": float(predictions["regime_probs"][0][1]),
                "bull": float(predictions["regime_probs"][0][2]),
            },
            "volatility_forecast": float(predictions["volatility"][0][0]),
            "completed_at": datetime.now().isoformat(),
        }

        logger.success(f"Predictions generated for {ticker}")
        return result

    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise


@shared_task(name="workers.ml_tasks.retrain_models")
def retrain_models_task():
    """
    Periodic task to retrain models with new data
    """
    try:
        logger.info("Starting periodic model retraining")

        # Get list of active models from database
        # For now, use a hardcoded list of popular tickers
        tickers_to_retrain = ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN"]

        results = []
        for ticker in tickers_to_retrain:
            try:
                # Trigger training task for each ticker
                job_id = f"retrain_{ticker}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

                hyperparams = {
                    "start_date": datetime.now() - timedelta(days=365 * 3),
                    "end_date": datetime.now(),
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

                # Submit async training task
                task = train_model_task.delay(
                    job_id=job_id,
                    ticker=ticker,
                    model_type="lstm",
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
                results.append(
                    {
                        "ticker": ticker,
                        "status": "failed",
                        "error": str(e),
                    }
                )

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


@shared_task(name="workers.ml_tasks.evaluate_model")
def evaluate_model_task(model_id: str, test_start_date: str, test_end_date: str):
    """
    Evaluate a trained model on test data
    """
    try:
        logger.info(f"Evaluating model {model_id}")

        # Parse dates
        start_date = datetime.fromisoformat(test_start_date)
        end_date = datetime.fromisoformat(test_end_date)

        # Load model metadata to get ticker
        # For now, extract from model_id (format: ticker_modeltype_jobid)
        ticker = model_id.split("_")[0]

        # Load model
        model_path = f"{settings.MODEL_STORAGE_PATH}/{model_id}.pt"
        checkpoint = torch.load(model_path, map_location="cpu")

        model = AdvancedLSTM(
            input_dim=50,
            hidden_dim=128,
            num_layers=3,
            dropout=0.3,
            output_horizon=5,
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        trainer = LSTMTrainer(model)

        # Collect test data
        import asyncio

        collector = YFinanceCollector()
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        data = loop.run_until_complete(
            collector.collect_with_retry(
                ticker=ticker,
                start_date=start_date,
                end_date=end_date,
            )
        )
        loop.close()

        if data is None or data.height == 0:
            raise ValueError(f"No test data available for {ticker}")

        # Engineer features
        fe = FeatureEngineer()
        enriched_data = fe.create_all_features(data)

        feature_columns = fe.get_all_feature_names()[:50]

        # Create test dataset
        test_dataset = TimeSeriesDataset(
            data=enriched_data,
            feature_columns=feature_columns,
            sequence_length=60,
            prediction_horizon=5,
        )

        test_loader = DataLoader(test_dataset, batch_size=32)

        # Evaluate
        test_metrics = trainer.validate(test_loader)

        # Calculate additional metrics
        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for batch_features, batch_targets in test_loader:
                outputs = model(batch_features)
                all_predictions.append(outputs["price"].cpu().numpy())
                all_targets.append(batch_targets["price"].cpu().numpy())

        predictions = np.concatenate(all_predictions, axis=0)
        targets = np.concatenate(all_targets, axis=0)

        # Calculate directional accuracy
        pred_direction = np.sign(predictions[:, 0] - targets[:, 0])
        actual_direction = np.sign(targets[:, 1] - targets[:, 0])
        directional_accuracy = np.mean(pred_direction == actual_direction)

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
                "volatility_loss": test_metrics["volatility_loss"],
                "regime_loss": test_metrics["regime_loss"],
                "directional_accuracy": float(directional_accuracy),
                "num_samples": len(test_dataset),
            },
            "evaluated_at": datetime.now().isoformat(),
        }

        logger.success(f"Model {model_id} evaluated successfully")
        return result

    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise


@shared_task(name="workers.ml_tasks.compute_feature_importance")
def compute_feature_importance_task(model_id: str, num_samples: int = 100):
    """
    Compute SHAP feature importance for a model
    """
    try:
        logger.info(f"Computing feature importance for model {model_id}")

        # Extract ticker from model_id
        ticker = model_id.split("_")[0]

        # Load model
        model_path = f"{settings.MODEL_STORAGE_PATH}/{model_id}.pt"
        checkpoint = torch.load(model_path, map_location="cpu")

        model = AdvancedLSTM(
            input_dim=50,
            hidden_dim=128,
            num_layers=3,
            dropout=0.3,
            output_horizon=5,
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        # Collect recent data
        import asyncio

        collector = YFinanceCollector()
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        data = loop.run_until_complete(
            collector.collect_with_retry(
                ticker=ticker,
                start_date=datetime.now() - timedelta(days=180),
                end_date=datetime.now(),
            )
        )
        loop.close()

        # Engineer features
        fe = FeatureEngineer()
        enriched_data = fe.create_all_features(data)

        feature_columns = fe.get_all_feature_names()[:50]
        features = enriched_data.select(feature_columns).to_numpy()

        # Sample data for SHAP
        if len(features) > num_samples:
            indices = np.random.choice(len(features) - 60, num_samples, replace=False)
            sample_data = np.array([features[i : i + 60] for i in indices])
        else:
            sample_data = np.array(
                [features[i : i + 60] for i in range(len(features) - 60)]
            )

        sample_data = torch.FloatTensor(sample_data)

        # Create SHAP explainer
        def model_predict(x):
            with torch.no_grad():
                x_tensor = torch.FloatTensor(x)
                outputs = model(x_tensor)
                return outputs["price"][:, 0].cpu().numpy()

        # Use a subset as background
        background = sample_data[: min(10, len(sample_data))]

        # DeepExplainer for neural networks
        explainer = shap.DeepExplainer(lambda x: model_predict(x), background.numpy())

        # Calculate SHAP values
        shap_values = explainer.shap_values(sample_data.numpy())

        # Average absolute SHAP values across samples and time steps
        # Shape: (num_samples, seq_length, num_features)
        mean_shap = np.abs(shap_values).mean(axis=(0, 1))

        # Create feature importance dictionary
        feature_importance = {
            feature_columns[i]: float(mean_shap[i]) for i in range(len(feature_columns))
        }

        # Sort by importance
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
        logger.error(f"Feature importance calculation failed: {e}")
        raise
