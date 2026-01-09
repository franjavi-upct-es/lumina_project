# backend/api/routes/ml.py
"""
Machine Learning endpoints for model training, prediction, and evaluation
"""

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Annotated, Any
from uuid import uuid4

import torch
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query
from loguru import logger
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.config.settings import get_settings
from backend.data_engine.collectors.yfinance_collector import YFinanceCollector
from backend.data_engine.transformers.feature_engineering import FeatureEngineer
from backend.db.models import get_async_session
from backend.ml_engine.models.lstm_advanced import AdvancedLSTM, LSTMTrainer, TimeSeriesDataset
from backend.workers.ml_tasks import (
    compute_feature_importance_task,
    evaluate_model_task,
    train_model_task,
)

router = APIRouter()
settings = get_settings()


# Request Models
class TrainModelRequest(BaseModel):
    ticker: str
    model_type: str = Field("lstm", pattern="^(lstm|transformer|xgboost|ensemble)$")
    start_date: datetime | None = None
    end_date: datetime | None = None

    # Model hyperparameters
    hidden_dim: int = Field(128, ge=32, le=512)
    num_layers: int = Field(3, ge=1, le=5)
    dropout: float = Field(0.3, ge=0.0, le=0.5)
    sequence_length: int = Field(60, ge=10, le=200)
    prediction_horizon: int = Field(5, ge=1, le=20)

    # Training parameters
    batch_size: int = Field(32, ge=8, le=128)
    num_epochs: int = Field(50, ge=5, le=200)
    learning_rate: float = Field(0.001, ge=0.0001, le=0.01)
    early_stopping_patience: int = Field(10, ge=3, le=30)

    # Feature selection
    max_features: int = Field(50, ge=10, le=200)
    feature_categories: list[str] | None = None

    # Options
    async_training: bool = True
    save_model: bool = True


class PredictRequest(BaseModel):
    ticker: str
    model_id: str | None = None
    model_type: str = Field("lstm", pattern="^(lstm|transformer|xgboost|ensemble)$")
    days_ahead: int = Field(5, ge=1, le=20)
    include_uncertainty: bool = True
    include_attention: bool = True


class ModelEvaluationRequest(BaseModel):
    model_id: str
    test_start_date: datetime
    test_end_date: datetime
    metrics: list[str] | None = None


# Response Models
class TrainJobResponse(BaseModel):
    job_id: str
    ticker: str
    model_type: str
    status: str
    message: str
    estimated_time_minutes: int | None = None


class PredictionResponse(BaseModel):
    ticker: str
    model_id: str
    model_type: str
    prediction_time: datetime
    predictions: list[dict[str, Any]]
    uncertainty: dict[str, Any] | None = None
    attention_weights: list[float] | None = None
    regime_probabilities: dict[str, float] | None = None


class ModelListResponse(BaseModel):
    models: list[dict[str, Any]]
    total: int


class ModelDetailsResponse(BaseModel):
    model_id: str
    model_name: str
    model_type: str
    ticker: str
    trained_on: datetime
    hyperparameters: dict[str, Any]
    performance: dict[str, float]
    feature_importance: dict[str, float] | None = None
    is_active: bool


# In-memory job tracking (in production, use Redis)
training_jobs = {}


@router.post("/train", response_model=TrainJobResponse)
async def train_model(request: TrainModelRequest, background_tasks: BackgroundTasks):
    """
    Train a machine learning model

    **Model Types:**
    - lstm: Advanced LSTM with attention mechanism
    - transformer: Temporal Transformer
    - xgboost: Gradient boosting for tabular data
    - ensemble: Combination of multiple models

    **Training Options:**
    - async_training: If True, returns immediately with job_id
    - save_model: If True, saves model to MLflow
    """
    try:
        job_id = str(uuid4())

        logger.info(f"Received training request for {request.ticker} ({request.model_type})")

        if request.async_training:
            # Async training with Celery
            task = train_model_task.delay(
                job_id=job_id,
                ticker=request.ticker,
                model_type=request.model_type,
                hyperparams=request.model_dump(),
            )

            training_jobs[job_id] = {
                "task_id": task.id,
                "ticker": request.ticker,
                "model_type": request.model_type,
                "status": "queued",
                "created_at": datetime.now(),
            }

            return TrainJobResponse(
                job_id=job_id,
                ticker=request.ticker,
                model_type=request.model_type,
                status="queued",
                message="Training job submitted. Check /ml/jobs/{job_id} for status.",
                estimated_time_minutes=10 if request.model_type == "lstm" else 5,
            )
        else:
            # Synchronous training (not recommended for production)
            background_tasks.add_task(_train_model_sync, job_id, request)

            return TrainJobResponse(
                job_id=job_id,
                ticker=request.ticker,
                model_type=request.model_type,
                status="training",
                message="Training started in background",
                estimated_time_minutes=10,
            )
    except Exception as e:
        logger.error(f"Error initiating training: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/jobs/{job_id}")
async def get_training_job_status(job_id: str):
    """
    Get status of a training job
    """
    if job_id not in training_jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = training_jobs[job_id]

    # Check Celery task status
    from workers.celery_app import celery_app

    task = celery_app.AsyncResult(job["task_id"])

    status = {
        "job_id": job_id,
        "ticker": job["ticker"],
        "model_type": job["model_type"],
        "status": task.state,
        "created_at": job["created_at"].isoformat(),
    }

    # Add progress if available
    if task.state == "PROGRESS":
        status["progress"] = task.info
    elif task.state == "SUCCESS":
        status["result"] = task.result
        status["model_id"] = f"{job['ticker']}_{job['model_type']}_{job_id}"
    elif task.state == "FAILURE":
        status["error"] = str(task.info)

    return status


@router.post("/predict", response_model=PredictionResponse)
async def predict_prices(request: PredictRequest):
    """
    Get price predictions from a trained model

    **Options:**
    - include_uncertainty: Add confidence intervals
    - include_attention: Return attention weights (LSTM only)
    """
    try:
        logger.info(f"Prediction request for {request.ticker}")

        # Load or get active model
        model_id = request.model_id or _get_active_model(request.ticker, request.model_type)

        if not model_id:
            raise HTTPException(
                status_code=404, detail=f"No active model found for {request.ticker}"
            )

        # Collect recent data for prediction
        collector = YFinanceCollector()
        data = await collector.collect_with_retry(
            ticker=request.ticker,
            start_date=datetime.now() - timedelta(days=90),
            end_date=datetime.now(),
        )

        if data is None:
            raise HTTPException(
                status_code=404, detail=f"Could not fetch data for {request.ticker}"
            )

        # Engineer features
        fe = FeatureEngineer()
        enriched_data = fe.create_all_features(data)

        # Load model and predict
        model, trainer = _load_model(model_id)

        # Prepare input sequence
        feature_columns = _get_feature_columns(model_id)
        features = enriched_data.select(feature_columns).to_numpy()

        # Take last sequence_length points
        sequence_length = model.sequence_length if hasattr(model, "sequence_length") else 60
        input_sequence = torch.FloatTensor(features[-sequence_length:])

        # Predict
        predictions = trainer.predict(input_sequence)

        # Format predictions
        current_price = float(data["close"].tail(1).item())
        pred_prices = predictions["price"][0].tolist()

        formatted_predictions = []
        for i, pred_price in enumerate(pred_prices[: request.days_ahead]):
            pred_dict = {
                "day": i + 1,
                "predicted_price": float(pred_price),
                "change": float(pred_price - current_price),
                "change_percent": float((pred_price - current_price) / current_price * 100),
            }

            if request.include_uncertainty:
                pred_dict["confidence_lower"] = float(predictions["price_lower"][0][i])
                pred_dict["confidence_upper"] = float(predictions["price_upper"][0][i])

            formatted_predictions.append(pred_dict)

        response = PredictionResponse(
            ticker=request.ticker,
            model_id=model_id,
            model_type=request.model_type,
            prediction_time=datetime.now(),
            predictions=formatted_predictions,
        )

        # Add optional fields
        if request.include_uncertainty:
            response.uncertainty = {
                "mean_uncertainty": float(predictions["uncertainty"][0].mean()),
                "max_uncertainty": float(predictions["uncertainty"][0].max()),
            }

        if request.include_attention and "attention_weights" in predictions:
            response.attention_weights = predictions["attention_weights"][0].tolist()

        if "regime_probs" in predictions:
            regime_names = ["bear", "sideways", "bull"]
            response.regime_probabilities = {
                regime_names[i]: float(prob)
                for i, prob in enumerate(predictions["regime_probs"][0])
            }

        return response

    except Exception as e:
        logger.error(f"Error in prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/models", response_model=ModelListResponse)
async def list_models(
    ticker: str | None = None,
    model_type: str | None = None,
    is_active: bool | None = None,
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
    db: Annotated[AsyncSession, Depends(get_async_session)] = "default",
):
    """
    List available models with filtering
    """
    try:
        # Query from database
        from db.models import Model

        query = select(Model)

        # Apply filters
        if ticker:
            query = query.where(Model.ticker == ticker)
        if model_type:
            query = query.where(Model.model_type == model_type)
        if is_active is not None:
            query = query.where(Model.is_active == is_active)

        # Apply pagination
        query = query.offset(offset).limit(limit).order_by(Model.trained_on.desc())

        result = await db.execute(query)
        models = result.scalars().all()

        # Count total
        from sqlalchemy import func

        count_query = select(func.count(Model.model_id))
        if ticker:
            count_query = count_query.where(Model.ticker == ticker)
        if model_type:
            count_query = count_query.where(Model.model_type == model_type)
        if is_active is not None:
            count_query = count_query.where(Model.is_active == is_active)

        total_result = await db.execute(count_query)
        total = total_result.scalar()

        # Format response
        models_list = [
            {
                "model_id": str(m.model_id),
                "model_name": m.model_name,
                "model_type": m.model_type,
                "ticker": m.ticker,
                "version": m.version,
                "trained_on": m.trained_on.isoformat(),
                "is_active": m.is_active,
                "mae": m.mae,
                "rmse": m.rmse,
                "r2_score": m.r2_score,
            }
            for m in models
        ]

        return ModelListResponse(models=models_list, total=total)

    except Exception as e:
        logger.error(f"Error listing models: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/models/{model_id}", response_model=ModelDetailsResponse)
async def get_model_details(
    model_id: str,
    db: Annotated[AsyncSession, Depends(get_async_session)],
):
    """
    Get detailed information about a specific model
    """
    try:
        from db.models import Model

        # Query from database
        query = select(Model).where(Model.model_id == model_id)
        result = await db.execute(query)
        model = result.scalar_one_or_none()

        if not model:
            raise HTTPException(status_code=404, detail="Model not found")

        return ModelDetailsResponse(
            model_id=str(model.model_id),
            model_name=model.model_name,
            model_type=model.model_type,
            ticker=model.ticker,
            trained_on=model.trained_on,
            hyperparameters=model.hyperparameters or {},
            performance={
                "mae": model.mae,
                "rmse": model.rmse,
                "r2_score": model.r2_score,
            },
            feature_importance=model.feature_importance,
            is_active=model.is_active,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching model details: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.delete("/models/{model_id}")
async def delete_model(
    model_id: str,
    db: Annotated[AsyncSession, Depends(get_async_session)],
):
    """
    Delete a model (soft delete - marks as inactive)
    """
    try:
        from db.models import Model

        # Query model
        query = select(Model).where(Model.model_id == model_id)
        result = await db.execute(query)
        model = result.scalar_one_or_none()

        if not model:
            raise HTTPException(status_code=404, detail="Model not found")

        # Soft delete
        model.is_active = False
        await db.commit()

        logger.info(f"Model {model_id} marked as inactive")

        return {"message": f"Model {model_id} deleted successfully"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting model: {e}")
        await db.rollback()
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/models/{model_id}/evaluate")
async def evaluate_model(model_id: str, request: ModelEvaluationRequest):
    """
    Evaluate model performance on test data
    """
    try:
        logger.info(f"Starting evaluation for model {model_id}")

        # Submit evaluation task
        task = evaluate_model_task.delay(
            model_id=model_id,
            test_start_date=request.test_start_date.isoformat(),
            test_end_date=request.test_end_date.isoformat(),
        )

        return {
            "message": "Evaluation started",
            "task_id": task.id,
            "model_id": model_id,
            "status": "queued",
        }

    except Exception as e:
        logger.error(f"Error evaluating model: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/features/importance/{model_id}")
async def get_feature_importance(
    model_id: str,
    num_samples: Annotated[int, Query(ge=10, le=500)] = 100,
):
    """
    Get feature importance for a trained model using SHAP
    """
    try:
        logger.info(f"Computing feature importance for model {model_id}")

        # Submit feature importance task
        task = compute_feature_importance_task.delay(
            model_id=model_id,
            num_samples=num_samples,
        )

        return {
            "message": "Feature importance computation started",
            "task_id": task.id,
            "model_id": model_id,
            "status": "queued",
        }

    except Exception as e:
        logger.error(f"Error computing feature importance: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


# Helper functions
async def _train_model_sync(job_id: str, request: TrainModelRequest):
    """
    Synchronous training (for background tasks)
    """
    try:
        training_jobs[job_id]["status"] = "training"

        # Collect data
        collector = YFinanceCollector()
        data = await collector.collect_with_retry(
            ticker=request.ticker,
            start_date=request.start_date,
            end_date=request.end_date,
        )

        # Engineer features
        fe = FeatureEngineer()
        enriched_data = fe.create_all_features(data)

        # Prepare dataset
        feature_columns = fe.get_all_feature_names()[: request.max_features]
        dataset = TimeSeriesDataset(
            data=enriched_data,
            feature_columns=feature_columns,
            sequence_length=request.sequence_length,
            prediction_horizon=request.prediction_horizon,
        )

        # Split dataset
        from torch.utils.data import DataLoader, random_split

        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        train_loader = DataLoader(train_dataset, batch_size=request.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=request.batch_size)

        # Train model
        model = AdvancedLSTM(
            input_dim=len(feature_columns),
            hidden_dim=request.hidden_dim,
            num_layers=request.num_layers,
            dropout=request.dropout,
            output_horizon=request.prediction_horizon,
        )

        trainer = LSTMTrainer(model)

        history = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=request.num_epochs,
            learning_rate=request.learning_rate,
            early_stopping_patience=request.early_stopping_patience,
        )

        # Save model
        model_id = f"{request.ticker}_{request.model_type}_{job_id}"
        model_path = f"{settings.MODEL_STORAGE_PATH}/{model_id}.pt"

        Path(settings.MODEL_STORAGE_PATH).mkdir(parents=True, exist_ok=True)
        trainer.save_checkpoint(model_path)

        # Save meta_data
        metadata = {
            "model_id": model_id,
            "feature_columns": feature_columns,
            "sequence_length": request.sequence_length,
            "prediction_horizon": request.prediction_horizon,
        }

        metadata_path = f"{settings.MODEL_STORAGE_PATH}/{model_id}_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f)

        training_jobs[job_id]["status"] = "completed"
        training_jobs[job_id]["model_id"] = model_id
        training_jobs[job_id]["metrics"] = {
            "train_loss": history["train_loss"][-1],
            "val_loss": history["val_loss"][-1],
        }

    except Exception as e:
        logger.error(f"Training error: {e}")
        training_jobs[job_id]["status"] = "failed"
        training_jobs[job_id]["error"] = str(e)


def _get_active_model(ticker: str, model_type: str) -> str | None:
    """Get active model ID for ticker"""
    try:
        # Look for most recent model file
        model_dir = Path(settings.MODEL_STORAGE_PATH)
        pattern = f"{ticker}_{model_type}_*.pt"

        model_files = list(model_dir.glob(pattern))
        if not model_files:
            return None

        # Get most recent
        most_recent = max(model_files, key=lambda p: p.stat().st_mtime)
        model_id = most_recent.stem

        logger.info(f"Found active model: {model_id}")
        return model_id

    except Exception as e:
        logger.error(f"Error finding active model: {e}")
        return None


def _load_model(model_id: str):
    """Load model from storage"""
    try:
        model_path = f"{settings.MODEL_STORAGE_PATH}/{model_id}.pt"
        metadata_path = f"{settings.MODEL_STORAGE_PATH}/{model_id}_metadata.json"

        # Load meta_data
        with open(metadata_path) as f:
            metadata = json.load(f)

        # Load model
        checkpoint = torch.load(model_path, map_location="cpu")

        # Reconstruct model
        model = AdvancedLSTM(
            input_dim=len(metadata["feature_columns"]),
            hidden_dim=128,  # Default, should be in meta_data
            num_layers=3,
            dropout=0.3,
            output_horizon=metadata["prediction_horizon"],
        )

        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        model.sequence_length = metadata["sequence_length"]

        trainer = LSTMTrainer(model)

        logger.info(f"Loaded model {model_id}")
        return model, trainer

    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}") from e


def _get_feature_columns(model_id: str) -> list[str]:
    """Get feature columns used by model"""
    try:
        metadata_path = f"{settings.MODEL_STORAGE_PATH}/{model_id}_metadata.json"

        with open(metadata_path) as f:
            metadata = json.load(f)

        return metadata["feature_columns"]

    except Exception as e:
        logger.error(f"Error loading feature columns: {e}")
        # Return default features
        fe = FeatureEngineer()
        return fe.get_all_feature_names()[:50]
