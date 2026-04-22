# backend/api/routes/ml.py
"""
Machine Learning endpoints for model training, prediction, and evaluation
"""

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Annotated, Any, cast
from uuid import UUID, uuid4

import torch
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query
from loguru import logger
from pydantic import BaseModel, Field, model_validator
from redis import Redis
from sqlalchemy import select
from sqlalchemy.exc import OperationalError, SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from backend.api.dependencies import check_rate_limit, get_redis, verify_api_key
from backend.api.job_store import JobStore, format_job_timestamp
from backend.config.settings import get_settings
from backend.data_engine.collectors.yfinance_collector import YFinanceCollector
from backend.data_engine.transformers.feature_engineering import FeatureEngineer
from backend.db.models import get_async_session
from backend.ml_engine.models.lstm_advanced import AdvancedLSTM, LSTMTrainer, TimeSeriesDataset
from backend.workers.ml_tasks import (
    train_model_task,
    train_xgboost_task,
)

router = APIRouter(dependencies=[Depends(check_rate_limit), Depends(verify_api_key)])
settings = get_settings()


def _get_training_job_store(redis: Redis = Depends(get_redis)) -> JobStore:
    return JobStore(redis, prefix="training")


def _is_valid_uuid(value: str) -> bool:
    try:
        UUID(str(value))
        return True
    except ValueError:
        return False


def _sort_market_data(data):
    for candidate in ("time", "date", "datetime"):
        if candidate in data.columns:
            return data.sort(candidate)
    return data


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
    learning_rate: float = Field(0.001, ge=0.0001, le=1.0)
    early_stopping_patience: int = Field(10, ge=3, le=30)

    # Feature selection
    max_features: int = Field(50, ge=10, le=200)
    feature_categories: list[str] | None = None

    # XGBoost hyperparameters
    n_estimators: int | None = Field(None, ge=10, le=5000)
    max_depth: int | None = Field(None, ge=1, le=15)
    subsample: float | None = Field(None, gt=0.0, le=1.0)
    colsample_bytree: float | None = Field(None, gt=0.0, le=1.0)
    reg_alpha: float | None = Field(None, ge=0.0)
    reg_lambda: float | None = Field(None, ge=0.0)
    min_child_weight: float | None = Field(None, ge=0.0)
    random_state: int | None = None
    n_jobs: int | None = None

    # Options
    async_training: bool = True
    save_model: bool = True

    @model_validator(mode="after")
    def validate_model_specific_fields(self) -> "TrainModelRequest":
        if self.model_type != "xgboost" and self.learning_rate > 0.01:
            raise ValueError("learning_rate must be less than or equal to 0.01 for neural models")
        return self


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


def _build_training_hyperparams(request: TrainModelRequest) -> dict[str, Any]:
    if request.model_type == "xgboost":
        hyperparams: dict[str, Any] = {}
        for field_name in (
            "n_estimators",
            "max_depth",
            "learning_rate",
            "subsample",
            "colsample_bytree",
            "reg_alpha",
            "reg_lambda",
            "min_child_weight",
            "random_state",
            "n_jobs",
        ):
            value = getattr(request, field_name)
            if value is not None:
                hyperparams[field_name] = value
        return hyperparams

    return {
        "start_date": request.start_date,
        "end_date": request.end_date,
        "hidden_dim": request.hidden_dim,
        "num_layers": request.num_layers,
        "dropout": request.dropout,
        "sequence_length": request.sequence_length,
        "prediction_horizon": request.prediction_horizon,
        "batch_size": request.batch_size,
        "num_epochs": request.num_epochs,
        "learning_rate": request.learning_rate,
        "early_stopping_patience": request.early_stopping_patience,
        "max_features": request.max_features,
    }


def _resolve_xgboost_lookback_days(request: TrainModelRequest) -> int:
    if request.start_date is None and request.end_date is None:
        return 500

    end_date = request.end_date or datetime.now()
    start_date = request.start_date or (end_date - timedelta(days=500))
    delta_days = int((end_date - start_date).total_seconds() // 86400)
    return max(delta_days, 30)


def _extract_completed_model_id(task_result: Any) -> str | None:
    if not isinstance(task_result, dict):
        return None
    for key in ("db_model_id", "model_id"):
        value = task_result.get(key)
        if value:
            return str(value)
    return None


@router.post("/train", response_model=TrainJobResponse)
async def train_model(
    request: TrainModelRequest,
    background_tasks: BackgroundTasks,
    job_store: JobStore = Depends(_get_training_job_store),
):
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
    job_id = str(uuid4())

    logger.info(f"Received training request for {request.ticker} ({request.model_type})")

    if request.async_training:
        try:
            # Route to appropriate task based on model type
            if request.model_type == "xgboost":
                task = train_xgboost_task.delay(
                    ticker=request.ticker,
                    hyperparams=_build_training_hyperparams(request),
                    lookback_days=_resolve_xgboost_lookback_days(request),
                    prediction_horizon=request.prediction_horizon,
                )
            else:
                task = train_model_task.delay(
                    job_id=job_id,
                    ticker=request.ticker,
                    model_type=request.model_type,
                    hyperparams=_build_training_hyperparams(request),
                )

            job_store.set(
                job_id,
                {
                    "task_id": task.id,
                    "ticker": request.ticker,
                    "model_type": request.model_type,
                    "status": "queued",
                    "created_at": datetime.now(),
                },
            )

            return TrainJobResponse(
                job_id=job_id,
                ticker=request.ticker,
                model_type=request.model_type,
                status="queued",
                message="Training job submitted. Check /ml/jobs/{job_id} for status.",
                estimated_time_minutes=10 if request.model_type == "lstm" else 5,
            )
        except Exception as e:
            # Check if it's a connection error (Celery broker unavailable)
            error_str = str(e).lower()
            if (
                "connection refused" in error_str
                or "connection" in error_str
                or "refused" in error_str
            ):
                # Celery broker not available, fall back to background task
                logger.warning(f"Celery broker unavailable, falling back to background task: {e}")
                background_tasks.add_task(_train_model_sync, job_id, request)

                job_store.set(
                    job_id,
                    {
                        "task_id": job_id,
                        "ticker": request.ticker,
                        "model_type": request.model_type,
                        "status": "training",
                        "created_at": datetime.now(),
                    },
                )

                return TrainJobResponse(
                    job_id=job_id,
                    ticker=request.ticker,
                    model_type=request.model_type,
                    status="training",
                    message="Training started in background (sync mode)",
                    estimated_time_minutes=10 if request.model_type == "lstm" else 5,
                )
            else:
                logger.error(f"Error initiating training: {e}")
                raise HTTPException(status_code=500, detail=str(e)) from e
    else:
        # Synchronous training (not recommended for production)
        background_tasks.add_task(_train_model_sync, job_id, request)

        job_store.set(
            job_id,
            {
                "task_id": job_id,
                "ticker": request.ticker,
                "model_type": request.model_type,
                "status": "training",
                "created_at": datetime.now(),
            },
        )

        return TrainJobResponse(
            job_id=job_id,
            ticker=request.ticker,
            model_type=request.model_type,
            status="training",
            message="Training started in background",
            estimated_time_minutes=10,
        )


@router.get("/jobs/{job_id}")
async def get_training_job_status(
    job_id: str,
    job_store: JobStore = Depends(_get_training_job_store),
):
    """
    Get status of a training job
    """
    job = job_store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found or expired")

    # Check Celery task status
    from backend.workers.celery_app import celery_app

    try:
        task = celery_app.AsyncResult(job["task_id"])
        task_state = task.state
        task_info = task.info
    except Exception as e:
        logger.warning(f"Failed to read Celery status for job {job_id}: {e}")
        return {
            "job_id": job_id,
            "ticker": job["ticker"],
            "model_type": job["model_type"],
            "status": job.get("status", "unknown"),
            "created_at": format_job_timestamp(job.get("created_at")),
            "error": f"Could not read task result: {e}",
        }

    status = {
        "job_id": job_id,
        "ticker": job["ticker"],
        "model_type": job["model_type"],
        "status": task_state,
        "created_at": format_job_timestamp(job.get("created_at")),
    }

    # Add progress if available
    if task_state == "PROGRESS":
        status["progress"] = task_info
    elif task_state == "SUCCESS":
        status["result"] = task.result
        model_id = _extract_completed_model_id(task.result)
        if model_id is not None:
            status["model_id"] = model_id
    elif task_state == "FAILURE":
        status["error"] = str(task_info)

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

        data = _sort_market_data(data)

        # Engineer features
        fe = FeatureEngineer()
        enriched_data = fe.create_all_features(data)

        # Load model and predict
        model, trainer = _load_model(model_id)
        metadata = _load_model_metadata(model_id)

        # Prepare input sequence
        feature_columns = _get_feature_columns(model_id)
        features = TimeSeriesDataset.transform_features(
            enriched_data,
            feature_columns,
            feature_scaler=metadata.get("feature_scaler"),
        )

        # Take last sequence_length points
        sequence_length = metadata.get(
            "sequence_length",
            model.sequence_length if hasattr(model, "sequence_length") else 60,
        )
        if len(features) < sequence_length:
            raise HTTPException(
                status_code=400,
                detail=f"Not enough data for prediction. Need {sequence_length}, got {len(features)}",
            )
        input_sequence = torch.FloatTensor(features[-sequence_length:])

        # Predict
        current_price = float(data["close"].tail(1).item())
        trainer.prediction_context = {
            "last_close": current_price,
            "target_mode": metadata.get("target_mode", "price"),
        }
        predictions = trainer.predict(input_sequence)

        # Format predictions
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

    except HTTPException:
        raise
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
):
    """
    List available models with filtering
    """
    try:
        # Try to get database session
        from sqlalchemy import func

        from backend.db.models import Model, get_async_engine

        engine = get_async_engine()
        async with AsyncSession(engine) as db:
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

            return ModelListResponse(models=models_list, total=total or 0)

    except OperationalError as e:
        logger.error(f"Database connection failed: {e}")
        raise HTTPException(
            status_code=503,
            detail="Database is temporarily unavailable. Please try again later.",
        ) from e
    except SQLAlchemyError as e:
        logger.error(f"Database query failed: {e}")
        raise HTTPException(
            status_code=500,
            detail="An internal database error occurred.",
        ) from e


@router.get("/models/{model_id}", response_model=ModelDetailsResponse)
async def get_model_details(model_id: str):
    """
    Get detailed information about a specific model
    """
    try:
        from backend.db.models import Model, get_async_engine

        if not _is_valid_uuid(model_id):
            raise HTTPException(status_code=404, detail="Model not found")

        engine = get_async_engine()
        async with AsyncSession(engine) as db:
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
                ticker=model.ticker,  # type: ignore
                trained_on=model.trained_on,
                hyperparameters=model.hyperparameters or {},
                performance={
                    "mae": model.mae,  # type: ignore
                    "rmse": model.rmse,  # type: ignore
                    "r2_score": model.r2_score,  # type: ignore
                },
                feature_importance=model.feature_importance,
                is_active=model.is_active,
            )

    except HTTPException:
        raise
    except OperationalError as e:
        logger.error(f"Database connection failed: {e}")
        raise HTTPException(
            status_code=503,
            detail="Database is temporarily unavailable. Please try again later.",
        ) from e
    except SQLAlchemyError as e:
        logger.error(f"Database query failed: {e}")
        raise HTTPException(
            status_code=500,
            detail="An internal database error occurred.",
        ) from e  # type: ignore


@router.delete("/models/{model_id}")
async def delete_model(  # type: ignore
    model_id: str,  # type: ignore
    db: Annotated[AsyncSession, Depends(get_async_session)],  # type: ignore
):
    """
    Delete a model (soft delete - marks as inactive)
    """
    try:
        from backend.db.models import Model

        if not _is_valid_uuid(model_id):
            raise HTTPException(status_code=404, detail="Model not found")

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
    raise HTTPException(status_code=501, detail="Not implemented")


@router.get("/features/importance/{model_id}")
async def get_feature_importance(
    model_id: str,
    num_samples: Annotated[int, Query(ge=10, le=500)] = 100,
):
    """
    Get feature importance for a trained model using SHAP
    """
    raise HTTPException(status_code=501, detail="Not implemented")


# Helper functions
async def _train_model_sync(job_id: str, request: TrainModelRequest):
    """
    Synchronous training (for background tasks)
    """
    redis_client = Redis.from_url(
        settings.REDIS_URL, decode_responses=True, socket_connect_timeout=5
    )
    bg_job_store = JobStore(redis_client, prefix="training")
    try:
        bg_job_store.update(job_id, {"status": "training"})

        if request.model_type == "xgboost":
            result = train_xgboost_task.run(
                ticker=request.ticker,
                hyperparams=_build_training_hyperparams(request),
                lookback_days=_resolve_xgboost_lookback_days(request),
                prediction_horizon=request.prediction_horizon,
            )
            bg_job_store.update(
                job_id,
                {
                    "status": "completed",
                    "model_id": _extract_completed_model_id(result),
                    "result": result,
                },
            )
            return

        # Collect data
        collector = YFinanceCollector()
        data = await collector.collect_with_retry(
            ticker=request.ticker,
            start_date=request.start_date,
            end_date=request.end_date,
        )
        data = _sort_market_data(data)

        # Engineer features
        fe = FeatureEngineer()
        enriched_data = fe.create_all_features(data)  # type: ignore

        # Prepare dataset
        feature_columns = [
            column
            for column in fe.get_all_feature_names()[: request.max_features]
            if column in enriched_data.columns
        ]
        if not feature_columns:
            raise ValueError("No valid feature columns generated for LSTM training")
        target_mode = "relative_returns"
        feature_scaler = TimeSeriesDataset.build_feature_scaler(enriched_data, feature_columns)
        dataset = TimeSeriesDataset(
            data=enriched_data,
            feature_columns=feature_columns,
            sequence_length=request.sequence_length,
            prediction_horizon=request.prediction_horizon,
            feature_scaler=feature_scaler,
            target_mode=target_mode,
        )

        # Split dataset
        from torch.utils.data import DataLoader, Subset

        if len(dataset) < 2:
            raise ValueError(
                f"Not enough sequences for LSTM training. Need at least 2, got {len(dataset)}."
            )

        train_size = int(0.8 * len(dataset))
        train_size = min(max(train_size, 1), len(dataset) - 1)
        train_dataset = Subset(dataset, range(0, train_size))
        val_dataset = Subset(dataset, range(train_size, len(dataset)))

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
        trainer.target_mode = target_mode

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

        # Save meta_data
        metadata = {
            "model_id": model_id,
            "input_dim": len(feature_columns),
            "hidden_dim": request.hidden_dim,
            "num_layers": request.num_layers,
            "dropout": request.dropout,
            "feature_columns": feature_columns,
            "feature_scaler": feature_scaler,
            "sequence_length": request.sequence_length,
            "prediction_horizon": request.prediction_horizon,
            "target_mode": target_mode,
            "ticker": request.ticker,
            "trained_at": datetime.now().isoformat(),
        }

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "meta_data": metadata,
                "history": history,
            },
            model_path,
        )

        metadata_path = f"{settings.MODEL_STORAGE_PATH}/{model_id}_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f)

        bg_job_store.update(
            job_id,
            {
                "status": "completed",
                "model_id": model_id,
                "metrics": {
                    "train_loss": history["train_loss"][-1],
                    "val_loss": history["val_loss"][-1],
                },
            },
        )

    except Exception as e:
        logger.error(f"Training error: {e}")
        bg_job_store.update(job_id, {"status": "failed", "error": str(e)})
    finally:
        redis_client.close()


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


def _load_model_metadata(model_id: str) -> dict[str, Any]:
    model_path = Path(settings.MODEL_STORAGE_PATH) / f"{model_id}.pt"
    metadata_path = Path(settings.MODEL_STORAGE_PATH) / f"{model_id}_metadata.json"

    if metadata_path.exists():
        with open(metadata_path) as f:
            loaded_metadata = json.load(f)
            if isinstance(loaded_metadata, dict):
                return cast(dict[str, Any], loaded_metadata)

    if model_path.exists():
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
        metadata = checkpoint.get("meta_data", {})
        if isinstance(metadata, dict) and metadata:
            return cast(dict[str, Any], metadata)

    raise FileNotFoundError(f"Metadata not found for model {model_id}")


def _load_model(model_id: str):
    """Load model from storage"""
    try:
        model_path = f"{settings.MODEL_STORAGE_PATH}/{model_id}.pt"
        metadata = _load_model_metadata(model_id)

        # Load model
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
        feature_columns = metadata.get("feature_columns", [])
        input_dim = metadata.get("input_dim", len(feature_columns) or 50)

        # Reconstruct model
        model = AdvancedLSTM(
            input_dim=input_dim,
            hidden_dim=metadata.get("hidden_dim", 128),
            num_layers=metadata.get("num_layers", 3),
            dropout=metadata.get("dropout", 0.3),
            output_horizon=metadata.get("prediction_horizon", 5),
        )

        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        model.sequence_length = metadata.get("sequence_length", 60)

        trainer = LSTMTrainer(model)
        trainer.target_mode = metadata.get("target_mode", "price")

        logger.info(f"Loaded model {model_id}")
        return model, trainer

    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}") from e


def _get_feature_columns(model_id: str) -> list[str]:
    """Get feature columns used by model"""
    try:
        metadata = _load_model_metadata(model_id)
        feature_columns = metadata.get("feature_columns")
        if isinstance(feature_columns, list):
            return [str(column) for column in feature_columns]

    except Exception as e:
        logger.error(f"Error loading feature columns: {e}")

    # Return default features
    fe = FeatureEngineer()
    return fe.get_all_feature_names()[:50]
