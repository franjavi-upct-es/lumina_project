# backend/api/main.py
"""
Lumina Quant Lab 2.0 - FastAPI Application
Main entry point for the API
"""

import logging
import sys
import uuid
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from redis import Redis
from sqlalchemy import text

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


from backend.api.routes import backtest, data, ml, portfolio, risk
from backend.config.logging_config import setup_logging
from backend.config.settings import get_settings
from backend.db.models import get_async_engine

# Setup logging
settings = get_settings()
setup_logging()
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Lumina Quant API",
    description="Advanced Quantitative Trading Platform API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS Configuration
allow_origins = settings.ALLOWED_ORIGINS
allow_credentials = settings.CORS_ALLOW_CREDENTIALS
if "*" in allow_origins and allow_credentials:
    logger.warning("CORS allow_origins '*' with credentials enabled; disabling credentials.")
    allow_credentials = False

app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=allow_credentials,
    allow_methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"],
    allow_headers=[
        "Authorization",
        "Content-Type",
        "Accept",
        "X-API-Key",
        "X-Request-ID",
    ],
    expose_headers=["X-Request-ID"],
)


@app.middleware("http")
async def add_request_id(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    request.state.request_id = request_id
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response


# Include routers - v1 routes (for backwards compatibility)
app.include_router(data.router, prefix="/api/v1/data", tags=["Data"])
app.include_router(ml.router, prefix="/api/v1/ml", tags=["Machine Learning"])
app.include_router(portfolio.router, prefix="/api/v1/portfolio", tags=["Portfolio"])
app.include_router(risk.router, prefix="/api/v1/risk", tags=["Risk Management"])
app.include_router(backtest.router, prefix="/api/v1/backtest", tags=["Backtesting"])

# TODO: v2 routes currently mirror v1. When v2 diverges, create separate
# router modules (e.g., backend/api/routes/v2/data.py) and import them here.
# For now, both versions point to the same handlers for backwards compatibility.
app.include_router(data.router, prefix="/api/v2/data", tags=["Data v2"])
app.include_router(ml.router, prefix="/api/v2/ml", tags=["Machine Learning v2"])
app.include_router(portfolio.router, prefix="/api/v2/portfolio", tags=["Portfolio v2"])
app.include_router(risk.router, prefix="/api/v2/risk", tags=["Risk Management v2"])
app.include_router(backtest.router, prefix="/api/v2/backtest", tags=["Backtesting v2"])


@app.get("/")
async def root():
    return {
        "message": "Lumina Quant API",
        "version": "2.0.0",
        "status": "operational",
    }


@app.get("/health", tags=["System"])
async def health_check():
    """
    Health check endpoint for container orchestration.
    Returns status of all critical dependencies.
    """
    health = {
        "status": "ok",
        "version": settings.VERSION,
        "environment": settings.ENVIRONMENT,
    }

    # Check database
    try:
        engine = get_async_engine()
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        health["database"] = "connected"
    except Exception as e:
        health["database"] = f"error: {type(e).__name__}"
        health["status"] = "degraded"

    # Check Redis
    redis_client: Redis | None = None
    try:
        redis_client = Redis.from_url(
            settings.REDIS_URL,
            decode_responses=True,
            socket_connect_timeout=2,
        )
        redis_client.ping()
        health["redis"] = "connected"
    except Exception as e:
        health["redis"] = f"error: {type(e).__name__}"
        health["status"] = "degraded"
    finally:
        if redis_client is not None:
            redis_client.close()

    status_code = 200 if health["status"] == "ok" else 503
    return JSONResponse(content=health, status_code=status_code)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    request_id = getattr(request.state, "request_id", "unknown")
    logger.error(
        f"Unhandled exception [request_id={request_id}]: {type(exc).__name__}: {exc}",
        exc_info=True,
    )
    return JSONResponse(
        status_code=500,
        content={
            "detail": "An internal server error occurred.",
            "request_id": request_id,
        },
    )
