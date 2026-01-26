# backend/api/main.py
"""
Lumina Quant Lab 2.0 - FastAPI Application
Main entry point for the API
"""

import logging
import sys
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
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers - v1 routes (for backwards compatibility)
app.include_router(data.router, prefix="/api/v1/data", tags=["Data"])
app.include_router(ml.router, prefix="/api/v1/ml", tags=["Machine Learning"])
app.include_router(portfolio.router, prefix="/api/v1/portfolio", tags=["Portfolio"])
app.include_router(risk.router, prefix="/api/v1/risk", tags=["Risk Management"])
app.include_router(backtest.router, prefix="/api/v1/backtest", tags=["Backtesting"])

# Include routers - v2 routes
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


@app.get("/health")
async def health_check():
    db_status = "disconnected"
    redis_status = "disconnected"
    errors: dict[str, str] = {}

    try:
        engine = get_async_engine()
        async with engine.connect() as connection:
            await connection.execute(text("SELECT 1"))
        db_status = "connected"
    except Exception as exc:
        errors["database"] = str(exc) if settings.DEBUG else "unavailable"
        logger.warning(f"Health check DB error: {exc}")

    redis_client: Redis | None = None
    try:
        redis_client = Redis.from_url(
            settings.REDIS_URL,
            decode_responses=True,
            socket_connect_timeout=5,
        )
        redis_client.ping()
        redis_status = "connected"
    except Exception as exc:
        errors["redis"] = str(exc) if settings.DEBUG else "unavailable"
        logger.warning(f"Health check Redis error: {exc}")
    finally:
        if redis_client is not None:
            redis_client.close()

    if db_status == "connected" and redis_status == "connected":
        overall_status = "healthy"
    elif db_status == "connected" or redis_status == "connected":
        overall_status = "degraded"
    else:
        overall_status = "unhealthy"

    response = {
        "status": overall_status,
        "database": db_status,
        "redis": redis_status,
    }

    if errors and settings.DEBUG:
        response["errors"] = errors

    status_code = 200 if overall_status == "healthy" else 503
    return JSONResponse(status_code=status_code, content=response)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Global exception: {exc}", exc_info=True)
    detail = "Internal server error"
    if settings.DEBUG:
        detail = f"{exc.__class__.__name__}: {exc}"
    return JSONResponse(status_code=500, content={"detail": detail})
