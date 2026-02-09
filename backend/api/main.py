# backend/api/main.py
"""
Lumina Quant Lab V3 - FastAPI Application
Main entry point for the API

V3 Enhancements:
- RL Agent monitoring and control routes
- Safety arbitrator integration
- Prometheus metrics endpoint
- Feature store health checks
- Multi-layer safety system monitoring

Architecture:
- RESTful API design
- CORS middleware for frontend integration
- Comprehensive health checks (DB, Redis, Celery, GPU)
- Structured logging with Loguru
- API versioning (v3 for new features)
"""

import sys
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger
from redis import Redis
from sqlalchemy import text

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.api.routes import agent, backtest, data, monitoring, portfolio, risk
from backend.config.logging_config import setup_logging
from backend.config.settings import get_settings
from backend.db.models import get_async_engine

# Setup logging
settings = get_settings()
setup_logging()

# Create FastAPI application
app = FastAPI(
    title="Lumina Quant Lab V3 API",
    description="""
    Advanced Qauntitative Trading Platform with Deep Reinforcement Learning

    Features:
    - Multi-modal market perception (TFT, BERT, GNN)
    - Deep sensor fusion architecture
    - Proximal Policy Optimization (PPO) agent
    - Three-layer safety system (Risk Gate, Circuit Breakers, Safety Arbitrator)
    - Real-time monitoring and metrics
    - Event-driven backtesting
    - Portfolio optimization

    V3 Architecture:
    - Perception Layer: Temporal, Semantic, Structural encoders
    - Fusion Layer: Cross-modal attention
    - Cognition Layer: RL agent with uncertainty estimation
    - Execution Layer: Safety arbitrator and broker integration
    """,
    version="3.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS Configuration
allow_origins = settings.ALLOWED_ORIGINS
allow_credentials = settings.CORS_ALLOW_CREDENTIALS

# Security check for CORS
if "*" in allow_origins and allow_credentials:
    logger.warning(
        "CORS allow_origins '*' with credentials enabled; disabling credentials for security."
    )
    allow_credentials = False

app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=allow_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# V3 Routes
# ============================================================================

app.include_router(
    agent.router,
    prefix="/api/v3/agent",
    tags=["V3 - RL Agent"],
)

app.include_router(
    monitoring.router,
    prefix="/api/v3/monitoring",
    tags=["V3 - Monitoring"],
)

app.include_router(
    risk.router,
    prefix="/api/v3/risk",
    tags=["V3 - Risk & Safety"],
)

app.include_router(
    data.router,
    prefix="/api/v3/data",
    tags=["V3 - Data"],
)

app.include_router(
    portfolio.router,
    prefix="/api/v3/portfolio",
    tags=["V3 - Portfolio"],
)

app.include_router(
    backtest.router,
    prefix="/api/v3/backtest",
    tags=["V3 - Backtesting"],
)

# ============================================================================
# Root and Health Endpoints
# ============================================================================


@app.get("/", tags=["Root"])
async def root():
    """
    Root endpoint with API information.

    Returns:
        dict: API metadata and version information
    """
    return {
        "name": "Lumina Quant Lab V3",
        "version": "3.0.0",
        "description": "Deep Reinforcement Learning Trading Platform",
        "architecture": "Chimera - Multi-Modal Perception + Deep Fusion",
        "status": "operational",
        "timestamp": datetime.utcnow().isoformat(),
        "documentation": {
            "swagger": "/docs",
            "redoc": "/redoc",
        },
        "endpoints": {
            "v3": {
                "agent": "/api/v3/agent",
                "monitoring": "/api/v3/monitoring",
                "risk": "/api/v3/risk",
                "data": "/api/v3/data",
                "portfolio": "/api/v3/portfolio",
                "backtest": "/api/v3/backtest",
            }
        },
    }


@app.get("/health", tags=["Health"])
async def health_check():
    """
    Comprehensive health check endpoint.

    Checks:
    - Database connectivity (TimescaleDB)
    - Redis connectivity (feature store)
    - Celery worker status
    - GPU availability (if configured)

    Returns:
        dict: Health status of all components
    """
    health_status = {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "components": {},
    }

    # Check database
    try:
        engine = get_async_engine()
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        health_status["components"]["database"] = {"status": "healthy", "type": "TimescaleDB"}
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        health_status["components"]["database"] = {"status": "unhealthy", "error": str(e)}
        health_status["status"] = "degraded"

    # Check Redis
    try:
        redis_client = Redis(
            host=settings.REDIS_HOST,
            port=settings.REDIS_PORT,
            db=settings.REDIS_DB,
            decode_responses=True,
        )
        redis_client.ping()
        redis_client.close()
        health_status["components"]["redis"] = {
            "status": "healthy",
            "type": "Feature Store & Cache",
        }
    except Exception as e:
        logger.error(f"Redis health check failed: {e}")
        health_status["components"]["redis"] = {
            "status": "unhealthy",
            "error": str(e),
        }
        health_status["status"] = "degraded"

    # Check Celery workers
    try:
        from backend.workers.celery_app import celery_app

        # Get active workers
        inspect = celery_app.control.inspect()
        stats = inspect.stats()

        if stats:
            health_status["components"]["celery"] = {
                "status": "healthy",
                "workers": len(stats),
                "worker_names": list(stats.keys()),
            }
        else:
            health_status["components"]["celery"] = {
                "status": "unhealthy",
                "error": "No active workers",
            }
            health_status["status"] = "degraded"
    except Exception as e:
        logger.error(f"Celery health check failed: {e}")
        health_status["components"]["celery"] = {
            "status": "unknown",
            "error": str(e),
        }

    # Check GPU availability (optional)
    try:
        import torch

        if torch.cuda.is_available():
            health_status["components"]["gpu"] = {
                "status": "available",
                "device_count": torch.cuda.device_count(),
                "device_name": torch.cuda.get_device_name(0)
                if torch.cuda.device_count() > 0
                else None,
            }
        else:
            health_status["components"]["gpu"] = {
                "status": "not_available",
                "message": "Training will use CPU",
            }
    except Exception as e:
        health_status["components"]["gpu"] = {
            "status": "unknown",
            "error": str(e),
        }

    return health_status


@app.get("/health/ready", tags=["Health"])
async def readines_check():
    """
    Kubernetes readiness probe endpoint.

    Returns 200 if service is ready to accept traffic.
    Returns 503 if service is not ready.

    Returns:
        dict: Readiness status
    """
    try:
        # Check critical services
        engine = get_async_engine()
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))

        redis_client = Redis(
            host=settings.REDIS_HOST,
            port=settings.REDIS_PORT,
            db=settings.REDIS_DB,
        )
        redis_client.ping()
        redis_client.close()

        return {
            "status": "ready",
            "timestamp": datetime.utcnow().isoformat(),
        }
    except Exception as e:
        logger.error(f"Readiness check failed: {e}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "not_ready",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat(),
            },
        )


@app.get("/health/live", tags=["Health"])
async def liveness_check():
    """
    Kubernetes liveness probe endpoint.

    Returns 200 if service is alive.

    Returns:
        dict: Liveness status
    """
    return {
        "status": "alive",
        "timestamp": datetime.utcnow().isoformat(),
    }


# ============================================================================
# Error Handlers
# ============================================================================


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """
    Global exception handler for unhandled errors.

    Args:
        request: FastAPI request
        exc: Exception raised

    Returns:
        JSONResponse: Error response
    """
    logger.error(f"Unhandled exception: {exc}", exc_info=True)

    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc) if not settings.is_production else "An error occurred",
            "timestamp": datetime.utcnow().isoformat(),
            "path": str(request.url),
        },
    )


# ============================================================================
# Startup and Shutdown Events
# ============================================================================


@app.on_event("startup")
async def startup_event():
    """
    Startup event handler.

    Performs initialization tasks:
    - Log application start
    - Verify database connection
    - Verify Redis connection
    - Initialize monitoring
    """
    logger.info("=" * 70)
    logger.info("🚀 Lumina Quant Lab V3 API Starting")
    logger.info("=" * 70)
    logger.info(f"Environment: {settings.ENVIRONMENT}")
    logger.info(f"Debug Mode: {settings.DEBUG}")
    logger.info(f"API Version: 3.0.0")
    logger.info(f"Python Path: {sys.executable}")
    logger.info("=" * 70)

    # Verify database connection
    try:
        engine = get_async_engine()
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        logger.info("✅ Database connection verified")
    except Exception as e:
        logger.error(f"❌ Database connection failed: {e}")

    # Verify Redis connection
    try:
        redis_client = Redis(
            host=settings.REDIS_HOST,
            port=settings.REDIS_PORT,
            db=settings.REDIS_DB,
        )
        redis_client.ping()
        redis_client.close()
        logger.info("✅ Redis connection verified")
    except Exception as e:
        logger.error(f"❌ Redis connection failed: {e}")

    logger.info("=" * 70)
    logger.info("🎯 API ready to accept requests")
    logger.info("=" * 70)


@app.on_event("shutdown")
async def shutdown_event():
    """
    Shutdown event handler.

    Performs cleanup tasks:
    - Log application shutdown
    - Close database connections
    - Close Redis connections
    """
    logger.info("=" * 70)
    logger.info("🛑 Lumina Quant Lab V3 API Shutting Down")
    logger.info("=" * 70)

    # Close database connections
    try:
        engine = get_async_engine()
        await engine.dispose()
        logger.info("✅ Database connections closed")
    except Exception as e:
        logger.error(f"❌ Error closing database: {e}")

    logger.info("=" * 70)
    logger.info("👋 Shutdown complete")
    logger.info("=" * 70)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "backend.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG,
        log_level="info",
    )
