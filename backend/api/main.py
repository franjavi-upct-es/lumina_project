# backend/api/main.py
"""
Lumina Quant Lab 2.0 - FastAPI Application
Main entry point for the API
"""

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from contextlib import asynccontextmanager
import logging
from datetime import datetime

from api.routes import data, ml, backtest, portfolio, risk
from config.settings import get_settings
from db.models import init_db

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifecycle management for the application
    """
    logger.info("🚀 Starting Lumina Quant Lab 2.0...")

    # Initialize database
    try:
        await init_db()
        logger.info("✅ Database initialized successfully")
    except Exception as e:
        logger.error(f"❌ Database initialization failed: {e}")
        raise

    # Startup tasks
    logger.info(f"📊 Environment: {settings.ENVIRONMENT}")
    logger.info(f"🔗 Database: {settings.DATABASE_URL.split('@')[1]}")  # Hide password
    logger.info(f"🔴 Redis: {settings.REDIS_URL.split('@')[1]}")

    yield

    # Shutdown tasks
    logger.info("🛑 Shutting down Lumina Quant Lab...")


# Create FastAPI app
app = FastAPI(
    title="Lumina Quant Lab API",
    description="Advanced Quantitative Trading Platform with AI/ML",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan,
)

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(GZipMiddleware, minimum_size=1000)


# Exception handlers
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """
    Handle validation errors with detailed messages
    """
    errors = []
    for error in exc.errors():
        errors.append(
            {
                "field": " -> ".join(str(x) for x in error["loc"]),
                "message": error["msg"],
                "type": error["type"],
            }
        )

    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "detail": "Validation Error",
            "errors": errors,
            "timestamp": datetime.utcnow().isoformat(),
        },
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """
    Catch-all exception handler
    """
    logger.error(f"Unhandled exception: {exc}", exc_info=True)

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "detail": "Internal Server Error",
            "message": str(exc)
            if settings.ENVIRONMENT == "development"
            else "An error occurred",
            "timestamp": datetime.utcnow().isoformat(),
        },
    )


# Health check endpoints
@app.get("/health", tags=["Health"])
async def health_check():
    """
    Health check endpoint for container orchestration
    """
    return {
        "status": "healthy",
        "version": "2.0.0",
        "timestamp": datetime.utcnow().isoformat(),
        "environment": settings.ENVIRONMENT,
    }


@app.get("/", tags=["Root"])
async def root():
    """
    Root endpoint with API information
    """
    return {
        "name": "Lumina Quant Lab API",
        "version": "2.0.0",
        "description": "Advanced Quantitative Trading Platform with AI/ML",
        "docs": "/docs",
        "redoc": "/redoc",
        "health": "/health",
        "endpoints": {
            "data": "/api/v2/data",
            "ml": "/api/v2/ml",
            "backtest": "/api/v2/backtest",
            "portfolio": "/api/v2/portfolio",
            "risk": "/api/v2/risk",
        },
    }


# Include routers
app.include_router(data.router, prefix="/api/v2/data", tags=["Data"])
app.include_router(ml.router, prefix="/api/v2/ml", tags=["Machine Learning"])
app.include_router(backtest.router, prefix="/api/v2/backtest", tags=["Backtesting"])
app.include_router(portfolio.router, prefix="/api/v2/portfolio", tags=["Portfolio"])
app.include_router(risk.router, prefix="/api/v2/risk", tags=["Risk Analysis"])


# Startup event
@app.on_event("startup")
async def startup_event():
    """
    Additional startup tasks
    """
    logger.info("🎯 All routes registered successfully")
    logger.info("📡 API is ready at http://0.0.0.0:8000")
    logger.info("📚 Docs available at http://0.0.0.0:8000/docs")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.ENVIRONMENT == "development",
        log_level="info",
    )
