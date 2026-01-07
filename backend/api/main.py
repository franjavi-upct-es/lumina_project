# backend/api/main.py
"""
Lumina Quant Lab 2.0 - FastAPI Application
Main entry point for the API
"""

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import settings
from config.logging_config import setup_logging
from api.routes import data, ml, portfolio, risk, backtest

# Setup logging
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
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(data.router, prefix="/api/v1/data", tags=["Data"])
app.include_router(ml.router, prefix="/api/v1/ml", tags=["Machine Learning"])
app.include_router(portfolio.router, prefix="/api/v1/portfolio", tags=["Portfolio"])
app.include_router(risk.router, prefix="/api/v1/risk", tags=["Risk Management"])
app.include_router(backtest.router, prefix="/api/v1/backtest", tags=["Backtesting"])


@app.get("/")
async def root():
    return {
        "message": "Lumina Quant API",
        "version": "1.0.0",
        "status": "operational",
    }


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "database": "connected",  # Add actual DB check
        "redis": "connected",  # Add actual Redis check
    }


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Global exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500, content={"detail": "Internal server error"}
    )
