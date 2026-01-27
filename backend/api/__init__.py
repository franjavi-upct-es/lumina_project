# backend/api/__init__.py
"""
FastAPI Application Module for Lumina Quant Lab

This module provides the REST API interface for the platform, including:
- Data endpoints for market data retrieval
- ML endpoints for model training and prediction
- Backtest endpoints for strategy testing
- Portfolio endpoints for optimization
- Risk endpoints for risk analysis

Usage:
    # Import the FastAPI app
    from backend.api import app

    # Run with uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

    # Import specific routers
    from backend.api.routes import data, ml, backtest, portfolio, risk
"""

# Import dependencies for use in extensions
from backend.api.dependencies import (
    create_access_token,
    get_async_db,
    get_feature_engineer,
    get_redis,
    get_yfinance_collector,
)
from backend.api.main import app

# Import routers for external access
from backend.api.routes import backtest, data, ml, portfolio, risk

__all__ = [
    # Main application
    "app",
    # Routers
    "data",
    "ml",
    "backtest",
    "portfolio",
    "risk",
    # Dependencies
    "get_async_db",
    "get_redis",
    "get_yfinance_collector",
    "get_feature_engineer",
    "create_access_token",
]
