# backend/api/routes/__init__.py
"""
API Routes Module for Lumina Quant Lab

Contains all FastAPI router definitions organized by functionality:
- data: Market data collection and feature retrieval
- ml: Machine learning model training and predictions
- backtest: Strategy backtesting and analysis
- portfolio: Portfolio optimization and analytics
- risk: Risk metrics and stress testing

Usage:
    from backend.api.routes import data, ml, backtest, portfolio, risk

    # Include in FastAPI app
    app.include_router(data.router, prefix="/api/v2/data", tags=["Data"])
    app.include_router(ml.router, prefix="/api/v2/ml", tags=["ML"])
"""

from backend.api.routes import backtest, data, ml, portfolio, risk

# Export router instances directly
data_router = data.router
ml_router = ml.router
backtest_router = backtest.router
portfolio_router = portfolio.router
risk_router = risk.router

__all__ = [
    # Modules
    "data",
    "ml",
    "backtest",
    "portfolio",
    "risk",
    # Router instances
    "data_router",
    "ml_router",
    "backtest_router",
    "portfolio_router",
    "risk_router",
]
