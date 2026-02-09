# backend/api/routes/__init__.py
"""
API Routes Module for Lumina Quant Lab V3

Contains all FastAPI router definitions organized by functionality:

V3 Routes:
- agent: RL agent monitoring, control, and training management
- monitoring: Prometheus metrics, system health, performance tracking
- risk: Safety arbitrator, circuit breakers, kill switches

Maintained Routes:
- data: Market data collection and feature retrieval
- backtest: Strategy backtesting and analysis
- portfolio: Portfolio optimization and analytics

Usage:
    from backend.api.routes import agent, data, monitoring, risk

    # Include in FastAPI app
    app.include_router(agent.router, prefix="/api/v3/agent", tags=["Agent"])
    app.include_router(monitoring.router, prefix="/api/v3/monitoring", tags=["Monitoring"])
"""

from backend.api.routes import (
    agent,
    backtest,
    data,
    monitoring,
    portfolio,
    risk,
)

# Export router instances directly
agent_router = agent.router
backtest_router = backtest.router
data_router = data.router
monitoring_router = monitoring.router
portfolio_router = portfolio.router
risk_router = risk.router

__all__ = [
    # Modules
    "agent",
    "backtest",
    "data",
    "monitoring",
    "portfolio",
    "risk",
    # Router instances
    "agent_router",
    "backtest_router",
    "data_router",
    "monitoring_router",
    "portfolio_router",
    "risk_router",
]
