# backend/api/__init__.py
"""
FastAPI Application Module for Lumina Quant Lab V3

This module provides the REST API interface for the platform, including:
- Data endpoints for market data retrieval
- Agent endpoints for RL model monitoring and control
- Backtest endpoints for strategy testing
- Portfolio endpoints for optimization
- Risk endpoints for safety management and kill switches
- Monitoring endpoints for Prometheus metrics

V3 Enhancements:
- RL agent monitoring and control
- Safety arbitrator integration
- Real-time metrics and health checks
- Feature store access
- Multi-modal perception layer APIs

Usage:
    # Import the FastAPI app
    from backend.api import app

    # Run with uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

    # Import specific routers
    from backend.api.routes import agent, data, monitoring, risk
"""

# Import dependencies for external use
from backend.api.deps import (
    RateLimiter,
    check_database_health,
    check_production_environment,
    check_rate_limit,
    check_redis_health,
    create_access_token,
    decode_access_token,
    get_async_db,
    get_current_user,
    get_current_user_optional,
    get_feature_engineer,
    get_redis,
    get_yfinance_collector,
    require_development_environment,
    verify_api_key,
)

# Import main FastAPI app
from backend.api.main import app

# Import routers for external access
from backend.api.routes import (
    agent,
    backtest,
    data,
    monitoring,
    portfolio,
    risk,
)

__all__ = [
    # Main application
    "app",
    # Routers
    "agent",
    "backtest",
    "data",
    "monitoring",
    "portfolio",
    "risk",
    # Database dependencies
    "get_async_db",
    "check_database_health",
    # Cache dependencies
    "get_redis",
    "check_redis_health",
    # Data collector dependencies
    "get_yfinance_collector",
    "get_feature_engineer",
    # Authentication dependencies
    "create_access_token",
    "decode_access_token",
    "get_current_user",
    "get_current_user_optional",
    "verify_api_key",
    # Rate limiting
    "check_rate_limit",
    "RateLimiter",
    # Environment dependencies
    "check_production_environment",
    "require_development_environment",
]
