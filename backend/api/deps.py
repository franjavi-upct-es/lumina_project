# backend/api/deps.py
"""
FastAPI Dependencies for Dependency Injection
Provides common dependencies like database sessions, authentication, rate limiting,
and data collectors for the Lumina V3 API.

This module implements the dependency injection pattern for:
- Database sessions (PostgreSQL/TimescaleDB)
- Redis cache connections
- API key verification
- JWT token authentication
- Rate limiting
- Data collectors (YFinance, etc.)
- Feature engineering pipelines
"""

import hashlib
import hmac
from collections.abc import AsyncGenerator, Generator
from datetime import datetime, timedelta

import jwt
from fastapi import Depends, Header, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from loguru import logger
from redis import Redis
from sqlalchemy.ext.asyncio import AsyncSession

from backend.config.settings import get_settings
from backend.data_engine.collectors.yfinance_collector import YFinanceCollector
from backend.data_engine.transformers.feature_engineering import FeatureEngineer
from backend.db.models import get_async_session

settings = get_settings()

# Security
security = HTTPBearer()

# ============================================================================
# Database Dependencies
# ============================================================================


async def get_async_db() -> AsyncGenerator[AsyncSession, None]:
    """
    Get async database session for PostgreSQL/TimescaleDB.

    Yields async session with automatic commit/rollback and cleanup.

    Usage:
        @router.get("/items")
        async def get_items(db: AsyncSession = Depends(get_async_db)):
            result = await db.execute(select(Item))
            return result.scalars().all()

    Yields:
        AsyncSession: Database session
    """
    async for session in get_async_session():
        try:
            yield session
        except Exception as e:
            logger.error(f"Database session error: {e}")
            raise
        finally:
            await session.close()


# ============================================================================
# Cache Dependencies
# ============================================================================


def get_redis() -> Generator[Redis, None, None]:
    """
    Get Redis connection for caching and feature store.

    Redis is used for:
    - Feature embeddings (hot storage)
    - API rate limiting
    - Session management
    - Celery task queue

    Usage:
        @router.get("/cached")
        async def get_cached(redis: Redis = Depends(get_redis)):
            value = redis.get("key")
            return {"value": value}

    Yields:
        Redis: Redis connection
    """
    redis_client = Redis(
        host=settings.REDIS_HOST,
        port=settings.REDIS_PORT,
        db=settings.REDIS_DB,
        password=settings.REDIS_PASSWORD if settings.REDIS_PASSWORD else None,
        decode_responses=True,
        socket_connect_timeout=5,
        socket_keepalive=True,
    )
    try:
        yield redis_client
    finally:
        redis_client.close()


# ============================================================================
# Data Collector Dependencies
# ============================================================================


def get_yfinance_collector() -> YFinanceCollector:
    """
    Get YFinance data collector instance.

    Used for fetching market data from Yahoo Finance.

    Usage:
        @router.get("/market-data/{ticker}")
        async def get_market_data(
            ticker: str,
            collector: YFinanceCollector = Depends(get_yfinance_collector)
        ):
            data = await collector.collect_stock_data(ticker)
            return data

    Returns:
        YFinanceCollector: Collector instance
    """
    return YFinanceCollector()


def get_feature_engineer() -> FeatureEngineer:
    """
    Get feature engineering pipeline instance.

    Used for transforming raw market data into ML features.

    Usage:
        @router.post("/features")
        async def engineer_features(
            engineer: FeatureEngineer = Depends(get_feature_engineer)
        ):
            features = engineer.create_features(data)
            return features

    Returns:
        FeatureEngineer: Feature engineering instance
    """
    return FeatureEngineer()


# ============================================================================
# Authentication Dependencies
# ============================================================================


def create_access_token(data: dict, expires_delta: timedelta | None = None) -> str:
    """
    Create JWT access token.

    Args:
        data: Payload to encode in token
        expires_delta: Token expiration time

    Returns:
        str: Encoded JWT token
    """
    to_encode = data.copy()

    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)

    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)
    return encoded_jwt


def decode_access_token(token: str) -> dict:
    """
    Decode and verify JWT access token.

    Args:
        token: JWT token to decode

    Returns:
        dict: Decoded token payload

    Raises:
        HTTPException: If token is invalid or expired
    """
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except jwt.PyJWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> dict:
    """
    Get current authenticated user for JWT token.

    Usage:
        @router.get("/protected")
        async def protected_route(user: dict = Depends(get_current_user)):
            return {"user_id": user["user_id"]}

    Args:
        credentials: HTTP Bearer credentials

    Returns:
        dict: User information

    Raises:
        HTTPException: If authentication fails
    """
    token = credentials.credentials
    payload = decode_access_token(token)
    user_id = payload.get("sub")

    if user_id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return {"user_id": user_id, "payload": payload}


async def get_current_user_optional(authorization: str | None = Header(None)) -> dict | None:
    """
    Get current user if authenticated, None otherwise.

    Allows optional authentication for routes that work with or without auth.

    Usage:
        @router.get("/public-or-private")
        async def route(user: dict | None = Depends(get_current_user_optional)):
            if user:
                # Authenticated user logic
                pass
            else:
                # Public access logic
                pass

    Args:
        authorization: Authorization header

    Returns:
        dict | None: User information or None
    """
    if not authorization:
        return None

    try:
        # Extract token from "Bearer <token>"
        token = authorization.replace("Bearer ", "")
        payload = decode_access_token(token)
        user_id = payload.get("sub")
        return {"user_id": user_id, "payload": payload}
    except Exception:
        return None


# ============================================================================
# API Key Dependencies
# ============================================================================


async def verify_api_key(
    x_api_key: str | None = Header(None),
) -> str:
    """
    Verify API key from header.

    Supports both plain text keys and hashed keys for security.
    In production, always use hashed keys stored in environment.

    Usage:
        @router.get("/protected")
        async def protected(api_key: str = Depends(verify_api_key)):
            return {"status": "authenticated"}

    Args:
        x_api_key: API key from X-API-Key header

    Returns:
        str: Verified API key

    Raises:
        HTTPException: If API key is missing or invalid
    """
    # Skip in development if not required
    if not x_api_key:
        if settings.is_production or settings.REQUIRE_API_KEY:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="API key required in X-API-Key header",
            )
        return ""

    # Verify against configured keys
    if settings.API_KEYS or settings.API_KEY_HASHES:
        # Check plain text keys (development only)
        if x_api_key in settings.API_KEYS:
            return x_api_key

        # Check hashed keys (production)
        hashed_key = hashlib.sha256(x_api_key.encode("utf-8")).hexdigest()
        for stored_hash in settings.API_KEY_HASHES:
            if hmac.compare_digest(hashed_key, stored_hash):
                return x_api_key

        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
        )

    # No keys configured - only allow in development
    if settings.is_production or settings.REQUIRE_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="API key validation is not configured",
        )

    return x_api_key


# ============================================================================
# Rate Limiting Dependencies
# ============================================================================


class RateLimiter:
    """
    Redis-based rate limiter for API endpoints.

    Uses sliding window algorithm with Redis for distributed rate limiting.
    """

    def __init__(
        self,
        redis: Redis,
        max_requests: int = 100,
        window_seconds: int = 60,
    ):
        """
        Initialize rate limiter.

        Args:
            redis: Redis connection
            max_requests: Maximum requests per window
            window_seconds: Time window in seconds
        """
        self.redis = redis
        self.max_requests = max_requests
        self.window_seconds = window_seconds

    async def check_rate_limit(self, identifier: str) -> bool:
        """
        Check if identifier has exceeded rate limit.

        Args:
            identifier: Unique identifier (user_id, IP, API key hash)

        Returns:
            bool: True if within limit, False if exceeded
        """
        key = f"rate_limit:{identifier}"
        current_time = datetime.utcnow().timestamp()
        window_start = current_time - self.window_seconds

        # Remove old entries
        self.redis.zremrangebyscore(key, 0, window_start)

        # Count requests in window
        request_count = self.redis.zcard(key)

        if request_count >= self.max_requests:
            return False

        # Add current request
        self.redis.zadd(key, {str(current_time): current_time})

        # Set expiration
        self.redis.expire(key, self.window_seconds)

        return True


async def check_rate_limit(request: Request, redis: Redis = Depends(get_redis)) -> None:
    """
    Check rate limit for incoming request.

    Uses client IP or API key as identifier.

    Usage:
        @router.get("/limited", dependencies=[Depends(check_rate_limit)])
        async def limited_route():
            return {"status": "ok"}

    Args:
        request: FastAPI request object
        redis: Redis connection

    Raises:
        HTTPException: If rate limit exceeded
    """
    # Get identifier (API key or IP)
    api_key = request.headers.get("X-API-Key")
    if api_key:
        identifier = hashlib.sha256(api_key.encode()).hexdigest()
    else:
        identifier = request.client.host

    limiter = RateLimiter(redis, max_requests=100, window_seconds=60)

    if not await limiter.check_rate_limit(identifier):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded. Please try again later.",
            headers={"Retry-After": "60"},
        )


# ============================================================================
# Environment Dependencies
# ============================================================================


def check_production_environment() -> None:
    """
    Ensure operation is not in production.

    Use for dangerous operations that should only run in development.

    Usage:
        @router.delete("/dangerous", dependencies=[Depends(check_production_environment)])
        async def dangerous_operation():
            # Only allowed in development
            return {"deleted": True}

    Raises:
        HTTPException: If running in production
    """
    if settings.is_production:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Operation not allowed in production environment",
        )


def require_development_environment() -> None:
    """
    Require development environment.

    Alias for check_production_environment for semantic clarity.

    Raises:
        HTTPException: If not in development
    """
    check_production_environment()


# ============================================================================
# Health Check Dependencies
# ============================================================================


async def check_database_health(db: AsyncGenerator = Depends(get_async_db)) -> bool:
    """
    Check database connection health.

    Args:
        db: Database session

    Returns:
        bool: True if healthy
    """
    try:
        from sqlalchemy import text

        await db.execute(text("SELECT 1"))
        return True
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        return False


async def check_redis_health(redis: Redis = Depends(get_redis)) -> bool:
    """
    Check Redis connection health.

    Args:
        redis: Redis connection

    Returns:
        bool: True if healthy
    """
    try:
        redis.ping()
        return True
    except Exception as e:
        logger.error(f"Redis health check failed: {e}")
        return False
