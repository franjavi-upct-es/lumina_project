# backend/api/dependencies.py
"""
FastAPI dependencies for dependency injection
Provides common dependencies like database sessions, authentication, rate limiting
"""

import hashlib
import hmac
from collections.abc import AsyncGenerator, Generator
from datetime import datetime, timedelta
from typing import Annotated

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
    Get database session

    Usage:
        @router.get("/items")
        async def get_items(db: AsyncSession = Depends(get_db)):
            ...
    """
    async for session in get_async_session():
        yield session


# ============================================================================
# Cache Dependencies
# ============================================================================


def get_redis() -> Generator[Redis, None, None]:
    """
    Get Redis connection

    Usage:
        @router.get("/cached")
        async def get_cached(redis: Redis = Depends(get_redis)):
            ...
    """
    redis_client = Redis.from_url(
        settings.REDIS_URL,
        decode_responses=True,
        socket_connect_timeout=5,
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
    Get YFinance data collector

    Usage:
        @router.get("/data/{ticker}")
        async def get_data(
            ticker: str,
            collector: YFinanceCollector = Depends(get_yfinance_collector)
        ):
            ...
    """
    return YFinanceCollector(rate_limit=settings.YFINANCE_RATE_LIMIT)


def get_feature_engineer() -> FeatureEngineer:
    """
    Get feature engineer

    Usage:
        @router.post("/features")
        async def compute_features(
            fe: FeatureEngineer = Depends(get_feature_engineer)
        ):
            ...
    """
    return FeatureEngineer()


# ============================================================================
# Authentication Dependencies
# ============================================================================


def create_access_token(data: dict, expires_delta: timedelta | None = None) -> str:
    """
    Create JWT access token

    Args:
        data: Data to encode in token
        expires_delta: Token expiration time

    Returns:
        Encoded JWT token
    """
    to_encode = data.copy()

    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)

    to_encode.update({"exp": expire})

    encoded_jwt = jwt.encode(
        to_encode,
        settings.SECRET_KEY,
        algorithm="HS256",
    )

    return encoded_jwt


def decode_access_token(token: str) -> dict:
    """
    Decode and verify JWT token

    Args:
        token: JWT token string

    Returns:
        Decoded token data

    Raises:
        HTTPException: If token is invalid or expired
    """
    try:
        payload = jwt.decode(
            token,
            settings.SECRET_KEY,
            algorithms=["HS256"],
        )
        return payload
    except jwt.ExpiredSignatureError as err:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired",
            headers={"WWW-Authenticate": "Bearer"},
        ) from err
    except jwt.JWTError as err:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        ) from err


async def get_current_user(
    credentials: Annotated[HTTPAuthorizationCredentials, Depends(security)],
) -> dict:
    """
    Get current authenticated user from JWT token

    Usage:
        @router.get("/protected")
        async def protected_route(
            user: dict = Depends(get_current_user)
        ):
            ...
    """
    token = credentials.credentials
    payload = decode_access_token(token)

    user_id = payload.get("sub")
    if user_id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
        )

    return {"user_id": user_id, "payload": payload}


async def get_current_user_optional(
    authorization: str | None = Header(None),
) -> dict | None:
    """
    Get current user if authenticated, None otherwise

    Usage:
        @router.get("/public-or-private")
        async def route(user: Optional[dict] = Depends(get_current_user_optional)):
            if user:
                # Authenticated
            else:
                # Public
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
# Rate Limiting Dependencies
# ============================================================================


class RateLimiter:
    """
    Simple rate limiter using Redis
    """

    def __init__(
        self,
        redis: Redis,
        max_requests: int = 100,
        window_seconds: int = 60,
    ):
        """
        Initialize rate limiter

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
        Check if identifier has exceeded rate limit

        Args:
            identifier: Unique identifier (user_id, IP, etc.)

        Returns:
            True if within limit, False if exceeded
        """
        key = f"rate_limit:{identifier}"

        try:
            # Increment counter
            current = self.redis.incr(key)

            # Set expiry on first request
            if current == 1:
                self.redis.expire(key, self.window_seconds)

            # Check limit
            if current > self.max_requests:
                logger.warning(f"Rate limit exceeded for {identifier}")
                return False

            return True

        except Exception as e:
            logger.error(f"Rate limit check failed: {e}")
            # Fail open - allow request if Redis unavailable
            return True

    def get_remaining_requests(self, identifier: str) -> int:
        """
        Get remaining requests for identifier

        Args:
            identifier: Unique identifier

        Returns:
            Number of remaining requests
        """
        key = f"rate_limit:{identifier}"

        try:
            current = int(self.redis.get(key) or 0)
            remaining = max(0, self.max_requests - current)
            return remaining
        except Exception:
            return self.max_requests


async def check_rate_limit(
    request: Request,
    redis: Annotated[Redis, Depends(get_redis)],
) -> None:
    """
    Rate limit dependency

    Usage:
        @router.get("/limited", dependencies=[Depends(check_rate_limit)])
        async def limited_route():
            ...
    """
    # Use IP address as identifier
    client_ip = request.client.host if request.client else "unknown"

    limiter = RateLimiter(
        redis=redis,
        max_requests=settings.RATE_LIMIT_MAX_REQUESTS,
        window_seconds=settings.RATE_LIMIT_WINDOW_SECONDS,
    )

    if not await limiter.check_rate_limit(client_ip):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded. Please try again later.",
            headers={"Retry-After": str(settings.RATE_LIMIT_WINDOW_SECONDS)},
        )


# ============================================================================
# Pagination Dependencies
# ============================================================================


class PaginationParams:
    """
    Pagination parameters
    """

    def __init__(
        self,
        page: int = 1,
        page_size: int = 50,
        max_page_size: int = 100,
    ):
        """
        Initialize pagination parameters

        Args:
            page: Page number (1-indexed)
            page_size: Items per page
            max_page_size: Maximum page size
        """
        self.page = max(1, page)
        self.page_size = min(max(1, page_size), max_page_size)
        self.offset = (self.page - 1) * self.page_size
        self.limit = self.page_size

    def get_response_metadata(self, total_items: int) -> dict:
        """
        Get pagination metadata for response

        Args:
            total_items: Total number of items

        Returns:
            Metadata dictionary
        """
        total_pages = (total_items + self.page_size - 1) // self.page_size

        return {
            "page": self.page,
            "page_size": self.page_size,
            "total_items": total_items,
            "total_pages": total_pages,
            "has_next": self.page < total_pages,
            "has_previous": self.page > 1,
        }


def get_pagination_params(
    page: int = 1,
    page_size: int = 50,
) -> PaginationParams:
    """
    Get pagination parameters

    Usage:
        @router.get("/items")
        async def get_items(
            pagination: PaginationParams = Depends(get_pagination_params)
        ):
            items = query.offset(pagination.offset).limit(pagination.limit)
            metadata = pagination.get_response_metadata(total_count)
    """
    return PaginationParams(page=page, page_size=page_size)


# ============================================================================
# Query Parameter Dependencies
# ============================================================================


class DateRangeParams:
    """
    Date range query parameters
    """

    def __init__(
        self,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        days: int | None = None,
    ):
        """
        Initialize date range

        Args:
            start_date: Start date
            end_date: End date
            days: Number of days (if start_date not provided)
        """
        # Set end_date
        if end_date is None:
            end_date = datetime.now()

        self.end_date = end_date

        # Set start_date
        if start_date is None:
            if days is not None:
                start_date = end_date - timedelta(days=days)
            else:
                start_date = end_date - timedelta(days=365)  # Default 1 year

        self.start_date = start_date

    def validate(self) -> None:
        """
        Validate date range

        Raises:
            HTTPException: If date range is invalid
        """
        if self.start_date > self.end_date:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="start_date must be before end_date",
            )

        # Limit date range to prevent abuse
        max_days = 365 * 10  # 10 years
        if (self.end_date - self.start_date).days > max_days:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Date range cannot exceed {max_days} days",
            )


def get_date_range(
    start_date: datetime | None = None,
    end_date: datetime | None = None,
    days: int | None = None,
) -> DateRangeParams:
    """
    Get validated date range parameters

    Usage:
        @router.get("/data")
        async def get_data(
            date_range: DateRangeParams = Depends(get_date_range)
        ):
            start = date_range.start_date
            end = date_range.end_date
    """
    params = DateRangeParams(start_date, end_date, days)
    params.validate()
    return params


# ============================================================================
# API Key Dependencies
# ============================================================================


async def verify_api_key(
    x_api_key: str | None = Header(None),
) -> str:
    """
    Verify API key from header

    Usage:
        @router.get("/protected")
        async def protected(api_key: str = Depends(verify_api_key)):
            ...
    """
    if not x_api_key:
        if settings.is_production or settings.REQUIRE_API_KEY:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="API key required",
            )
        return ""

    if settings.API_KEYS or settings.API_KEY_HASHES:
        if x_api_key in settings.API_KEYS:
            return x_api_key

        hashed_key = hashlib.sha256(x_api_key.encode("utf-8")).hexdigest()
        for stored_hash in settings.API_KEY_HASHES:
            if hmac.compare_digest(hashed_key, stored_hash):
                return x_api_key

        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
        )

    if settings.is_production or settings.REQUIRE_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="API key validation is not configured",
        )

    return x_api_key


# ============================================================================
# Environment Dependencies
# ============================================================================


def check_production_environment() -> None:
    """
    Ensure we're not in production

    Usage:
        @router.delete("/dangerous", dependencies=[Depends(check_production_environment)])
        async def dangerous_operation():
            # Only allowed in development
            ...
    """
    if settings.is_production:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Operation not allowed in production environment",
        )


def require_development_environment() -> None:
    """
    Require development environment
    """
    if not settings.is_development:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Operation only allowed in development environment",
        )


# ============================================================================
# Request Context Dependencies
# ============================================================================


class RequestContext:
    """
    Request context information
    """

    def __init__(self, request: Request):
        self.request = request
        self.client_ip = request.client.host if request.client else None
        self.user_agent = request.headers.get("user-agent")
        self.method = request.method
        self.url = str(request.url)
        self.timestamp = datetime.utcnow()

    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return {
            "client_ip": self.client_ip,
            "user_agent": self.user_agent,
            "method": self.method,
            "url": self.url,
            "timestamp": self.timestamp.isoformat(),
        }


def get_request_context(request: Request) -> RequestContext:
    """
    Get request context

    Usage:
        @router.get("/info")
        async def get_info(ctx: RequestContext = Depends(get_request_context)):
            logger.info(f"Request from {ctx.client_ip}")
    """
    return RequestContext(request)


# ============================================================================
# Validation Dependencies
# ============================================================================


def validate_ticker(ticker: str) -> str:
    """
    Validate stock ticker format

    Usage:
        @router.get("/stocks/{ticker}")
        async def get_stock(ticker: str = Depends(validate_ticker)):
            ...
    """
    ticker = ticker.upper().strip()

    if not ticker:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Ticker cannot be empty",
        )

    if len(ticker) > 10:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Ticker too long (max 10 characters)",
        )

    if not ticker.isalnum():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Ticker must be alphanumeric",
        )

    return ticker


# ============================================================================
# Caching Dependencies
# ============================================================================


class CacheManager:
    """
    Simple cache manager using Redis
    """

    def __init__(self, redis: Redis, ttl: int = 3600):
        """
        Initialize cache manager

        Args:
            redis: Redis connection
            ttl: Time to live in seconds
        """
        self.redis = redis
        self.ttl = ttl

    def get(self, key: str) -> str | None:
        """Get value from cache"""
        try:
            return self.redis.get(key)
        except Exception as e:
            logger.error(f"Cache get failed: {e}")
            return None

    def set(self, key: str, value: str) -> bool:
        """Set value in cache"""
        try:
            self.redis.setex(key, self.ttl, value)
            return True
        except Exception as e:
            logger.error(f"Cache set failed: {e}")
            return False

    def delete(self, key: str) -> bool:
        """Delete value from cache"""
        try:
            self.redis.delete(key)
            return True
        except Exception as e:
            logger.error(f"Cache delete failed: {e}")
            return False


def get_cache_manager(
    redis: Annotated[Redis, Depends(get_redis)],
) -> CacheManager:
    """
    Get cache manager

    Usage:
        @router.get("/cached")
        async def cached_route(cache: CacheManager = Depends(get_cache_manager)):
            cached = cache.get(key)
            if cached:
                return cached

            # Compute result
            result = expensive_operation()
            cache.set(key, result)
            return result
    """
    return CacheManager(redis, ttl=settings.REDIS_CACHE_TTL)
