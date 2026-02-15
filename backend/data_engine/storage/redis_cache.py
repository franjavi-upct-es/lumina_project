# backend/data_engine/storage/redis_cache.py
"""
Redis Cache for V3
==================

General-purpose Redis caching layer for:
- Deduplication of data collection
- Rate limiting
- Session management
- Temporary computations

Version: 3.0.0
"""

import hashlib
import json
from typing import Any

from loguru import logger
from redis.asyncio import Redis

from backend.config.settings import get_settings

settings = get_settings()


class RedisCache:
    """
    General-purpose Redis cache

    Provides caching, deduplication, and rate limiting.
    """

    def __init__(self, redis_url: str | None = None, default_ttl: int = 3600):
        """
        Initialize Redis cache

        Args:
            redis_url: Redis connection URL
            default_ttl: Default TTL in seconds
        """
        self.redis_url = redis_url or settings.REDIS_URL
        self.default_ttl = default_ttl
        self.redis: Redis | None = None

        logger.info("RedisCache initialized")

    async def connect(self):
        """Establish Redis connection"""
        if self.redis is None:
            self.redis = await Redis.from_url(
                self.redis_url,
                decode_responses=True,
                socket_connect_timeout=5,
            )
            logger.success("Connected to Redis")

    async def disconnect(self):
        """Close Redis connection"""
        if self.redis:
            await self.redis.close()
            self.redis = None

    async def get(self, key: str) -> Any | None:
        """Get value from cache"""
        try:
            await self.connect()
            value = await self.redis.get(key)

            if value:
                return json.loads(value)
            return None

        except Exception as e:
            logger.error(f"Error getting from cache: {e}")
            return None

    async def set(
        self,
        key: str,
        value: Any,
        ttl: int | None = None,
    ) -> bool:
        """Set value in cache"""
        try:
            await self.connect()

            serialized = json.dumps(value, default=str)
            ttl = ttl or self.default_ttl

            await self.redis.set(key, serialized, ex=ttl)
            return True

        except Exception as e:
            logger.error(f"Error setting cache: {e}")
            return False

    async def delete(self, key: str) -> bool:
        """Delete key from cache"""
        try:
            await self.connect()
            await self.redis.delete(key)
            return True

        except Exception as e:
            logger.error(f"Error deleting from cache: {e}")
            return False

    async def exists(self, key: str) -> bool:
        """Check if key exists"""
        try:
            await self.connect()
            return await self.redis.exists(key) > 0

        except Exception as e:
            logger.error(f"Error checking existence: {e}")
            return False

    async def increment(
        self,
        key: str,
        amount: int = 1,
        ttl: int | None = None,
    ) -> int:
        """Increment counter"""
        try:
            await self.connect()

            value = await self.redis.incr(key, amount)

            if ttl:
                await self.redis.expire(key, ttl)

            return value

        except Exception as e:
            logger.error(f"Error incrementing: {e}")
            return 0

    def hash_key(self, *args) -> str:
        """Generate hash key from arguments"""
        key_str = "|".join(str(arg) for arg in args)
        return hashlib.sha256(key_str.encode()).hexdigest()[:16]


# Global instance
_cache: RedisCache | None = None


def get_redis_cache() -> RedisCache:
    """Get global Redis cache"""
    global _cache
    if _cache is None:
        _cache = RedisCache()
    return _cache
