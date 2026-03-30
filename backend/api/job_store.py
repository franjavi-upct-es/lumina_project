"""
Redis-backed job state store.
Replaces in-memory dictionaries for training and backtest job tracking.
"""

import json
from datetime import datetime, timedelta
from typing import Any

from loguru import logger
from redis import Redis

from backend.config.settings import get_settings

settings = get_settings()


class JobStore:
    """Redis-backed store for async job state."""

    def __init__(self, redis: Redis, prefix: str = "job", ttl_hours: int = 72):
        self.redis = redis
        self.prefix = prefix
        self.ttl = int(timedelta(hours=ttl_hours).total_seconds())

    def _key(self, job_id: str) -> str:
        return f"{self.prefix}:{job_id}"

    def set(self, job_id: str, data: dict[str, Any]) -> None:
        """Store job state with automatic expiration."""
        serializable = {}
        for k, v in data.items():
            if isinstance(v, datetime):
                serializable[k] = v.isoformat()
            else:
                serializable[k] = v
        self.redis.setex(self._key(job_id), self.ttl, json.dumps(serializable))

    def get(self, job_id: str) -> dict[str, Any] | None:
        """Retrieve job state. Returns None if expired or not found."""
        raw = self.redis.get(self._key(job_id))
        if raw is None:
            return None
        return json.loads(raw)

    def exists(self, job_id: str) -> bool:
        return self.redis.exists(self._key(job_id)) > 0

    def update(self, job_id: str, updates: dict[str, Any]) -> None:
        """Partial update of job state."""
        current = self.get(job_id)
        if current is None:
            logger.warning(f"Attempted to update non-existent job {job_id}")
            return
        current.update(updates)
        self.set(job_id, current)

    def delete(self, job_id: str) -> None:
        self.redis.delete(self._key(job_id))
