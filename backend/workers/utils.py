# backend/workers/utils.py
"""Shared async utilities for Celery workers."""

import asyncio
from collections.abc import Coroutine
from typing import Any, TypeVar

T = TypeVar("T")


def run_async(coro: Coroutine[Any, Any, T]) -> T:
    """
    Run an async coroutine safely from a synchronous Celery task.
    Resets the global DB engine before creating a new event loop to prevent
    'Future attached to a different loop' errors in Celery prefork workers.
    """
    from backend.db.models import close_db, reset_db_engine

    reset_db_engine()
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(coro)
    finally:
        try:
            loop.run_until_complete(close_db())
        except Exception:
            pass
        loop.close()
