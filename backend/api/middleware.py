# backend/api/middleware.py
"""Observability middleware for FastAPI."""

from __future__ import annotations

import asyncio
import time
import uuid

from fastapi import Request
from fastapi.responses import JSONResponse
from loguru import logger
from prometheus_client import Counter, Histogram
from starlette.middleware.base import BaseHTTPMiddleware

HTTP_REQUESTS = Counter(
    "http_requests_total",
    "Total HTTP requests",
    labelnames=("method", "path", "status"),
)
HTTP_LATENCY = Histogram(
    "http_request_latency_seconds",
    "HTTP request latency",
    labelnames=("method", "path"),
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
)


class ObservabilityMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get("x-request-id") or uuid.uuid4().hex
        request.state.request_id = request_id
        t0 = time.perf_counter()
        try:
            response = await call_next(request)
            status_code = response.status_code
        except Exception as exc:
            logger.error(f"[{request_id}] Unhandled: {exc}")
            raise
        duration = time.perf_counter() - t0
        path = request.url.path
        HTTP_REQUESTS.labels(request.method, path, str(status_code)).inc()
        HTTP_LATENCY.labels(request.method, path).observe(duration)
        response.headers["x-request-id"] = request_id
        response.headers["x-response-time-ms"] = f"{duration * 1000:.2f}"
        return response


class CongestionControlMiddleware(BaseHTTPMiddleware):
    """Simple concurrency limiter for multi-tenant safety.
    
    Prevents the backend from being overwhelmed by too many simultaneous
    requests, returning 429 Too Many Requests when saturated.
    """
    def __init__(self, app, max_concurrent_requests: int = 100):
        super().__init__(app)
        self.semaphore = asyncio.Semaphore(max_concurrent_requests)

    async def dispatch(self, request: Request, call_next):
        if self.semaphore.locked():
            logger.warning("Congestion control engaged: rejecting request")
            return JSONResponse(
                status_code=429,
                content={"detail": "Too many concurrent requests. Please try again later."},
            )
        async with self.semaphore:
            return await call_next(request)
