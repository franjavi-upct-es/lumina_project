# backend/data_engine/collectors/news_stream.py
"""NewsAPI poller. No websocket available, so polling with persisted cursor."""

from __future__ import annotations

import asyncio
import json
import re
from datetime import UTC, datetime, timedelta

import httpx
from loguru import logger
from prometheus_client import Counter

from backend.config.constants import TARGET_TICKERS
from backend.config.settings import get_settings
from backend.data_engine.collectors.circuit_breaker import CircuitBreaker
from backend.data_engine.storage.redis_cache import RedisCache

NEWS_ARTICLES_COLLECTED = Counter(
    "news_articles_collected_total", "Articles fetched", labelnames=("source",)
)
NEWS_ARTICLES_PUBLISHED = Counter("news_articles_published_total", "Articles passing the filter")

_CURSOR_KEY = "news:cursor:newsapi"
_TICKER_PATTERNS = {
    t: re.compile(rf"(?:^|\W)\$?{re.escape(t)}(?:\W|$)", re.IGNORECASE) for t in TARGET_TICKERS
}


class NewsCollector:
    def __init__(self, redis: RedisCache, circuit_breaker: CircuitBreaker | None = None) -> None:
        self._redis = redis
        self._cb = circuit_breaker
        self._settings = get_settings()
        self._running = False
        self._client: httpx.AsyncClient | None = None

    async def run(self) -> None:
        self._running = True
        self._client = httpx.AsyncClient(timeout=20.0)
        interval = self._settings.NEWS_POLL_INTERVAL_SECONDS
        try:
            while self._running:
                if self._cb and self._cb.is_open:
                    await self._cb.wait_if_open()
                try:
                    await self._poll_once()
                except Exception as exc:
                    logger.error(f"News poll failed: {exc}")
                await asyncio.sleep(interval)
        finally:
            if self._client:
                await self._client.aclose()

    async def stop(self) -> None:
        self._running = False

    async def _poll_once(self) -> None:
        since = await self._get_cursor()
        q = " OR ".join(f'"{t}"' for t in list(TARGET_TICKERS)[:20])
        assert self._client
        resp = await self._client.get(
            "https://newsapi.org/v2/everything",
            params={
                "q": q,
                "from": since.isoformat(),
                "sortBy": "publishedAt",
                "language": "en",
                "pageSize": 100,
                "apiKey": self._settings.NEWSAPI_KEY,
            },
        )
        if resp.status_code == 429:
            logger.warning("NewsAPI rate limit; skipping this poll")
            return
        resp.raise_for_status()
        articles = resp.json().get("articles") or []
        latest_seen = since
        for art in articles:
            published_raw = art.get("publishedAt")
            if not published_raw:
                continue
            try:
                published = datetime.fromisoformat(published_raw.replace("Z", "+00:00"))
            except ValueError:
                continue
            if published <= since:
                continue
            if published > latest_seen:
                latest_seen = published
            await self._emit_if_relevant(art, published)
        if latest_seen > since:
            await self._set_cursor(latest_seen)

    async def _emit_if_relevant(self, art: dict, published: datetime) -> None:
        source = (art.get("source") or {}).get("name", "newsapi")
        NEWS_ARTICLES_COLLECTED.labels(source=source).inc()
        headline = art.get("title") or ""
        body = art.get("description") or art.get("content") or ""
        full_text = f"{headline} {body}"
        matched = [t for t, pat in _TICKER_PATTERNS.items() if pat.search(full_text)]
        if not matched:
            return
        payload = {
            "headline": headline,
            "body": body,
            "source": source,
            "url": art.get("url"),
            "published_at": published.isoformat(),
            "tickers": matched,
        }
        await self._redis.client.publish("channel:news.global", json.dumps(payload, default=str))
        NEWS_ARTICLES_PUBLISHED.inc()

    async def _get_cursor(self) -> datetime:
        raw = await self._redis.client.get(_CURSOR_KEY)
        if raw:
            try:
                return datetime.fromisoformat(raw.decode("utf-8"))
            except (ValueError, AttributeError):
                pass
        return datetime.now(UTC) - timedelta(hours=1)

    async def _set_cursor(self, ts: datetime) -> None:
        await self._redis.client.set(_CURSOR_KEY, ts.isoformat())
