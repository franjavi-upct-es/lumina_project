# backend/data_engine/collectors/social_scraper.py
"""Social-media scraper — intentionally minimal placeholder.

The architecture spec lists Reddit and Twitter as a *fourth* signal
source beyond OHLCV, news headlines, and the supply-chain graph. We
have deliberately kept this layer minimal until the semantic encoder
is trained and validated for two reasons:

1. **Signal-to-noise ratio.** Raw social posts without an NLP filter
   are pure noise; feeding them to the ingestion pipeline before the
   filter exists pollutes the training set.

2. **Provider stability.** Twitter's API terms have shifted twice in
   the last two years; pinning collector code to a moving target is
   wasted engineering.

This class therefore exists to:

* Lock in the database schema (see :class:`SocialPost`) so the rest
  of the system can be built around it.
* Provide a no-op ``collect_batch`` that downstream consumers can call
  safely.
* Emit exactly one ``WARNING`` log on first instantiation, so an
  operator who enables social collection in production sees an
  immediate reminder that no data is being collected.

The full implementation will live in a future commit once the semantic
encoder has been validated on news data and we have an NLP filter
robust enough to operate on the noisier social stream.
"""

from __future__ import annotations

from datetime import datetime

from loguru import logger
from pydantic import BaseModel, Field


class SocialPost(BaseModel):
    """Canonical schema for a single social-media post."""

    time: datetime
    post_id: str
    ticker: str
    platform: str
    author: str | None = None
    content: str
    engagement_score: float = Field(ge=0, default=0.0)


class SocialScraper:
    """No-op scraper that locks in the contract. See module docstring."""

    _warned: bool = False

    def __init__(self) -> None:
        if not SocialScraper._warned:
            logger.warning(
                "SocialScraper is a placeholder; no social posts will be collected. "
                "See the module docstring for rationale.",
            )
            SocialScraper._warned = True

    async def collect_batch(self, tickers: list[str], since: datetime) -> list[SocialPost]:
        """Always returns an empty list."""
        del tickers, since
        return []

    async def run(self) -> None:
        """No-op long-running task; returns immediately."""
        return
