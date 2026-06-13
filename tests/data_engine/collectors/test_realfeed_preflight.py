"""Real-feed preflight tests.

The price_stream and news_collector services must fail loudly and
*before* any Redis/network resource is acquired when their required API
key is missing. This keeps the ``real-feed`` profile honest: an operator
who forgot to set ``POLYGON_API_KEY`` gets a clear error on container
start, not a silent no-op.
"""

from __future__ import annotations

import pytest

from backend.config.settings import Settings
from backend.data_engine.collectors import news_stream, price_stream


@pytest.mark.anyio
async def test_price_stream_preflight_fails_without_polygon_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``POLYGON_API_KEY`` empty → ``_amain`` raises a clear RuntimeError."""
    monkeypatch.setattr(
        price_stream, "get_settings", lambda: Settings(_env_file=None, POLYGON_API_KEY="")
    )

    with pytest.raises(RuntimeError, match="POLYGON_API_KEY is required"):
        await price_stream._amain()


@pytest.mark.anyio
async def test_news_collector_preflight_fails_without_newsapi_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``NEWSAPI_KEY`` empty → ``_amain`` raises a clear RuntimeError."""
    monkeypatch.setattr(
        news_stream, "get_settings", lambda: Settings(_env_file=None, NEWSAPI_KEY="")
    )

    with pytest.raises(RuntimeError, match="NEWSAPI_KEY is required"):
        await news_stream._amain()
