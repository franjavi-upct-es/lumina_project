# backend/api/routes/data.py
"""Data routes: OHLCV, embeddings, news."""

from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, Depends

from backend.api.deps import get_redis, get_timescale, require_api_key
from backend.api.schemas import OHLCVResponse
from backend.data_engine.storage.redis_cache import EmbeddingKind, RedisCache
from backend.data_engine.storage.timescale import TimescaleStore

router = APIRouter(prefix="/api/data", tags=["data"])


@router.get(
    "/ohlcv/{ticker}", response_model=list[OHLCVResponse], dependencies=[Depends(require_api_key)]
)
async def get_ohlcv(
    ticker: str,
    start: datetime,
    end: datetime,
    ts: TimescaleStore = Depends(get_timescale),
) -> list[OHLCVResponse]:
    df = await ts.get_historical_window(ticker, start, end)
    return [
        OHLCVResponse(
            time=row["time"],
            open=row["open"],
            high=row["high"],
            low=row["low"],
            close=row["close"],
            volume=int(row["volume"]),
        )
        for row in df.iter_rows(named=True)
    ]


@router.get("/embedding/{kind}/{ticker}", dependencies=[Depends(require_api_key)])
async def get_embedding(kind: EmbeddingKind, ticker: str, redis: RedisCache = Depends(get_redis)):
    vec = await redis.get_embedding(kind, ticker)
    if vec is None:
        return {"ticker": ticker, "kind": kind, "vector": None}
    return {"ticker": ticker, "kind": kind, "vector": vec.tolist()}
