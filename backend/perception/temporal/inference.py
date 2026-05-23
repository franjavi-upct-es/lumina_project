# backend/perception/temporal/inference.py
"""Live TFT inference service."""

from __future__ import annotations

from collections import defaultdict, deque

import numpy as np
import torch
from loguru import logger

from backend.config.constants import OHLCV_WINDOW_MINUTES, TARGET_TICKERS
from backend.data_engine.storage.redis_cache import RedisCache
from backend.perception.common.embedding_writer import EmbeddingWriter
from backend.perception.temporal.tft_model import TemporalFusionTransformer


class TFTInferenceService:
    def __init__(
        self,
        model: TemporalFusionTransformer,
        redis: RedisCache,
        device: str = "cuda",
        window: int = OHLCV_WINDOW_MINUTES,
    ):
        self.model = model.to(device).eval()
        self.device = device
        self.window = window
        self.redis = redis
        self.writer = EmbeddingWriter(redis)
        self._buffers: dict[str, deque] = defaultdict(lambda: deque(maxlen=window))
        self._running = False

    async def run(self, tickers: list[str] | None = None) -> None:
        self._running = True
        tickers = tickers or list(TARGET_TICKERS)
        async for ticker, tick in self.redis.subscribe_ticks(tickers):
            if not self._running:
                break
            try:
                await self._handle(ticker, tick)
            except Exception as exc:
                logger.error(f"TFT inference error on {ticker}: {exc}")

    async def stop(self) -> None:
        self._running = False

    async def _handle(self, ticker: str, tick: dict) -> None:
        self._buffers[ticker].append(
            [
                float(tick["o"]),
                float(tick["h"]),
                float(tick["l"]),
                float(tick["c"]),
                float(tick["v"]),
            ]
        )
        buf = self._buffers[ticker]
        if len(buf) < self.window:
            return
        arr = np.array(buf, dtype=np.float32)
        arr = (arr - arr.min(0)) / (arr.max(0) - arr.min(0) + 1e-8)
        x = torch.from_numpy(arr).unsqueeze(0).to(self.device)
        with torch.no_grad():
            emb, _ = self.model(x)
        await self.writer.write("price", ticker, emb.cpu().numpy().squeeze(0))
