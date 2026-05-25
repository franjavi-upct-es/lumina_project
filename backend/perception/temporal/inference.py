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


_TFT_CHECKPOINT = "models/temporal/best.pt"


def _pick_device() -> str:
    if not torch.cuda.is_available():
        return "cpu"
    try:
        torch.zeros(1, device="cuda") + 1
        return "cuda"
    except RuntimeError as exc:
        logger.warning(f"CUDA reports available but probe failed ({exc}); falling back to CPU.")
        return "cpu"


async def _amain() -> None:
    import asyncio
    from pathlib import Path

    device = _pick_device()
    model = TemporalFusionTransformer()
    ckpt_path = Path(_TFT_CHECKPOINT)
    if ckpt_path.is_file():
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        logger.info(f"Loaded TFT weights from {ckpt_path} (device={device})")
    else:
        logger.warning(
            f"TFT checkpoint {ckpt_path} not found — running with random weights. "
            "Train via backend.perception.temporal.trainer first."
        )

    redis = RedisCache()
    await redis.connect()
    service = TFTInferenceService(model, redis, device=device)
    logger.info("TFTInferenceService starting (window={}).", service.window)
    stop_event = asyncio.Event()
    stop_tasks: set[asyncio.Task[None]] = set()

    import signal

    loop = asyncio.get_running_loop()

    def _request_stop() -> None:
        stop_event.set()
        task = asyncio.create_task(service.stop())
        stop_tasks.add(task)
        task.add_done_callback(stop_tasks.discard)

    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, _request_stop)

    try:
        await service.run(list(TARGET_TICKERS))
    finally:
        await redis.disconnect()


def main() -> int:
    import asyncio

    from backend.config.logging import configure_logging

    configure_logging()
    try:
        asyncio.run(_amain())
    except Exception:
        logger.exception("TFTInferenceService crashed")
        return 1
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
